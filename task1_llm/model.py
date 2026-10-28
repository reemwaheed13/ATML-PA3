"""Build Qwen3-0.6B and wire up the chosen fine-tuning method. (handout p3-p4)

Responsible for: loading the post-trained Qwen3-0.6B checkpoint (enable_thinking=False), enabling
gradient checkpointing, and configuring the method-specific parameterization:
  * full_ft / full_ft_l2sp: all params trainable, FP32 master copies.
  * lora / oplora: frozen FP16 backbone + FP32 adapters via task1_llm.adapters.

Also builds the AdamW parameter groups that exclude weight decay from biases and parameters with
fewer than two dimensions (p3 table).

Precision wiring (p3): Full-FT keeps FP32 *master* parameters — i.e. the model itself is loaded
in FP32 and autocast does the FP16 compute (common.precision). Adapter runs load the backbone in
FP16, freeze it, and add FP32 adapter parameters. So the load dtype follows precision.fp32_master.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from common import SEED
from .adapters import adapter_config_from_dict, attach_adapters, load_matched_init

log = logging.getLogger(__name__)

# Qwen3-0.6B architecture the diagnostics assume (README open-question #6: hidden=1024, 28 layers;
# the {0,13,27} diagnostic layers and the connector output dim depend on this). Verified, not
# hard-asserted, so a deliberately different model does not crash the loader.
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_NUM_LAYERS = 28

_ADAPTER_METHODS = ("lora", "oplora")
_FULL_METHODS = ("full_ft", "full_ft_l2sp")


def _compute_dtype(cfg: dict[str, Any]) -> torch.dtype:
    name = cfg.get("precision", {}).get("compute_dtype", "float16")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_base_model(cfg: dict[str, Any]):
    """Load Qwen3-0.6B + tokenizer; enable gradient checkpointing; set load dtype by method.

    Returns (model, tokenizer). The model is loaded in FP32 for Full-FT (its params ARE the FP32
    master copies) and in FP16 for adapter runs (frozen backbone). use_cache is disabled so it is
    compatible with gradient checkpointing.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg["model"]
    name = model_cfg["name"]
    fp32_master = bool(cfg.get("precision", {}).get("fp32_master", False))
    load_dtype = torch.float32 if fp32_master else _compute_dtype(cfg)

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=load_dtype)

    model.config.use_cache = False
    if model_cfg.get("gradient_checkpointing", False):
        # use_reentrant=False: required so gradients flow when the backbone (and thus the block
        # inputs) are frozen, as in the adapter runs.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    _verify_architecture(model)
    return model, tokenizer


def _verify_architecture(model) -> None:
    hidden = getattr(model.config, "hidden_size", None)
    layers = getattr(model.config, "num_hidden_layers", None)
    if hidden != EXPECTED_HIDDEN_SIZE or layers != EXPECTED_NUM_LAYERS:
        log.warning(
            "Qwen3 architecture (hidden=%s, layers=%s) differs from the assumed (%s, %s); "
            "diagnostics layer indices / connector dim may need updating.",
            hidden, layers, EXPECTED_HIDDEN_SIZE, EXPECTED_NUM_LAYERS,
        )


def configure_method(model, cfg: dict[str, Any]):
    """Apply the method in cfg["method"]; return (model, adapters_or_none, trainable_params).

    Full-FT: every parameter trainable (FP32 master). Adapters: freeze the whole backbone, attach
    LoRA/OPLoRA from a dedicated seeded CPU generator (so LoRA-16 and OPLoRA-16 draw bit-identical
    factors), and for OPLoRA optionally load LoRA-16's matched initial factors.
    """
    method = cfg["method"]

    if method in _FULL_METHODS:
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        return model, None, trainable

    if method in _ADAPTER_METHODS:
        for p in model.parameters():
            p.requires_grad_(False)                       # freeze the entire FP16 backbone

        adapter_cfg = adapter_config_from_dict(cfg["adapter"])
        targets = list(cfg["adapter"]["targets"])
        # Dedicated CPU generator, isolated from global RNG, seeded identically for every run so
        # matched init across LoRA-16 / OPLoRA-16 holds (see adapters.attach_adapters).
        generator = torch.Generator().manual_seed(SEED)
        adapters = attach_adapters(model, adapter_cfg, targets, generator=generator)

        if method == "oplora":
            source = cfg["adapter"].get("match_init_from")
            if source:
                load_matched_init(adapters, source)

        # Let gradients reach the adapters through the frozen input embeddings under grad ckpt.
        if cfg["model"].get("gradient_checkpointing", False):
            model.enable_input_require_grads()

        trainable = [p for p in model.parameters() if p.requires_grad]
        return model, adapters, trainable

    raise ValueError(f"Unknown method {method!r}")


def build_param_groups(model, cfg: dict[str, Any]) -> list[dict]:
    """Build AdamW param groups: exclude weight decay from biases and <2-dim params (p3 table)."""
    opt_cfg = cfg["train"]["optimizer"]
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    exclude = bool(opt_cfg.get("exclude_wd_no_decay", True))

    if not exclude:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "weight_decay": weight_decay}]

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No weight decay on biases or any parameter with fewer than two dimensions (norms, etc.).
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer_and_scheduler(param_groups, cfg: dict[str, Any]):
    """Return (AdamW optimizer, cosine scheduler with warmup). (p3 table)

    LR warmup count is read from the canonical train.lr_schedule.warmup_updates (with a legacy
    fallback to train.warmup_updates). num_training_steps is total_updates; note the LR scheduler
    is advanced on APPLIED optimizer steps only (common.precision), so with AMP skips it tracks the
    parameter trajectory rather than wall-clock iterations (see README "Step counting").
    """
    from transformers import get_cosine_schedule_with_warmup

    train_cfg = cfg["train"]
    opt_cfg = train_cfg["optimizer"]
    sched_cfg = train_cfg.get("lr_schedule", {})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(train_cfg["lr_peak"]),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )

    total_updates = int(train_cfg["total_updates"])
    warmup = int(sched_cfg.get("warmup_updates", train_cfg.get("warmup_updates", 0)))

    name = sched_cfg.get("name", "cosine")
    if name != "cosine":
        raise ValueError(f"Only the cosine schedule is implemented; got {name!r}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_updates
    )
    return optimizer, scheduler


def snapshot_params(model) -> dict[str, torch.Tensor]:
    """Capture a detached FP32 CPU copy of every parameter (theta0 for displacement). (p4)"""
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}


def full_model_displacement(model, theta0: dict[str, torch.Tensor]) -> float:
    """Return ||theta - theta0||_2 over all parameters (Full FT / L2-SP displacement). (p4)

    theta0 is a snapshot from snapshot_params (keyed by parameter name). Computed in FP32 on CPU.
    """
    total = 0.0
    for name, p in model.named_parameters():
        if name not in theta0:
            continue
        diff = p.detach().float().cpu() - theta0[name]
        total += float(diff.pow(2).sum())
    return math.sqrt(total)
