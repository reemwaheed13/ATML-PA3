"""Shared plumbing for the Step-6 weight-space and activation diagnostics.

Both diagnostics compare, for down_proj in layers {0, 13, 27}, the *trained update* dW against
the original weight W0 and its leading singular subspace. The one thing they must agree on
bit-for-bit is how dW is built from a checkpoint, so that code lives here and is imported by
both `diagnostics_weight` and `diagnostics_activation`:

  * Full FT / L2-SP : dW = W_trained - W0   (W_trained read from the full checkpoint state_dict)
  * LoRA            : dW = (alpha/r) B A     (factors A, B read from the adapter checkpoint)
  * OPLoRA          : dW = (alpha/r) B~ A~   using the PROJECTED factors, with the same two-sided
                      projection the adapter applies at train time (bases = leading-k SVD of W0).

The projection and the (alpha/r) scaling mirror task1_llm.adapters.LoRAOPLoRALinear exactly
(._project / .delta_weight), so a diagnostic dW equals the dW that merge_into_backbone would add.
Everything is computed in FP32 on CPU.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch

from common.checkpoint import checkpoint_path, load_checkpoint
from common.config import load_yaml, run_id_from_config
from common.paths import task1_dir

from .adapters import AdapterConfig, adapter_config_from_dict

_FULL_METHODS = ("full_ft", "full_ft_l2sp")

# down_proj locators. Module path is e.g. "model.layers.13.mlp.down_proj"; the Full-FT checkpoint
# keys it with a trailing ".weight", adapter checkpoints store the trainable ".A" / ".B" factors.
_DOWN_PROJ_MODULE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj$")
_DOWN_PROJ_WEIGHT_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj\.weight$")
_DOWN_PROJ_A_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj\.A$")
_DOWN_PROJ_B_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj\.B$")


# ---------------------------------------------------------------------------------------------
# Config / checkpoint resolution
# ---------------------------------------------------------------------------------------------
def resolve_train_cfg(run: str, config_override: Optional[str] = None) -> dict[str, Any]:
    """Load the train config for a run id (default configs/task1/<run>.yaml); fill run_id.

    The diagnostics config lists checkpoints by run id (full_ft, lora_r16, ...), which are also
    the config filename stems, so the method + adapter settings come straight from that config.
    """
    rel = config_override or f"configs/task1/{run}.yaml"
    cfg = load_yaml(rel)
    cfg.setdefault("run_id", run_id_from_config(rel, cfg))
    return cfg


def checkpoint_file(run: str, update: int) -> str:
    """DRIVE_ROOT/task1/checkpoints/<run>/ckpt_step<update>.pt (where train.py writes them)."""
    return checkpoint_path(str(task1_dir("checkpoints") / run), int(update))


def base_model_name(train_cfg: dict[str, Any]) -> str:
    """The HF model id for the original weights W0 (all runs share Qwen/Qwen3-0.6B)."""
    return train_cfg["model"]["name"]


# ---------------------------------------------------------------------------------------------
# Original weights W0 and leading singular bases
# ---------------------------------------------------------------------------------------------
def down_proj_modules(model, layers) -> dict[int, Any]:
    """Map layer index -> down_proj submodule for the requested layers of a live model."""
    want = {int(layer) for layer in layers}
    found: dict[int, Any] = {}
    for name, module in model.named_modules():
        m = _DOWN_PROJ_MODULE_RE.search(name)
        if m and int(m.group(1)) in want:
            found[int(m.group(1))] = module
    missing = want - set(found)
    if missing:
        raise KeyError(f"down_proj not found for layers {sorted(missing)}")
    return found


def extract_w0(model, layers) -> dict[int, torch.Tensor]:
    """Return {layer: W0 (FP32 CPU, d_out x d_in)} from a live (original) model's down_proj."""
    return {
        layer: module.weight.detach().to(torch.float32).cpu().clone()
        for layer, module in down_proj_modules(model, layers).items()
    }


def load_original_down_proj(model_name: str, layers) -> dict[int, torch.Tensor]:
    """Load the original HF model in FP32 and return its down_proj W0 by layer, then free it."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    w0 = extract_w0(model, layers)
    del model
    return w0


def leading_bases(w0: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (U_k, V_k) from the FP32 SVD of W0 — the same bases OPLoRA protects.

    U_k is (d_out, k) leading left singular vectors; V_k is (d_in, k) leading right singular
    vectors. Mirrors adapters.LoRAOPLoRALinear: U[:, :k] and Vh[:k, :].T, full_matrices=False.
    """
    U, _S, Vh = torch.linalg.svd(w0.to(torch.float32), full_matrices=False)
    k = min(int(k), U.shape[1])
    U_k = U[:, :k].contiguous()
    V_k = Vh[:k, :].transpose(0, 1).contiguous()
    return U_k, V_k


def project_factors(A: torch.Tensor, B: torch.Tensor, U_k: torch.Tensor, V_k: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-sided OPLoRA projection: A~ = A - (A V_k) V_k^T, B~ = B - U_k (U_k^T B). (FP32)"""
    A_tilde = A - (A @ V_k) @ V_k.transpose(0, 1)
    B_tilde = B - U_k @ (U_k.transpose(0, 1) @ B)
    return A_tilde, B_tilde


# ---------------------------------------------------------------------------------------------
# dW construction (shared by both diagnostics)
# ---------------------------------------------------------------------------------------------
def _collect_by_layer(weights: dict[str, Any], regex: re.Pattern, want: set[int]
                      ) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for key, val in weights.items():
        m = regex.search(key)
        if m and int(m.group(1)) in want:
            out[int(m.group(1))] = val.detach().to(torch.float32).cpu()
    return out


def build_delta_weights(train_cfg: dict[str, Any], weights: dict[str, Any],
                        w0_by_layer: dict[int, torch.Tensor], layers,
                        ) -> dict[int, torch.Tensor]:
    """Build {layer: dW (FP32 CPU, d_out x d_in)} for one checkpoint's down_proj updates.

    `weights` is the checkpoint's `weights` dict (full state_dict for Full-FT; trainable adapter
    factors for LoRA/OPLoRA). `w0_by_layer` is the original down_proj weights (needed for the
    Full-FT difference and for the OPLoRA projection bases). This is the single definition of dW
    shared by the weight-space and activation diagnostics.
    """
    method = train_cfg["method"]
    want = {int(layer) for layer in layers}
    dW: dict[int, torch.Tensor] = {}

    if method in _FULL_METHODS:
        trained = _collect_by_layer(weights, _DOWN_PROJ_WEIGHT_RE, want)
        missing = want - set(trained)
        if missing:
            raise KeyError(f"Full-FT checkpoint missing down_proj weights for layers {sorted(missing)}")
        for layer in want:
            dW[layer] = trained[layer] - w0_by_layer[layer]
        return dW

    # Adapter runs (LoRA / OPLoRA): reconstruct dW from the stored factors.
    adapter_cfg: AdapterConfig = adapter_config_from_dict(train_cfg["adapter"])
    scaling = adapter_cfg.alpha / adapter_cfg.rank
    A_by = _collect_by_layer(weights, _DOWN_PROJ_A_RE, want)
    B_by = _collect_by_layer(weights, _DOWN_PROJ_B_RE, want)
    missing = want - (set(A_by) & set(B_by))
    if missing:
        raise KeyError(f"Adapter checkpoint missing A/B factors for layers {sorted(missing)}")

    for layer in want:
        A, B = A_by[layer], B_by[layer]
        if adapter_cfg.protect_k > 0:
            U_k, V_k = leading_bases(w0_by_layer[layer], adapter_cfg.protect_k)
            A, B = project_factors(A, B, U_k, V_k)
        dW[layer] = scaling * (B @ A)
    return dW


def load_checkpoint_weights(run: str, update: int) -> dict[str, Any]:
    """Load just the `weights` dict from a run's checkpoint at `update`."""
    return load_checkpoint(checkpoint_file(run, update)).weights


# ---------------------------------------------------------------------------------------------
# Smoke trimming (shared)
# ---------------------------------------------------------------------------------------------
def apply_smoke(cfg: dict[str, Any], checkpoints: list[dict], layers: list[int]
                ) -> tuple[list[dict], list[int]]:
    """Trim to a fast subset for Colab smoke tests: one layer + the first `max_checkpoints`."""
    sm = cfg.get("smoke", {}) or {}
    layers = list(sm.get("layers", layers[:1]))
    checkpoints = checkpoints[: int(sm.get("max_checkpoints", 2))]
    return checkpoints, layers
