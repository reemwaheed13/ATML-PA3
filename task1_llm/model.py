"""Build Qwen3-0.6B and wire up the chosen fine-tuning method. (handout p3-p4)

Responsible for: loading the post-trained Qwen3-0.6B checkpoint (enable_thinking=False), enabling
gradient checkpointing, and configuring the method-specific parameterization:
  * full_ft / full_ft_l2sp: all params trainable, FP32 master copies.
  * lora / oplora: frozen FP16 backbone + FP32 adapters via task1_llm.adapters.

Also builds the AdamW parameter groups that exclude weight decay from biases and parameters with
fewer than two dimensions (p3 table).
"""

from __future__ import annotations

from typing import Any


def load_base_model(cfg: dict[str, Any]):
    """Load Qwen3-0.6B + tokenizer; enable gradient checkpointing; set eval/train dtypes.

    TODO: AutoModelForCausalLM / AutoTokenizer from cfg["model"]["name"]; enable gradient
          checkpointing; confirm hidden size + layer count match diagnostics assumptions.
    """
    raise NotImplementedError


def configure_method(model, cfg: dict[str, Any]):
    """Apply the method in cfg["method"]; return (model, adapters_or_none, trainable_params).

    TODO: dispatch on method; for adapters call attach_adapters with a seeded generator (and
          load_matched_init for OPLoRA); for full_ft set up FP32 master params.
    """
    raise NotImplementedError


def build_param_groups(model, cfg: dict[str, Any]) -> list[dict]:
    """Build AdamW param groups: exclude weight decay from biases and <2-dim params (p3 table).

    TODO: partition parameters into decay / no-decay groups.
    """
    raise NotImplementedError


def build_optimizer_and_scheduler(param_groups, cfg: dict[str, Any]):
    """Return (AdamW optimizer, cosine scheduler with 38 warmup updates). (p3 table)

    TODO: AdamW(betas, eps, wd); cosine schedule over total_updates with warmup.
    """
    raise NotImplementedError


def full_model_displacement(model, theta0) -> float:
    """Return ||theta - theta0||_2 over all parameters (Full FT / L2-SP displacement). (p4)

    TODO: flatten and compute the L2 norm of the parameter difference.
    """
    raise NotImplementedError
