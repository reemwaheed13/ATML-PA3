"""FP16 / FP32 precision policy shared by Task 1 and Task 2 training.

Both tasks specify the same recipe (Task 1 p3; Task 2 p8):
  * FP16 compute, no quantization, automatic mixed precision (autocast).
  * Dynamic gradient scaling (torch.cuda.amp.GradScaler).
  * Full-FT variants keep FP32 master copies of parameters; adapter/connector variants keep a
    FROZEN FP16 backbone with FP32 trainable parameters.
  * Optimizer states in FP32. Unscale the gradients BEFORE global-norm clipping.
  * Log any skipped optimizer steps (when the scaler detects inf/nan).

Centralizing this here keeps the two training loops from diverging on the subtle parts
(unscale-before-clip, FP32 master params, skipped-step accounting).

Skipped steps and the scheduler
--------------------------------
With dynamic loss scaling, a step whose unscaled gradients contain inf/nan is *skipped*: the
scaler does not call `optimizer.step()`, and `scaler.update()` shrinks the scale for next time.
We detect this by comparing the scale before and after `update()`: a strict decrease means the
step was skipped (a scale *increase* is the periodic growth of a healthy run, not a skip).

Design choice — we do **not** advance the LR scheduler on a skipped step. The scheduler is
indexed by *applied* optimizer updates, so the LR schedule tracks the parameter trajectory
rather than wall-clock iterations. A handful of early skips while the scaler calibrates would
otherwise silently shift the whole cosine schedule forward, changing the LR at every subsequent
update and making "1,250 updates" mean 1,250 *attempts* instead of 1,250 *applied* steps. Skips
are rare after warmup, so the two policies differ only slightly, but tying the scheduler to
applied steps is the reproducible choice and is what the handout's update counts refer to.
`optimizer_step` therefore steps the scheduler itself, only when the step was applied, and
returns `skipped` so the caller logs it and does not double-count the update.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

log = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class PrecisionConfig:
    """Precision knobs resolved from a run config.

    Attributes:
        compute_dtype: autocast dtype (float16 per handout).
        fp32_master: True for Full-FT (keep FP32 master params); False for adapter/connector
            runs that keep a frozen FP16 backbone + FP32 trainable params.
        grad_clip_norm: global-norm clip value (1.0 per handout).
    """

    compute_dtype: str = "float16"
    fp32_master: bool = False
    grad_clip_norm: float = 1.0


def torch_dtype(name: str) -> torch.dtype:
    """Map a config dtype string (e.g. "float16") to a torch.dtype."""
    try:
        return _DTYPES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown compute_dtype {name!r}; expected one of {sorted(_DTYPES)}")


def make_grad_scaler(enabled: bool = True, device_type: str = "cuda"):
    """Return a dynamic GradScaler.

    `enabled=False` yields a transparent no-op scaler (scale fixed at 1.0, never skips), which
    is what we use on CPU where FP16 AMP does not apply — keeping the training loop identical on
    CPU and GPU. Prefers the modern `torch.amp.GradScaler(device_type, ...)` API, falling back
    to `torch.cuda.amp.GradScaler` on older torch.
    """
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except (AttributeError, TypeError):
        from torch.cuda.amp import GradScaler  # pragma: no cover - old-torch fallback
        return GradScaler(enabled=enabled)


def autocast_context(cfg: PrecisionConfig, device_type: str = "cuda", enabled: bool = True):
    """Return the autocast context manager for the forward/loss computation.

    On CPU (or when `enabled=False`) this is a null context so the same loop runs un-autocast in
    FP32 — the handout's FP16 recipe only applies on the GPU.
    """
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=torch_dtype(cfg.compute_dtype))


def optimizer_step(
    optimizer,
    scaler,
    parameters,
    cfg: PrecisionConfig,
    *,
    scheduler=None,
    step: Optional[int] = None,
) -> dict[str, Any]:
    """Unscale, clip to global norm, step, update the scaler, and (if not skipped) step the LR.

    Order is load-bearing: scaler.unscale_(optimizer) -> clip_grad_norm_ -> scaler.step ->
    scaler.update. Compare the scaler scale before/after update() to detect a skipped (inf/nan)
    step; on a skip we do NOT advance `scheduler` (see module docstring) and we log a warning.
    Grads are zeroed after the step so the next accumulation window starts clean (and a skipped
    step's inf/nan grads don't leak forward).

    Args:
        parameters: the trainable parameters to clip (e.g. model.parameters()); materialized to
            a list so it can be both clipped and, if ever needed, reused.
        scheduler: optional LR scheduler; stepped only when the optimizer step was applied.
        step: optional update index, for the skip log message only.

    Returns:
        {"skipped": bool, "grad_norm": float, "scale": float} for JSONL logging.
    """
    params = [p for p in parameters]

    # Unscale BEFORE clipping so the global-norm clip sees true-magnitude gradients. On a
    # disabled scaler this is a no-op and clipping acts on the raw FP32 grads.
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)

    scale_before = scaler.get_scale()
    scaler.step(optimizer)          # applies optimizer.step() unless inf/nan was found
    scaler.update()                 # grows the scale, or shrinks it if this step was skipped
    scale_after = scaler.get_scale()

    # A strict decrease in scale means the scaler found inf/nan and skipped the optimizer step.
    skipped = scale_after < scale_before

    if skipped:
        log.warning(
            "AMP skipped optimizer step%s (inf/nan grads); scale %g -> %g; "
            "scheduler NOT advanced.",
            f" {step}" if step is not None else "",
            scale_before,
            scale_after,
        )
    elif scheduler is not None:
        scheduler.step()

    optimizer.zero_grad(set_to_none=True)

    return {
        "skipped": bool(skipped),
        "grad_norm": float(grad_norm),
        "scale": float(scale_after),
    }
