"""L2-SP starting-point regularization. (handout p4, Step 2)

Omega(theta, theta0) = (1/2) sum_j (theta_j - theta0_j)^2, added as lambda * Omega to the SFT
loss (lambda = 1e-3). Sum over ALL unique parameters including biases and normalization params,
counting tied parameters once; keep theta0 fixed; do NOT average Omega over parameters.

Unlike AdamW weight decay (reference 0), the reference here is the starting model theta0 (see
"What to Watch For", p5). theta0 is held in a frozen reference store (adds storage/transfer
overhead that the run must track).

Ties are counted once by iterating model.named_parameters() (remove_duplicate=True by default),
which yields each shared tensor a single time.

Memory / gradient design
------------------------
theta0 lives in PINNED CPU memory, not on the GPU: a full FP32 copy of a Full-FT model would
otherwise permanently occupy ~2.4 GB of GPU alongside the FP32 master params, optimizer states,
and activations — the headroom that caused Full-FT+L2-SP to OOM on the L4. Pinned host memory
makes the per-step H2D copy fast.

The penalty gradient is applied WITHOUT building an autograd graph. The gradient of lambda*Omega
w.r.t. theta_j is exactly lambda * (theta_j - theta0_j), so `accumulate_grad` just adds
`scale * lambda * (theta - theta0)` straight into each p.grad (under no_grad), where `scale` is the
AMP loss scale so the subsequent `scaler.unscale_` recovers the true-magnitude penalty gradient.
This is the identical gradient the old `scaler.scale(lambda*Omega).backward()` produced, but with
no vocab-sized graph and no GPU-resident reference tensor.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch


class L2SPPenalty:
    """Holds a frozen (pinned-CPU) copy of theta0 and injects the L2-SP gradient directly."""

    def __init__(self, model, lam: float, *, include_bias_and_norm: bool = True) -> None:
        self.lam = float(lam)
        self.include_bias_and_norm = include_bias_and_norm
        # Frozen FP32 reference in PINNED CPU memory (see module docstring), one entry per UNIQUE
        # parameter (ties counted once). Pinning may be unavailable off-GPU (e.g. a CPU-only
        # smoke); fall back to pageable host memory there.
        self._theta0: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not self._included(name, p):
                continue
            ref = p.detach().float().cpu().clone()
            try:
                ref = ref.pin_memory()
            except (RuntimeError, AssertionError):
                pass
            self._theta0[name] = ref
        self._bytes = sum(t.numel() * t.element_size() for t in self._theta0.values())

    def _included(self, name: str, p: torch.Tensor) -> bool:
        """Whether a parameter participates in Omega.

        include_bias_and_norm=True (handout default) => every parameter. Otherwise exclude biases
        and sub-2-dim params (normalization scales/shifts), matching the no-decay partition.
        """
        if self.include_bias_and_norm:
            return True
        return p.ndim >= 2 and not name.endswith(".bias")

    @torch.no_grad()
    def accumulate_grad(self, model, *, scale: float = 1.0) -> dict[str, float]:
        """Add the (scaled) L2-SP gradient into each parameter's .grad; return logging terms.

        Call AFTER the data-term backward and BEFORE `scaler.unscale_`: adds
        `scale * lambda * (theta - theta0)` in place into p.grad, which is exactly the gradient of
        lambda*Omega pre-scaled by the AMP loss scale so unscale recovers the true gradient. No
        autograd graph is built. theta0 is streamed one parameter at a time host->device (so the
        full FP32 reference is never resident on GPU at once); the per-update cost of that copy
        plus the Omega reduction is timed as the reference-transfer overhead.

        Args:
            scale: the current AMP loss scale (`scaler.get_scale()`); 1.0 when AMP is disabled.

        Returns:
            {"omega", "lambda_omega", "transfer_seconds"} for the per-update log.
        """
        omega = 0.0
        transfer_seconds = 0.0
        for name, p in model.named_parameters():
            ref = self._theta0.get(name)
            if ref is None:
                continue
            # Time the streamed copy + reduction together: reading the sum back as a Python float
            # forces the async H2D copy and the reduction to complete, so perf_counter captures the
            # real cost (dominated by the transfer) without an extra explicit synchronize per param.
            t0 = time.perf_counter()
            ref_dev = ref.to(p.device, non_blocking=True)
            diff = p.detach().float() - ref_dev            # theta - theta0, FP32
            sq_sum = float(diff.pow(2).sum())
            transfer_seconds += time.perf_counter() - t0

            omega += 0.5 * sq_sum
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.add_((scale * self.lam) * diff.to(p.grad.dtype))
        return {
            "omega": omega,
            "lambda_omega": self.lam * omega,
            "transfer_seconds": transfer_seconds,
        }

    @torch.no_grad()
    def terms(self, model) -> dict[str, float]:
        """Return {"omega", "lambda_omega"} for logging without touching gradients."""
        omega = 0.0
        for name, p in model.named_parameters():
            ref = self._theta0.get(name)
            if ref is None:
                continue
            diff = p.detach().float() - ref.to(p.device)
            omega += 0.5 * float(diff.pow(2).sum())
        return {"omega": omega, "lambda_omega": self.lam * omega}

    def reference_overhead_bytes(self) -> int:
        """Return the byte size of the theta0 reference store (transfer/storage overhead, p4)."""
        return int(self._bytes)


def build_l2sp(model, cfg: dict[str, Any]) -> Optional[L2SPPenalty]:
    """Construct an L2SPPenalty from a run config, or None when L2-SP is disabled (lambda <= 0).

    Lets Full FT (l2sp_lambda: 0.0) and Full FT + L2-SP share one training loop: the loop adds the
    penalty only when this returns a penalty object.
    """
    reg = cfg.get("regularization", {})
    lam = float(reg.get("l2sp_lambda", 0.0))
    if lam <= 0.0:
        return None
    return L2SPPenalty(
        model,
        lam,
        include_bias_and_norm=bool(reg.get("l2sp_include_bias_and_norm", True)),
    )
