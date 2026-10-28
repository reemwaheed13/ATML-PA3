"""L2-SP starting-point regularization. (handout p4, Step 2)

Omega(theta, theta0) = (1/2) sum_j (theta_j - theta0_j)^2, added as lambda * Omega to the SFT
loss (lambda = 1e-3). Sum over ALL unique parameters including biases and normalization params,
counting tied parameters once; keep theta0 fixed; do NOT average Omega over parameters.

Unlike AdamW weight decay (reference 0), the reference here is the starting model theta0 (see
"What to Watch For", p5). theta0 is held in a frozen reference store (adds storage/transfer
overhead that the run must track).

Ties are counted once by iterating model.named_parameters() (remove_duplicate=True by default),
which yields each shared tensor a single time. The reference copy lives on the parameters' own
device so the per-step penalty needs no host<->device transfer; reference_overhead_bytes reports
its footprint for the resource logs.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


class L2SPPenalty:
    """Holds a frozen copy of theta0 and computes the L2-SP penalty over unique parameters."""

    def __init__(self, model, lam: float, *, include_bias_and_norm: bool = True) -> None:
        self.lam = float(lam)
        self.include_bias_and_norm = include_bias_and_norm
        # Frozen FP32 reference, one entry per UNIQUE parameter (ties counted once).
        self._theta0: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not self._included(name, p):
                continue
            self._theta0[name] = p.detach().float().clone()
        self._bytes = sum(t.numel() * t.element_size() for t in self._theta0.values())

    def _included(self, name: str, p: torch.Tensor) -> bool:
        """Whether a parameter participates in Omega.

        include_bias_and_norm=True (handout default) => every parameter. Otherwise exclude biases
        and sub-2-dim params (normalization scales/shifts), matching the no-decay partition.
        """
        if self.include_bias_and_norm:
            return True
        return p.ndim >= 2 and not name.endswith(".bias")

    def penalty(self, model) -> torch.Tensor:
        """Return Omega (unweighted, differentiable). Caller adds lambda * Omega to the SFT loss.

        Omega = sum_j 0.5 * (theta_j - theta0_j)^2, summed (NOT averaged) over the unique
        parameters snapshotted at construction.
        """
        omega: Optional[torch.Tensor] = None
        for name, p in model.named_parameters():
            ref = self._theta0.get(name)
            if ref is None:
                continue
            term = 0.5 * (p - ref).pow(2).sum()
            omega = term if omega is None else omega + term
        if omega is None:
            # No matching parameters (shouldn't happen); return a graph-free zero on model device.
            return torch.zeros((), device=next(model.parameters()).device)
        return omega

    def terms(self, model) -> dict[str, float]:
        """Return {"omega": ..., "lambda_omega": ...} for logging throughout training (p4)."""
        with torch.no_grad():
            omega = float(self.penalty(model))
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
