"""L2-SP starting-point regularization. (handout p4, Step 2)

Omega(theta, theta0) = (1/2) sum_j (theta_j - theta0_j)^2, added as lambda * Omega to the SFT
loss (lambda = 1e-3). Sum over ALL unique parameters including biases and normalization params,
counting tied parameters once; keep theta0 fixed; do NOT average Omega over parameters.

Unlike AdamW weight decay (reference 0), the reference here is the starting model theta0 (see
"What to Watch For", p5). theta0 is held in a frozen reference store (adds storage/transfer
overhead that the run must track).
"""

from __future__ import annotations

from typing import Any


class L2SPPenalty:
    """Holds a frozen copy of theta0 and computes the L2-SP penalty over unique parameters."""

    def __init__(self, model, lam: float, *, include_bias_and_norm: bool = True) -> None:
        """Snapshot theta0 from `model` (counting tied params once) and store lambda.

        TODO: capture a frozen reference copy of each unique parameter; record reference-storage
              size for the resource logs.
        """
        raise NotImplementedError

    def penalty(self, model) -> Any:
        """Return Omega (unweighted). Caller adds lambda * Omega to the SFT loss.

        TODO: sum 0.5 * (theta - theta0)^2 over unique parameters; do not average.
        """
        raise NotImplementedError

    def terms(self, model) -> dict[str, float]:
        """Return {"omega": ..., "lambda_omega": ...} for logging throughout training (p4).

        TODO: compute Omega and lambda*Omega.
        """
        raise NotImplementedError

    def reference_overhead_bytes(self) -> int:
        """Return the byte size of the theta0 reference store (transfer/storage overhead, p4).

        TODO: sum reference tensor sizes.
        """
        raise NotImplementedError
