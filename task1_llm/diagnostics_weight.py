"""Weight-space diagnostics: subspace-interference ratio rho_k and relative magnitude eta.
(handout p5, Step 6)

For all five variants at final + Full FT at midpoint, on down_proj in layers 0, 13, 27
(zero-based):
    rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2        (fraction of dW in the protected subspace)
    eta   = ||dW||_F / ||W0||_F                       (relative update magnitude)
U_k is the leading output-subspace basis of the starting weight W0 (same bases as OPLoRA's
protection, k=128). Compute in FP32.

For Full FT variants, dW = W_trained - W0. For adapter variants, dW comes from the adapter's
delta_weight() (projected factors for OPLoRA).
"""

from __future__ import annotations

from typing import Any


def leading_output_basis(w0, k: int):
    """Return U_k: the top-k left singular vectors of W0 (FP32). Cached per (layer, weight).

    TODO: SVD of W0 in FP32; return U[:, :k].
    """
    raise NotImplementedError


def rho_k(delta_w, u_k) -> float:
    """Return ||U_k U_k^T dW||_F^2 / ||dW||_F^2.

    TODO: project dW onto U_k's column space; ratio of squared Frobenius norms.
    """
    raise NotImplementedError


def eta(delta_w, w0) -> float:
    """Return ||dW||_F / ||W0||_F.

    TODO: ratio of Frobenius norms.
    """
    raise NotImplementedError


def run(cfg: dict[str, Any]) -> None:
    """Compute rho_k and eta for every (checkpoint, layer) in the diagnostics config; save table.

    TODO: load each checkpoint's dW for the target layers; compute metrics; write a compact
          machine-readable table under results/task1/diagnostics_weight.json.
    """
    raise NotImplementedError
