"""Shared LoRA / OPLoRA adapter module with a `protect_k` option. (handout p4, Steps 3-4)

Design decision (approved): a single custom module serves both LoRA and OPLoRA.
  * protect_k == 0  -> plain LoRA:   W = W0 + (alpha/r) B A
  * protect_k >  0  -> OPLoRA:       orthogonally project the correction off the leading singular
                                     subspaces of W0 before applying it.

Why custom (not PEFT): OPLoRA must reuse LoRA-16's *initial unprojected* A/B factors bit-for-bit
(Step 4). Owning the random draw makes that trivial (same seeded generator, same order); with
PEFT we'd have to reproduce its internal init RNG and still subclass its Linear for the projected
forward. See README "Attribution" and the design note.

OPLoRA math (Step 4): for a starting weight W0 = U S V^T, with leading bases U_k, V_k (rank k):
    B_tilde = B - U_k (U_k^T B)
    A_tilde = A - (A V_k) V_k^T
    dW      = (alpha / r) B_tilde A_tilde
Bases computed once in FP32; use EFFECTIVE factors (not dense d_out x d_out / d_in x d_in
projection matrices); export the projected factors for inference.

Precision: adapters are FP32 parameters over a frozen FP16 backbone (p3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AdapterConfig:
    rank: int
    alpha: int
    dropout: float
    protect_k: int = 0            # 0 => plain LoRA
    init_a: str = "random"
    init_b: str = "zeros"
    trainable_bias: bool = False
    basis_dtype: str = "float32"


class LoRAOPLoRALinear:
    """Wraps a frozen base nn.Linear with an FP32 low-rank correction (LoRA or OPLoRA).

    Subclasses/wraps a target Linear (q/k/v/o/gate/up/down_proj). Holds:
      * frozen base weight W0 (FP16),
      * FP32 factors A (r x d_in), B (d_out x r),
      * for OPLoRA: cached leading bases U_k (d_out x k), V_k (d_in x k) computed once in FP32.
    """

    def __init__(self, base_linear, cfg: AdapterConfig, *, generator=None) -> None:
        """Attach adapter factors to `base_linear`.

        Args:
            generator: seeded torch.Generator used to draw A (and zero B). Passing the SAME
                generator state to LoRA-16 and OPLoRA-16 yields bit-identical initial factors.

        TODO: store frozen W0; draw A from `generator` (init_a), set B to zeros (init_b); register
              FP32 parameters; if protect_k>0, compute+cache U_k,V_k from SVD of W0 (FP32).
        """
        raise NotImplementedError

    def forward(self, x):
        """Return base(x) + scaled low-rank correction.

        For OPLoRA, apply the effective-factor projection (B_tilde, A_tilde) rather than forming
        dense projection matrices.

        TODO: implement LoRA and OPLoRA forward paths; apply dropout to the LoRA input.
        """
        raise NotImplementedError

    def effective_factors(self):
        """Return the factors used for inference/merge.

        LoRA: (A, B). OPLoRA: the projected (A_tilde, B_tilde) to export for inference (Step 4).

        TODO: compute and return effective factors.
        """
        raise NotImplementedError

    def delta_weight(self):
        """Return dW = (alpha/r) * B_eff @ A_eff as a dense matrix (for diagnostics / merge check).

        TODO: build dW from effective factors. Used by merge and by diagnostics_weight.
        """
        raise NotImplementedError


def attach_adapters(model, cfg: AdapterConfig, target_names: list[str], *, generator=None) -> dict:
    """Replace each targeted Linear in every transformer layer with a LoRAOPLoRALinear.

    Returns a handle mapping module path -> adapter, so matched-init and export can be verified.

    TODO: walk named_modules; for each name ending in a target, wrap it; return the handle.
    """
    raise NotImplementedError


def load_matched_init(adapters: dict, source_run_id: str) -> None:
    """Load initial unprojected A/B factors from another run (OPLoRA reuses LoRA-16's). (Step 4)

    TODO: load the saved initial factors for `source_run_id` and copy into the adapters, asserting
          shapes match and that they are copied BEFORE any optimizer step (bit-identical init).
    """
    raise NotImplementedError


def merge_into_backbone(model, adapters: dict) -> Any:
    """Return a fresh backbone copy with adapters merged (projected factors for OPLoRA). (Step 5)

    Used for evaluation. Caller verifies merged vs unmerged logits agree on a small batch.

    TODO: deep-copy the FP16 backbone; add each adapter's dW into the corresponding weight.
    """
    raise NotImplementedError


def export_initial_factors(adapters: dict, path: str) -> None:
    """Persist the initial unprojected A/B factors (so OPLoRA can match LoRA-16's init).

    TODO: save factor tensors keyed by module path.
    """
    raise NotImplementedError
