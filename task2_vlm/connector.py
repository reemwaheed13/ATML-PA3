"""Visual connector: 2-layer MLP mapping SigLIP features to Qwen embedding space. (handout p6-p7)

    Z(I) = g_phi(F(I)),   F(I) in R^{196x768},   Z(I) in R^{196x1024}
Architecture: Linear 768 -> 1024, GELU, Linear 1024 -> 1024. Trainable throughout all stages.
The connector output magnitude is calibrated against text embeddings using training data only
(p9 Step 1); the rule + resulting scale are recorded.
"""

from __future__ import annotations

from typing import Any


class Connector:
    """Two-layer MLP connector (FP32 trainable params over the frozen backbone)."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024, out_dim: int = 1024) -> None:
        """TODO: Linear(in,hidden) -> GELU -> Linear(hidden,out); FP32 params."""
        raise NotImplementedError

    def forward(self, feats):
        """Map [*, 196, 768] features to [*, 196, 1024] visual tokens.

        TODO: apply the MLP (optionally scaled by the calibrated magnitude).
        """
        raise NotImplementedError


def calibrate_magnitude(connector, text_embeddings, sample_features) -> float:
    """Return a scale aligning connector-output magnitude with text-embedding magnitude.

    Uses TRAINING data only (p9 Step 1). The rule + scale are logged for reproducibility.

    TODO: compute e.g. ratio of RMS norms of text embeddings vs connector outputs; return scale.
    """
    raise NotImplementedError


def pool_2x2(feats):
    """Average each 2x2 block of the 14x14 grid -> 49 tokens (optional compression, p9 Step 7).

    TODO: reshape [*,196,768]->[*,14,14,768], average 2x2 blocks -> [*,7,7,768]->[*,49,768].
    """
    raise NotImplementedError
