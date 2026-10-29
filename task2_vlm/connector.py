"""Visual connector: 2-layer MLP mapping SigLIP features to Qwen embedding space. (handout p6-p7)

    Z(I) = g_phi(F(I)),   F(I) in R^{196x768},   Z(I) in R^{196x1024}
Architecture: Linear 768 -> 1024, GELU, Linear 1024 -> 1024. Trainable throughout all stages.
The connector output magnitude is calibrated against text embeddings using training data only
(p9 Step 1); the rule + resulting scale are recorded.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Connector(nn.Module):
    """Two-layer MLP connector (FP32 trainable params over the frozen backbone).

    A fixed post-calibration `scale` (a registered buffer, not a trained parameter) multiplies the
    MLP output so visual-token magnitude matches text-embedding magnitude. It defaults to 1.0 and
    is set once by `calibrate_magnitude` using training data only; being a buffer, it is saved and
    restored with the connector weights.
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024, out_dim: int = 1024) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.float()  # FP32 trainable params (p8: FP32 connector over the frozen FP16 backbone)

    def project(self, feats: torch.Tensor) -> torch.Tensor:
        """Unscaled MLP output (used during magnitude calibration)."""
        return self.fc2(self.act(self.fc1(feats)))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Map [*, 196, 768] features to [*, 196, 1024] visual tokens, applying the calibrated scale."""
        return self.project(feats) * self.scale

    @torch.no_grad()
    def set_scale(self, scale: float) -> None:
        """Freeze the calibration scale into the connector (recorded for reproducibility)."""
        self.scale.fill_(float(scale))


@torch.no_grad()
def calibrate_magnitude(connector: Connector, text_embeddings: torch.Tensor,
                        sample_features: torch.Tensor) -> float:
    """Return a scale aligning connector-output magnitude with text-embedding magnitude.

    Rule (recorded, p9 Step 1): ratio of the mean per-token RMS norm of text embeddings to that of
    the UNSCALED connector outputs, over training-data samples only. Applying this scale makes the
    average visual-token magnitude match the average text-token magnitude. Since both live in the
    same 1024-dim space, the RMS ratio equals the L2-norm ratio.
    """
    z = connector.project(sample_features.to(connector.fc1.weight.dtype)).float()
    txt_rms = text_embeddings.float().pow(2).mean(dim=-1).sqrt().mean()
    con_rms = z.pow(2).mean(dim=-1).sqrt().mean().clamp_min(1e-8)
    return float(txt_rms / con_rms)


def pool_2x2(feats: torch.Tensor) -> torch.Tensor:
    """Average each 2x2 block of the 14x14 grid -> 49 tokens (optional compression, p9 Step 7).

    [*, 196, 768] -> [*, 14, 14, 768] -> [*, 7, 7, 768] (2x2 block mean) -> [*, 49, 768].
    """
    *lead, n, d = feats.shape
    g = int(round(n ** 0.5))
    if g * g != n or g % 2 != 0:
        raise ValueError(f"pool_2x2 expects a square grid with even side; got {n} tokens.")
    half = g // 2
    x = feats.reshape(*lead, half, 2, half, 2, d)
    x = x.mean(dim=(-4, -2))                      # average the two block axes -> [*, 7, 7, d]
    return x.reshape(*lead, half * half, d)
