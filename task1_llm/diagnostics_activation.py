"""Activation diagnostics: local update response R(z) on math vs non-math prompts.
(handout p5, Step 6)

Using the probe set (32 GSM8K test questions + 32 HellaSwag validation contexts, seed 6304),
register forward hooks to cache the STARTING model's down_proj INPUTS z at the last non-padding
prompt token before any answer continuation. Then, for the same checkpoints as the weight-space
diagnostics, reuse the cached z and compute per layer:
    R(z) = ||dW z||_2 / (||W0 z||_2 + 1e-8)
on math and non-math prompts. Compute in FP32. For Full FT, dW = W_trained - W0.

Reusing the same cached activations isolates the local weight change; it does not capture changes
elsewhere in the network ("What to Watch For", p5).
"""

from __future__ import annotations

from typing import Any


def cache_downproj_inputs(base_model, probe_prompts: list[dict], layers: list[int]) -> dict:
    """Run the STARTING model with forward hooks; cache down_proj input z per (layer, prompt).

    z is taken at the last non-padding prompt token before answer continuation.
    NOTE (TA-question): the "prompt token before answer continuation" convention for HellaSwag
    (candidate-likelihood) is pending; finalize before caching.

    TODO: register forward_pre_hooks on down_proj of target layers; run prompts; store z (FP32).
    """
    raise NotImplementedError


def response_ratio(delta_w, w0, z) -> float:
    """Return R(z) = ||dW z||_2 / (||W0 z||_2 + 1e-8).

    TODO: matrix-vector products in FP32; ratio of L2 norms.
    """
    raise NotImplementedError


def run(cfg: dict[str, Any]) -> None:
    """Cache starting-model activations once, then compute R(z) for every checkpoint/layer/prompt.

    Separates math (GSM8K) from non-math (HellaSwag) prompts in the output table.

    TODO: load probe IDs from splits/; cache z from the starting model; loop checkpoints; write
          results/task1/diagnostics_activation.json.
    """
    raise NotImplementedError
