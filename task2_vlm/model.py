"""Assemble the VLM: insert visual tokens into Qwen's chat, with LoRA on the LM. (handout p9 Step 1)

Builds the forward path cached-features -> connector -> visual tokens inserted into the user
message BEFORE the question, preserving Qwen's chat structure. Constructs matching labels,
attention masks, and position indices; masks visual tokens / prompts / padding in the loss. The
Qwen backbone is frozen (FP16); LoRA adapters (reuse task1_llm.adapters) are enabled in stages
2-3, targeting q/k/v/o_proj. Autograd must still reach the connector even though the backbone is
frozen ("What to Watch For", p9 -- do not disable grad on the whole forward pass).

For the no-image condition, visual tokens are omitted and masks/positions rebuilt (p8).
"""

from __future__ import annotations

from typing import Any, Optional


class SigLIPQwenVLM:
    """Frozen SigLIP features + trainable connector + frozen Qwen (+ optional LoRA)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """TODO: load frozen Qwen3-0.6B; build Connector; optionally attach LoRA (stages 2-3)."""
        raise NotImplementedError

    def build_inputs(self, text_tokens, visual_tokens, *, with_image: bool = True):
        """Insert visual tokens into the chat and build labels/attention mask/position ids.

        TODO: splice connector outputs into the embedding sequence before the question; set label
              ignore-index on visual/prompt/pad positions; rebuild masks/positions for no-image.
        """
        raise NotImplementedError

    def forward(self, batch):
        """Compute the example-weighted causal CE loss (or logits for eval).

        TODO: embed text, map features through connector, assemble, run Qwen, compute masked loss.
        """
        raise NotImplementedError

    def generate(self, batch, *, max_new_tokens: int, greedy: bool = True):
        """Greedy generation under an image condition (16-new-token cap at final eval).

        TODO: assemble inputs for the given image condition; call Qwen generate.
        """
        raise NotImplementedError


def sanity_checks(model, cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify feature shapes, frozen parameter sets, connector gradients, and split isolation.
    (p9 Step 1)

    TODO: assert SigLIP/Qwen frozen; connector (and LoRA in stages 2-3) require grad; features are
          [196,768]; no eval image id appears in training/replay pools. Return a report dict.
    """
    raise NotImplementedError
