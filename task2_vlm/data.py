"""Task 2 data: COCO captions, annotation-derived grounding, natural VQAv2 pairs, and the seeded
curriculum samplers. (handout p7-p9)

Three components (all selected once with seed 6304; IDs saved under splits/task2 and reused):
  * COCO captions: 3 captions per training image (6,000 images -> 18,000), 1 per dev image.
  * Annotation-derived grounding (generated programmatically from COCO instances):
      presence: "Does this image contain [object]?" (yes/no), pairing an annotated-object image
                with one lacking that category.
      counting: "How many [objects] can you see?" pairing images with different annotated counts
                in 1..4.
  * Natural VQAv2 complementary pairs: keep pairs whose normalized consensus answers differ with
    >=8/10 annotator agreement; target = original consensus answer string; retain all 10 for
    scoring. Per-type counts yes/no:number:other as in configs/task2/data.yaml.

Curriculum sampling (p9): seeded random sampler; each DRAW contributes two examples (a complete
question pair, or two caption replays); no exact batch/quotas; repeats allowed. Stage mixtures
are in the config. The sampler position is part of the checkpoint (resumable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


# ----- selection / split freezing -----

def select_coco_captions(cfg: dict[str, Any]) -> dict[str, Any]:
    """Select COCO caption examples + dev images; save IDs to splits/task2. (write-once)

    TODO: pick 6,000 train2014 images x 3 captions + 96 val2014 dev images x 1; persist IDs.
          TA-question: which 3 of the 5 captions per image (document the rule used).
    """
    raise NotImplementedError


def build_annotation_grounding_pools(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate presence/counting pair pools from COCO instance annotations; freeze dev/test.

    TODO: build presence pairs (has-object vs lacks-category) and counting pairs (counts 1..4,
          different counts); hold out image-disjoint dev (24 pairs) + test (64 pairs); save IDs.
    """
    raise NotImplementedError


def select_natural_vqav2(cfg: dict[str, Any]) -> dict[str, Any]:
    """Select natural VQAv2 complementary pairs by the agreement/consensus rule; freeze splits.

    TODO: filter complementary pairs (consensus answers differ, >=8/10 agreement); hit the
          per-type counts; keep all 10 answers for scoring; save image/question/pair IDs.
    """
    raise NotImplementedError


# ----- examples / batching -----

@dataclass
class VLMExample:
    """One training example: cached visual features + chat-formatted text + supervised labels."""

    image_id: str
    component: str          # caption | presence | counting | natural_vqa (+ _replay)
    input_ids: Any
    labels: Any             # mask visual tokens, prompts, padding; supervise caption/answer + EOT
    attention_mask: Any
    pair_id: Any = None     # set for question pairs (kept together for scoring/bootstrap)


class CurriculumSampler:
    """Seeded sampler producing draws per the stage mixture; position is checkpointable."""

    def __init__(self, cfg: dict[str, Any], stage_cfg: dict[str, Any], *, seed: int = 6304) -> None:
        """TODO: build component pools; init a seeded RNG; expose get_state/set_state for resume."""
        raise NotImplementedError

    def draws(self, n_updates: int) -> Iterator[list[VLMExample]]:
        """Yield draws (each = a question pair or two caption replays) per the mixture probs.

        TODO: sample component by mixture; materialize two examples per draw; attach cached feats.
        """
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        """TODO: return RNG + position for checkpointing."""
        raise NotImplementedError

    def set_state(self, state: dict[str, Any]) -> None:
        """TODO: restore RNG + position on resume."""
        raise NotImplementedError
