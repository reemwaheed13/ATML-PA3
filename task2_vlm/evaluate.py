"""Task 2 grounding evaluation: development + final. (handout p8-p9)

Development (before training, every 500 updates, stage endpoints): caption reference loss +
reference-answer NLL on both VQA dev suites under correct / complementary / no-image, and the
image-swap penalty delta_NLL = NLL(y|q, I_partner) - NLL(y|q, I_correct) (exclude termination;
average token NLL within example before averaging examples).

Final (both test suites, greedy, 16-new-token cap, batch 8) under four image conditions
(correct / complementary swap / mismatched / no image). Scores: grounding normalized exact match
(presence/counting separate); natural VQAv2 soft consensus accuracy (yes-no/number/other
separate). Also pair accuracy, identical-answer rate, empty-answer rate, mean reference-answer
NLL, and generation time per answer. Correct-image advantages over complementary swaps in
accuracy and NLL, with 95% CIs via 4,000 bootstrap resamples of complete pairs (matched pair IDs).
"""

from __future__ import annotations

from typing import Any


def reference_nll(model, batch, *, exclude_termination: bool = True) -> float:
    """Teacher-forced per-example reference-answer NLL (averaged over tokens within example).

    TODO: forward with labels = reference answer; average token NLL excluding the EOT/termination.
    """
    raise NotImplementedError


def delta_nll(model, pair, *, condition_correct, condition_partner) -> float:
    """Return delta_NLL = NLL(y|q, I_partner) - NLL(y|q, I_correct) for one pair. (p8)

    TODO: compute reference_nll under correct vs partner image.
    """
    raise NotImplementedError


def run_development(model, cfg: dict[str, Any], update: int) -> dict[str, Any]:
    """Dev metrics at a checkpoint: caption loss, reference-answer NLL, delta_NLL per condition.

    TODO: iterate dev suites; compute metrics under correct/complementary/no-image; return dict.
    """
    raise NotImplementedError


def run_final(cfg: dict[str, Any]) -> dict[str, Any]:
    """Final-test evaluation of the Stage-3 model under the four image conditions; save results.

    TODO: generate (greedy, 16 tokens) per condition; score grounding + VQA (types separated);
          compute pair accuracy / identical-answer / empty-answer / NLL / time; bootstrap
          (4,000 resamples of complete pairs) for correct-image advantages; write results/task2/.
    """
    raise NotImplementedError


def bootstrap_pair_difference(per_pair_values, resamples: int = 4000, *, seed: int = 6304):
    """95% CI for a correct-vs-complementary difference by resampling complete pairs. (p9)

    TODO: resample pair IDs with replacement; recompute the matched difference; return CI.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry. TODO: parse --config; run development or final per args."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
