"""Qualitative generations: 168 responses across prompts x conditions. (handout p5, Step 7)

Using the 24 prompts (8 questions x {plain, JSON, template}) from first_plot_questions.yaml,
generate one greedy response per prompt (thinking disabled, 512-new-token cap) for each of the
seven evaluation conditions: baseline, Full FT midpoint, Full FT final, L2-SP, LoRA-4, LoRA-16,
OPLoRA. Save all 168 responses grouped by question + format.

Then annotate the first and last underlying questions in plain-question order, across all three
formats and all seven conditions (42 responses): format compliance, coherence/relevance,
repetition, unnecessary mathematical framing, inappropriate content, with truncation + support
flags and short excerpts.
"""

from __future__ import annotations

from typing import Any


def load_prompts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the 24 prompts from the configured first_plot_questions.yaml path.

    NOTE: the file lives on DRIVE_ROOT/data; its path is a config field (not assumed in repo).
    TODO: read YAML; expand 8 questions x 3 formats into 24 prompt records with ids.
    """
    raise NotImplementedError


def generate_for_condition(cfg: dict[str, Any], run_id: str, update: int, prompts) -> list[dict]:
    """Greedy-generate one response per prompt for one condition (512-new-token cap).

    TODO: load checkpoint (merge adapters); apply chat template (enable_thinking=False); generate;
          record text + truncation flag.
    """
    raise NotImplementedError


def run(cfg: dict[str, Any]) -> None:
    """Produce all 168 generations and the 42-response annotation subset; save to results/.

    TODO: loop the seven conditions x 24 prompts; save grouped generations to DRIVE_ROOT and the
          annotation subset (machine-readable fields only; NO analysis prose) to results/task1/.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point. TODO: parse --config; call run."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
