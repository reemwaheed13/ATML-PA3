"""Task 3 dev/eval splits and the live-50 selection. (handout p11)

Within each dataset: sort the 1,000 question IDs lexicographically, permute with NumPy
default_rng(6304); first 200 are development, the remaining 800 are held-out evaluation. Keep all
rollouts of a question together. The first 50 MATH evaluation questions in this permutation are
the live-generation subset. Split IDs are saved under splits/task3 and reused.
"""

from __future__ import annotations

from typing import Any


def make_splits(question_ids: list[str], *, seed: int = 6304) -> dict[str, list[str]]:
    """Return {"dev": [...200], "eval": [...800]} via lexicographic sort + default_rng(seed) permute.

    TODO: sorted(question_ids); numpy.random.default_rng(seed).permutation; slice 200/800.
    """
    raise NotImplementedError


def live_math_50(eval_ids: list[str]) -> list[str]:
    """Return the first 50 MATH evaluation question IDs (in the permuted order) for the live run.

    TODO: take eval_ids[:50] for the MATH dataset.
    """
    raise NotImplementedError


def save_splits(dataset: str, splits: dict[str, Any]) -> None:
    """Persist split IDs to splits/task3/<dataset>_splits.json (write-once).

    TODO: json dump under splits/task3.
    """
    raise NotImplementedError
