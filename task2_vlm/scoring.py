"""VQA answer normalization + soft consensus accuracy. (handout p9)

Natural VQAv2 is scored with soft consensus accuracy against all ten human answers:
    s(a_hat) = (1/10) * sum_{i=1..10} min(1, (1/3) * sum_{j != i} 1[a_hat == a_j])
Grounding questions use normalized exact match (presence and counting kept separate).

The answer normalization follows the official VQA evaluation code (see README "Attribution"),
applied consistently throughout training and scoring.
"""

from __future__ import annotations

from typing import Any


def normalize_answer(ans: str) -> str:
    """Apply the official VQA answer normalization (articles, punctuation, digits, contractions).

    Adapted from the official VQA eval code (attributed in README).
    TODO: implement the standard normalization steps.
    """
    raise NotImplementedError


def soft_consensus_accuracy(pred: str, answers: list[str]) -> float:
    """Return s(a_hat) over the ten normalized human answers.

    TODO: normalize pred + answers; apply the min(1, matches/3) averaged over the 10 held-out sets.
    """
    raise NotImplementedError


def normalized_exact_match(pred: str, target: str) -> float:
    """Return 1.0 if normalized pred == normalized target else 0.0 (grounding questions).

    TODO: normalize both and compare.
    """
    raise NotImplementedError
