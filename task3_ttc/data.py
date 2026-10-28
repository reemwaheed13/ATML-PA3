"""Load/join supplied rollout tables; form answer groups + correctness labels. (handout p11-p12)

Each model folder has generations.parquet and probability_metrics.parquet (VLM also
attention_metrics.parquet); join on (dataset, sample_id, rollout_id). Fields include text, token
IDs, lengths, extracted answers, gold answers, sampled-token logp, and full-distribution kl_u_p
(raw model distributions, before sampling transforms); VLM attention is split into image /
prompt-text / generated-text groups. Images are not needed for offline analysis.

Correctness (p11-12): MATH equivalence via math-verify; MathVista via answer-type/choice/precision
rules from released question metadata (MC answers as option letters). Answer GROUPS are formed
WITHOUT consulting gold; the same equivalence rule is used for grading. Missing extractions map to
a single NO_ANSWER group, always graded incorrect. The scorer is frozen before final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class QuestionRollouts:
    """All 32 rollouts for one question, aligned across the joined tables."""

    dataset: str
    sample_id: str
    rollouts: list[dict[str, Any]]   # per-rollout: text, token_ids, length, extracted, logp, kl_u_p, (attn)
    gold: Any                        # for retrospective analysis only; NOT for selection/stopping


def load_joined(cfg: dict[str, Any], model_spec: dict[str, Any]) -> list[QuestionRollouts]:
    """Load + join the parquet files for one model/dataset; verify token-list alignment. (p12 Step 1)

    TODO: read parquet with pyarrow/pandas; join on (dataset, sample_id, rollout_id); assert
          per-token arrays (logp, kl_u_p, attention) align with token IDs excluding termination.
    """
    raise NotImplementedError


def equivalence(a: str, b: str, dataset: str, metadata: Optional[dict] = None) -> bool:
    """Answer equivalence: math-verify (MATH) or MathVista metadata rules (p11-12).

    TODO: dispatch on dataset; use math-verify for MATH; apply answer-type/choice/precision for
          MathVista using question metadata.
    """
    raise NotImplementedError


def answer_groups(rollout_answers: list[str], dataset: str, metadata: Optional[dict] = None):
    """Group extracted answers by equivalence WITHOUT gold; NO_ANSWER is its own group. (p12 Step 1)

    Returns a group assignment per rollout, used by voting/selection.

    TODO: cluster answers by `equivalence`; missing extraction -> NO_ANSWER group.
    """
    raise NotImplementedError


def correctness_labels(groups, gold, dataset, metadata: Optional[dict] = None) -> list[bool]:
    """Label each rollout correct/incorrect vs gold (retrospective; NO_ANSWER always incorrect).

    TODO: grade each group's representative answer against gold with the same equivalence rule.
    """
    raise NotImplementedError
