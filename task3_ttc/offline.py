"""Offline selection experiments over candidate budgets. (handout p12-p13, Steps 1-3)

For budgets N in {1,2,4,8,16,32}, using ten seeded permutations of the 32 rollout IDs per
question (nested prefixes, orders reused across methods), evaluate selection rules and report
accuracy, oracle coverage, and generated-token cost (sum of retained token counts; FirstAns
counts only its first candidate). Average over orders within a question, then across questions.
Ties resolved by the earliest supporting rollout in the current order; RWC ties by average ranks.

Selection rules (p13):
  * Both datasets: FirstAns, self-consistency (one vote/rollout), highest mean log-prob, highest
    self-certainty, self-certainty + RWC.
  * MathVista additionally: vision-only and VisConf, each with highest-score and RWC.

Step 1 also decomposes selection failures (no correct candidate generated vs correct available
but missed). Step 3 examines within-question associations (length/score/correctness at N=32,
top/bottom quartiles, shortest-rollout length control) and the VisConf tokenwise-fusion ablation.
95% CIs via 1,000 paired question-level bootstrap on main accuracy differences (after averaging
over orders).
"""

from __future__ import annotations

from typing import Any, Callable


def generate_orders(n_rollouts: int, n_orders: int, *, seed: int) -> list[list[int]]:
    """Return `n_orders` permutations of range(n_rollouts), derived deterministically from seed.

    NOTE (TA-question #4): the exact seeding of the ten orders is unspecified; this derivation is
    recorded so results are reproducible.
    TODO: use common.seeding.spawn_generator(seed, ...) to produce independent permutations.
    """
    raise NotImplementedError


def select(rule: str, candidates: list[dict[str, Any]], order: list[int], n: int) -> dict[str, Any]:
    """Apply a selection rule to the first `n` candidates in `order`; return selected answer + cost.

    TODO: dispatch on rule (first_ans / self_consistency / highest_logprob / highest_self_certainty
          / self_certainty_rwc / vision_only_* / visconf_*); tie-break by earliest supporting
          rollout; compute retained-token cost.
    """
    raise NotImplementedError


def evaluate_budget(question: Any, rule: str, orders: list[list[int]], n: int) -> dict[str, float]:
    """Average a rule's outcome over orders for one question at budget n.

    TODO: for each order take the nested prefix of length n; select; grade; average accuracy,
          oracle coverage, token cost over orders.
    """
    raise NotImplementedError


def run(cfg: dict[str, Any]) -> None:
    """Run all rules x budgets for both datasets; write machine-readable results + bootstrap CIs.

    TODO: load joined data + splits; build orders; loop datasets/rules/budgets/questions; average;
          decompose failures; bootstrap accuracy differences; save under results/task3/.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry. TODO: parse --config; call run."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
