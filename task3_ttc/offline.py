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
but missed). 95% CIs via 1,000 paired question-level bootstrap on main accuracy differences (after
averaging over orders).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import numpy as np

from common.config import build_arg_parser, load_yaml
from common.paths import REPO_ROOT, ensure_dir
from common.seeding import spawn_generator
from . import data as t3data
from . import signals
from .splits import make_splits


# =================================================================================================
# Orders + per-question precomputation
# =================================================================================================

def generate_orders(n_rollouts: int, n_orders: int, *, seed: int) -> list[list[int]]:
    """Return `n_orders` independent permutations of range(n_rollouts), derived from seed.

    NOTE (TA-question #4): the exact seeding of the ten orders is unspecified; we derive each from
    an independent sub-stream of `seed` (common.seeding.spawn_generator) and record it so results
    are reproducible.
    """
    return [list(spawn_generator(seed, i).permutation(n_rollouts)) for i in range(n_orders)]


def _question_stats(q: t3data.QuestionRollouts, *, has_vision: bool,
                    metadata=None) -> dict[str, Any]:
    """Precompute groups/correctness and per-rollout signals (L/K/V + token length) for a question."""
    rollouts = q.rollouts
    metadata = q.metadata if metadata is None else metadata   # MathVista equivalence rules
    answers = [r.get("extracted") for r in rollouts]
    groups = t3data.answer_groups(answers, q.dataset, metadata)
    correct = t3data.correctness_labels(groups, q.gold, q.dataset, answers, metadata)
    group_correct: dict[int, bool] = {}
    for gid, ok in zip(groups, correct):
        group_correct[gid] = group_correct.get(gid, False) or ok
    return {
        "groups": groups,
        "group_correct": group_correct,
        "length": [int(r.get("length") or len(r.get("logp", []))) for r in rollouts],
        "L": [signals.mean_logprob(r) for r in rollouts],
        "K": [signals.self_certainty(r) for r in rollouts],
        "V": [signals.visual_engagement(r) for r in rollouts] if has_vision else [0.0] * len(rollouts),
        "rollouts": rollouts,
    }


# =================================================================================================
# Selection rules
# =================================================================================================

def _vote(groups: list[int]) -> int:
    """Majority group over a prefix; ties -> earliest supporting rollout (first in the prefix)."""
    counts = Counter(groups)
    best = max(counts.values())
    for g in groups:                                   # earliest occurrence among tied leaders
        if counts[g] == best:
            return g
    return groups[0]


def _argmax_group(groups: list[int], scores: list[float]) -> int:
    """Group of the single highest-scoring candidate; ties -> earliest in the prefix order."""
    best_i, best_s = 0, scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best_s:
            best_i, best_s = i, scores[i]
    return groups[best_i]


def select(rule: str, qs: dict[str, Any], order: list[int], n: int) -> dict[str, Any]:
    """Apply a selection rule to the first `n` candidates in `order`; return group, cost, oracle."""
    idx = [int(i) for i in order[:n]]
    groups = [qs["groups"][i] for i in idx]

    if rule == "first_ans":
        chosen = groups[0]
        cost = qs["length"][idx[0]]
    else:
        cost = sum(qs["length"][i] for i in idx)        # all consumed candidates are retained
        if rule == "self_consistency":
            chosen = _vote(groups)
        elif rule == "highest_logprob":
            chosen = _argmax_group(groups, [qs["L"][i] for i in idx])
        elif rule == "highest_self_certainty":
            chosen = _argmax_group(groups, [qs["K"][i] for i in idx])
        elif rule == "self_certainty_rwc":
            chosen = signals.rwc_winner(groups, [qs["K"][i] for i in idx])
        elif rule == "vision_only_highest":
            chosen = _argmax_group(groups, [qs["V"][i] for i in idx])
        elif rule == "vision_only_rwc":
            chosen = signals.rwc_winner(groups, [qs["V"][i] for i in idx])
        elif rule in ("visconf_highest", "visconf_rwc"):
            z = signals.visconf([qs["rollouts"][i] for i in idx])
            chosen = (_argmax_group(groups, z) if rule.endswith("highest")
                      else signals.rwc_winner(groups, z))
        else:
            raise ValueError(f"Unknown rule {rule!r}")

    oracle = any(qs["group_correct"].get(qs["groups"][i], False) for i in idx)
    return {"group": chosen, "correct": bool(qs["group_correct"].get(chosen, False)),
            "cost": int(cost), "oracle": bool(oracle),
            "distinct": len({g for g in groups if g != 0})}


def evaluate_budget(qs: dict[str, Any], rule: str, orders: list[list[int]], n: int) -> dict[str, float]:
    """Average a rule's outcome over orders for one question at budget n."""
    acc = ora = cost = dist = 0.0
    fail_no_correct = fail_missed = 0.0
    for order in orders:
        r = select(rule, qs, order, n)
        acc += r["correct"]
        ora += r["oracle"]
        cost += r["cost"]
        dist += r["distinct"]
        if not r["correct"]:
            if not r["oracle"]:
                fail_no_correct += 1          # no correct candidate generated
            else:
                fail_missed += 1              # correct available but missed
    k = len(orders)
    return {"accuracy": acc / k, "oracle_coverage": ora / k, "cost": cost / k,
            "distinct": dist / k, "fail_no_correct": fail_no_correct / k,
            "fail_missed": fail_missed / k}


# =================================================================================================
# Bootstrap
# =================================================================================================

def _bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# =================================================================================================
# Run
# =================================================================================================

def run(cfg: dict[str, Any]) -> None:
    """Run all rules x budgets for both datasets; write machine-readable results + bootstrap CIs."""
    budgets = [int(b) for b in cfg["offline"]["budgets"]]
    n_orders = int(cfg["offline"]["n_orders"])
    order_seed = int(cfg["offline"]["order_seed"])
    resamples = int(cfg["report"]["bootstrap_resamples"])
    both_rules = list(cfg["comparisons"]["both"])
    mv_extra = list(cfg["comparisons"]["mathvista_extra"])

    results: dict[str, Any] = {}
    for spec in cfg["data"]["models"]:
        dataset = spec["dataset"]
        has_vision = spec.get("type") == "vlm"
        questions = t3data.load_joined(cfg, spec)
        qmap = {q.sample_id: q for q in questions}
        split = make_splits(list(qmap), seed=int(cfg["splits"]["seed"]),
                            dev_size=int(cfg["splits"]["dev_size"]))
        eval_ids = split["eval"]
        # Smoke runs can cap the evaluated questions (offline.limit_questions); 0/absent = all.
        limit_q = int(cfg["offline"].get("limit_questions", 0) or 0)
        if limit_q:
            eval_ids = eval_ids[:limit_q]

        n_roll = max(len(q.rollouts) for q in questions)
        orders = generate_orders(n_roll, n_orders, seed=order_seed)
        stats = {sid: _question_stats(qmap[sid], has_vision=has_vision) for sid in eval_ids}

        rules = both_rules + (mv_extra if has_vision else [])
        ds_res: dict[str, Any] = {}
        # Per-question accuracy arrays (averaged over orders) for bootstrap vs self-consistency.
        per_q_acc: dict[str, dict[int, np.ndarray]] = {r: {} for r in rules}

        for rule in rules:
            rule_res = {}
            for N in budgets:
                rows = [evaluate_budget(stats[sid], rule, _prefix_orders(orders, N), N)
                        for sid in eval_ids]
                acc = np.array([r["accuracy"] for r in rows])
                per_q_acc[rule][N] = acc
                rule_res[N] = {
                    "accuracy": float(acc.mean()),
                    "oracle_coverage": float(np.mean([r["oracle_coverage"] for r in rows])),
                    "generated_token_cost": float(np.mean([r["cost"] for r in rows])),
                    "distinct_answer_groups": float(np.mean([r["distinct"] for r in rows])),
                    "fail_no_correct": float(np.mean([r["fail_no_correct"] for r in rows])),
                    "fail_missed": float(np.mean([r["fail_missed"] for r in rows])),
                    "accuracy_ci95": _bootstrap_ci(acc, resamples, order_seed),
                }
            ds_res[rule] = rule_res

        # Main accuracy differences vs self-consistency, paired question-level bootstrap.
        base = "self_consistency"
        diffs: dict[str, Any] = {}
        if base in per_q_acc:
            for rule in rules:
                if rule == base:
                    continue
                diffs[rule] = {}
                for N in budgets:
                    d = per_q_acc[rule][N] - per_q_acc[base][N]
                    diffs[rule][N] = {"mean": float(d.mean()),
                                      "ci95": _bootstrap_ci(d, resamples, order_seed)}
        ds_res["_accuracy_diff_vs_self_consistency"] = diffs
        results[dataset] = ds_res

    out_dir = ensure_dir(REPO_ROOT / "results" / "task3")
    with open(os.fspath(out_dir / "offline.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[task3.offline] wrote {out_dir / 'offline.json'} "
          f"({', '.join(results)} datasets).")


def _prefix_orders(orders: list[list[int]], n: int) -> list[list[int]]:
    """Nested prefixes of length n of each order (reused across methods/budgets)."""
    return [o[:n] for o in orders]


def main() -> None:
    parser = build_arg_parser("Task 3 offline selection experiments.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
