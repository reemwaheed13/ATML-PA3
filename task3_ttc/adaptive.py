"""Adaptive sampling with dev-selected thresholds. (handout p13-p14, Step 5)

Simulated on the supplied pools: reveal rollouts in the saved random orders, check stopping after
n in {4,8,16} candidates (else stop at 32). Both controllers return the CURRENT self-consistency
answer, isolating stopping from final selection. Scores/calibration/ranks are recomputed using
only revealed candidates.

Controllers:
  * agreement_based: stop if a unique, non-NO_ANSWER leading answer is supported by >= fraction
    tau of the n candidates.
  * quality_aware: same agreement condition PLUS agreement between the self-consistency and RWC
    winners, and an RWC vote margin >= gamma. Quality signal = self-certainty (MATH) / VisConf
    (MathVista).

Threshold selection (on each dataset's dev set): pick the setting with the lowest mean token cost
whose dev accuracy is <= 1pp below fixed-32 self-consistency (averaged over 10 orders). Break ties
by higher accuracy, then smaller tau/gamma. Freeze before held-out evaluation.
"""

from __future__ import annotations

import itertools
import json
import os
from collections import Counter
from typing import Any, Optional

import numpy as np

from common.config import build_arg_parser, load_yaml
from common.paths import REPO_ROOT, ensure_dir
from . import data as t3data
from . import signals
from .offline import _question_stats, _vote, generate_orders
from .splits import make_splits

_ALWAYS_32 = {"always_32": True}


# =================================================================================================
# Stopping controllers
# =================================================================================================

def _leading_nonzero(groups: list[int]) -> tuple[Optional[int], int, bool]:
    """Return (unique leading non-NO_ANSWER group or None, its count, is-unique)."""
    counts = Counter(g for g in groups if g != 0)
    if not counts:
        return None, 0, False
    top = max(counts.values())
    leaders = [g for g, c in counts.items() if c == top]
    return (leaders[0] if len(leaders) == 1 else None), top, len(leaders) == 1


def agreement_stop(groups: list[int], tau: float) -> bool:
    """True if a unique non-NO_ANSWER leading answer has support fraction >= tau of the prefix."""
    g, cnt, _ = _leading_nonzero(groups)
    if g is None:
        return False
    return (cnt / len(groups)) >= tau


def quality_aware_stop(groups: list[int], scores: list[float], tau: float, gamma: float) -> bool:
    """agreement_stop AND self-consistency==RWC winner AND RWC margin >= gamma."""
    if not agreement_stop(groups, tau):
        return False
    if _vote(groups) != signals.rwc_winner(groups, scores):
        return False
    return signals.rwc_margin(groups, scores) >= gamma


def _quality_scores(qs: dict[str, Any], prefix: list[int], has_vision: bool) -> list[float]:
    if has_vision:
        return signals.visconf([qs["rollouts"][i] for i in prefix])
    return [qs["K"][i] for i in prefix]


def simulate(qs: dict[str, Any], controller: str, params: dict, order: list[int],
             *, checkpoints=(4, 8, 16), max_candidates: int = 32, has_vision: bool = False):
    """Reveal candidates in `order`, apply the stopping check at each checkpoint, else stop at 32.

    Returns the returned self-consistency answer, token cost, revealed-rollout count, correctness.
    """
    idx_all = [int(i) for i in order][:max_candidates]
    stop_n = max_candidates
    if not params.get("always_32"):
        for n in checkpoints:
            prefix = idx_all[:n]
            groups = [qs["groups"][i] for i in prefix]
            if controller == "agreement_based":
                stop = agreement_stop(groups, params["tau"])
            else:
                stop = quality_aware_stop(groups, _quality_scores(qs, prefix, has_vision),
                                          params["tau"], params["gamma"])
            if stop:
                stop_n = n
                break

    prefix = idx_all[:stop_n]
    answer = _vote([qs["groups"][i] for i in prefix])
    cost = sum(qs["length"][i] for i in prefix)
    return {"answer": answer, "cost": int(cost), "rollouts": stop_n,
            "correct": bool(qs["group_correct"].get(answer, False))}


# =================================================================================================
# Parameter grids + evaluation
# =================================================================================================

def _grid(controller: str, cfg: dict[str, Any]) -> list[dict]:
    cc = cfg["controllers"][controller]
    if controller == "agreement_based":
        grid = [{"tau": float(t)} for t in cc["tau_grid"]]
    else:
        grid = [{"tau": float(t), "gamma": float(g)}
                for t, g in itertools.product(cc["tau_grid"], cc["gamma_grid"])]
    if cfg.get("always_32_fallback", True):
        grid = grid + [dict(_ALWAYS_32)]
    return grid


def _run_params(stats, ids, controller, params, orders, cfg, has_vision):
    """Mean accuracy / token cost / rollouts over dev|eval questions x orders for one param set."""
    cps = tuple(cfg.get("check_points", [4, 8, 16]))
    mc = int(cfg.get("max_candidates", 32))
    accs, costs, rolls, stops = [], [], [], []
    for sid in ids:
        a = c = r = 0.0
        for order in orders:
            out = simulate(stats[sid], controller, params, order, checkpoints=cps,
                           max_candidates=mc, has_vision=has_vision)
            a += out["correct"]; c += out["cost"]; r += out["rollouts"]
            stops.append((out["rollouts"], out["correct"]))
        k = len(orders)
        accs.append(a / k); costs.append(c / k); rolls.append(r / k)
    return {"accuracy": float(np.mean(accs)), "cost": float(np.mean(costs)),
            "rollouts": float(np.mean(rolls)), "per_q_acc": accs, "stops": stops}


def select_thresholds(stats, dev_ids, controller, orders, cfg, has_vision) -> dict[str, Any]:
    """Pick per-controller thresholds on dev by the lowest-cost-within-tolerance rule. (p14)"""
    tol = float(cfg["selection"]["dev_accuracy_tolerance_pp"]) / 100.0
    baseline = _run_params(stats, dev_ids, controller, dict(_ALWAYS_32), orders, cfg, has_vision)
    ref_acc = baseline["accuracy"]

    scored = []
    for params in _grid(controller, cfg):
        res = _run_params(stats, dev_ids, controller, params, orders, cfg, has_vision)
        scored.append((params, res))

    eligible = [(p, r) for p, r in scored if r["accuracy"] >= ref_acc - tol]
    if not eligible:
        eligible = [(p, r) for p, r in scored if p.get("always_32")]

    def key(pr):
        p, r = pr
        return (r["cost"], -r["accuracy"], p.get("tau", 1.0), p.get("gamma", 1.0))

    best_params, best_res = min(eligible, key=key)
    return {"params": best_params, "dev_accuracy": best_res["accuracy"],
            "dev_cost": best_res["cost"], "ref_fixed32_accuracy": ref_acc}


# =================================================================================================
# Run
# =================================================================================================

def run(cfg: dict[str, Any]) -> None:
    """Select thresholds on dev, freeze, evaluate on held-out; save accuracy/cost/stop distributions."""
    offline_cfg = load_yaml(cfg.get("offline_config", "configs/task3/offline.yaml"))
    seed = int(offline_cfg["splits"]["seed"])
    dev_size = int(offline_cfg["splits"]["dev_size"])
    n_orders = int(offline_cfg["offline"]["n_orders"])
    order_seed = int(offline_cfg["offline"]["order_seed"])

    results: dict[str, Any] = {}
    for spec in offline_cfg["data"]["models"]:
        dataset = spec["dataset"]
        has_vision = spec.get("type") == "vlm"
        questions = t3data.load_joined(offline_cfg, spec)
        qmap = {q.sample_id: q for q in questions}
        split = make_splits(list(qmap), seed=seed, dev_size=dev_size)
        n_roll = max(len(q.rollouts) for q in questions)
        orders = generate_orders(n_roll, n_orders, seed=order_seed)
        stats = {sid: _question_stats(qmap[sid], has_vision=has_vision)
                 for sid in split["dev"] + split["eval"]}

        ds_res: dict[str, Any] = {}
        fixed32 = _run_params(stats, split["eval"], "agreement_based", dict(_ALWAYS_32),
                              orders, cfg, has_vision)
        ds_res["fixed_32_self_consistency"] = {"accuracy": fixed32["accuracy"],
                                               "cost": fixed32["cost"]}

        for controller in ("agreement_based", "quality_aware"):
            frozen = select_thresholds(stats, split["dev"], controller, orders, cfg, has_vision)
            held = _run_params(stats, split["eval"], controller, frozen["params"], orders, cfg,
                               has_vision)
            stop_dist = Counter(n for n, _ in held["stops"])
            incorrect_early = sum(1 for n, ok in held["stops"] if n < 32 and not ok)
            ds_res[controller] = {
                "frozen": frozen,
                "eval_accuracy": held["accuracy"],
                "eval_mean_token_cost": held["cost"],
                "eval_mean_rollouts": held["rollouts"],
                "stop_count_distribution": {str(k): int(v) for k, v in sorted(stop_dist.items())},
                "incorrect_early_stops": int(incorrect_early),
            }
        results[dataset] = ds_res

    out_dir = ensure_dir(REPO_ROOT / "results" / "task3")
    with open(os.fspath(out_dir / "adaptive.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[task3.adaptive] wrote {out_dir / 'adaptive.json'} ({', '.join(results)}).")


def main() -> None:
    parser = build_arg_parser("Task 3 adaptive sampling (dev-selected thresholds).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
