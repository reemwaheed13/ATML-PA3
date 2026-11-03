"""Adaptive sampling with dev-selected thresholds. (handout p13-p14, Step 5)

Simulated on the supplied pools using the SAME per-question orders and VisConf cache as the offline
step (offline.build_question_arrays / load_or_make_orders / load_or_build_visconf). Reveal rollouts
in the saved order; check stopping after n in {4,8,16} candidates, else stop at 32. Both controllers
return the CURRENT self-consistency answer, isolating stopping from final selection.

Controllers:
  * agreement_based: count ALL groups (including NO_ANSWER). Stop only if the top count is held by
    exactly one group, that group is not NO_ANSWER, and count/n >= tau.
  * quality_aware: the same agreement condition PLUS the self-consistency and RWC winners agree and
    the RWC vote margin >= gamma. Quality signal = self-certainty K (MATH) / VisConf Z (MathVista).

Thresholds are selected on each dataset's dev set: lowest mean token cost among settings whose dev
accuracy is >= fixed-32 self-consistency minus the tolerance (with a 1e-12 slack). Ties: higher
accuracy, then smaller tau, then smaller gamma. Frozen before held-out evaluation.
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
from .offline import (_sc_winner, build_question_arrays, load_or_build_visconf, load_or_make_orders)
from .splits import make_splits

_ALWAYS_32 = {"always_32": True}


# =================================================================================================
# Stopping controllers
# =================================================================================================

def _agreement_stop(groups: np.ndarray, tau: float) -> bool:
    """Stop iff a unique top group (counting NO_ANSWER) is not NO_ANSWER and has support >= tau."""
    counts = np.bincount(groups)
    best = int(counts.max())
    leaders = np.flatnonzero(counts == best)
    if leaders.size != 1:
        return False
    g = int(leaders[0])
    if g == 0:                                              # NO_ANSWER leads -> never stop
        return False
    return (best / len(groups)) >= tau


def _quality_stop(groups: np.ndarray, scores: list, tau: float, gamma: float) -> bool:
    """agreement_stop AND self-consistency==RWC winner AND RWC margin >= gamma."""
    if not _agreement_stop(groups, tau):
        return False
    if _sc_winner(groups) != int(signals.rwc_winner(groups.tolist(), scores)):
        return False
    return signals.rwc_margin(groups.tolist(), scores) >= gamma


def simulate(e: dict, pos_order: np.ndarray, controller: str, params: dict, *, checkpoints,
             max_candidates: int, has_vision: bool, zdict=None) -> dict[str, Any]:
    """Reveal candidates in order; apply the stop check at each checkpoint, else stop at 32."""
    idx_all = pos_order[:max_candidates]
    stop_n = max_candidates
    if not params.get("always_32"):
        for n in checkpoints:
            pos = idx_all[:n]
            groups = e["group"][pos]
            if controller == "agreement_based":
                stop = _agreement_stop(groups, params["tau"])
            else:
                scores = list(zdict[n]) if has_vision else e["K"][pos].tolist()
                stop = _quality_stop(groups, scores, params["tau"], params["gamma"])
            if stop:
                stop_n = n
                break
    pos = idx_all[:stop_n]
    groups = e["group"][pos]
    winner = _sc_winner(groups)
    counts = np.bincount(groups)
    return {"stop_n": int(stop_n), "group": int(winner),
            "correct": bool(e["group_correct"].get(winner, False)),
            "cost": int(e["length"][pos].sum()),
            "leading_frac": float(counts[winner] / len(groups))}


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


def _run_params(qa, posfull, ids, controller, params, n_orders, cfg, has_vision, vz):
    """Mean accuracy / cost / rollouts over ids x orders for one param set; also per-q acc + stops."""
    cps = tuple(cfg.get("check_points", [4, 8, 16]))
    mc = int(cfg.get("max_candidates", 32))
    accs, costs, rolls = [], [], []
    stops: list[tuple[int, bool]] = []
    for sid in ids:
        e = qa[sid]
        a = c = r = 0.0
        for o in range(n_orders):
            zdict = {n: vz[sid]["Z"][(o, n)] for n in cps} if has_vision else None
            out = simulate(e, posfull[sid][o], controller, params, checkpoints=cps,
                           max_candidates=mc, has_vision=has_vision, zdict=zdict)
            a += out["correct"]; c += out["cost"]; r += out["stop_n"]
            stops.append((out["stop_n"], out["correct"]))
        accs.append(a / n_orders); costs.append(c / n_orders); rolls.append(r / n_orders)
    return {"accuracy": float(np.mean(accs)), "cost": float(np.mean(costs)),
            "rollouts": float(np.mean(rolls)), "per_q_acc": np.array(accs), "stops": stops}


def select_thresholds(qa, posfull, dev_ids, controller, n_orders, cfg, has_vision, vz):
    """Pick per-controller thresholds on dev; return frozen choice + the full dev grid."""
    tol = float(cfg["selection"]["dev_accuracy_tolerance_pp"]) / 100.0
    ref = _run_params(qa, posfull, dev_ids, controller, dict(_ALWAYS_32), n_orders, cfg,
                      has_vision, vz)
    ref_acc = ref["accuracy"]

    grid_rows = []
    scored = []
    for params in _grid(controller, cfg):
        res = _run_params(qa, posfull, dev_ids, controller, params, n_orders, cfg, has_vision, vz)
        scored.append((params, res))
        grid_rows.append({"params": params, "dev_accuracy": res["accuracy"],
                          "dev_cost": res["cost"], "dev_mean_rollouts": res["rollouts"]})

    eligible = [(p, r) for p, r in scored if r["accuracy"] >= ref_acc - tol - 1e-12]
    if not eligible:
        eligible = [(p, r) for p, r in scored if p.get("always_32")]

    def key(pr):
        p, r = pr
        return (r["cost"], -r["accuracy"], p.get("tau", 1.0), p.get("gamma", 1.0))

    best_params, best_res = min(eligible, key=key)
    return ({"params": best_params, "dev_accuracy": best_res["accuracy"],
             "dev_cost": best_res["cost"], "dev_mean_rollouts": best_res["rollouts"],
             "ref_fixed32_accuracy": ref_acc}, grid_rows)


# =================================================================================================
# Held-out evaluation + per-question records
# =================================================================================================

def _fixed_n_sc(qa, posfull, ids, n_orders, Ns):
    """Fixed-budget self-consistency accuracy + cost on eval, same orders."""
    out = {}
    for N in Ns:
        accs, costs = [], []
        for sid in ids:
            e = qa[sid]
            a = c = 0.0
            for o in range(n_orders):
                pos = posfull[sid][o][:N]
                w = _sc_winner(e["group"][pos])
                a += bool(e["group_correct"].get(w, False)); c += int(e["length"][pos].sum())
            accs.append(a / n_orders); costs.append(c / n_orders)
        out[int(N)] = {"accuracy": float(np.mean(accs)), "cost": float(np.mean(costs)),
                       "per_q_acc": np.array(accs)}
    return out


def _eval_controller(qa, posfull, eval_ids, controller, params, n_orders, cfg, has_vision, vz,
                     dataset, rows):
    """Run the frozen controller on eval; collect per-question rows + stop/recoverable stats."""
    cps = tuple(cfg.get("check_points", [4, 8, 16]))
    mc = int(cfg.get("max_candidates", 32))
    per_q_acc = []
    stop_counter: Counter = Counter()
    incorrect_early = recoverable = 0
    total = 0
    for sid in eval_ids:
        e = qa[sid]
        full_frac = float(e["correct"].mean())
        a = 0.0
        for o in range(n_orders):
            zdict = {n: vz[sid]["Z"][(o, n)] for n in cps} if has_vision else None
            out = simulate(e, posfull[sid][o], controller, params, checkpoints=cps,
                           max_candidates=mc, has_vision=has_vision, zdict=zdict)
            fixed32 = _sc_winner(e["group"][posfull[sid][o][:mc]])
            fixed32_correct = bool(e["group_correct"].get(fixed32, False))
            a += out["correct"]; total += 1
            stop_counter[out["stop_n"]] += 1
            if out["stop_n"] < mc and not out["correct"]:
                incorrect_early += 1
                if fixed32_correct:
                    recoverable += 1
            rows["dataset"].append(dataset); rows["controller"].append(controller)
            rows["sample_id"].append(sid); rows["order_idx"].append(o)
            rows["stop_n"].append(out["stop_n"])
            rows["answer"].append(e["group_rep"].get(out["group"], t3data.NO_ANSWER))
            rows["correct"].append(out["correct"]); rows["cost"].append(out["cost"])
            rows["leading_support_fraction_at_stop"].append(out["leading_frac"])
            rows["fixed32_correct"].append(fixed32_correct)
            rows["full_pool_correct_fraction"].append(full_frac)
        per_q_acc.append(a / n_orders)
    stop_dist = {str(k): stop_counter[k] for k in sorted(stop_counter)}
    stop_frac = {str(k): stop_counter[k] / total for k in sorted(stop_counter)} if total else {}
    return {"per_q_acc": np.array(per_q_acc), "stop_count_distribution": stop_dist,
            "stop_count_fractions": stop_frac, "incorrect_early_stops": incorrect_early,
            "recoverable_by_fixed32": recoverable}


# =================================================================================================
# Run
# =================================================================================================

def run(cfg: dict[str, Any]) -> None:
    """Select thresholds on dev, freeze, evaluate on held-out; write adaptive.json + per-question."""
    offline_cfg = load_yaml(cfg.get("offline_config", "configs/task3/offline.yaml"))
    seed = int(offline_cfg["splits"]["seed"])
    dev_size = int(offline_cfg["splits"]["dev_size"])
    n_orders = int(offline_cfg["offline"]["n_orders"])
    order_seed = int(offline_cfg["offline"].get("order_seed", 6304))
    budgets = [int(b) for b in offline_cfg["offline"]["budgets"]]
    resamples = int(offline_cfg["report"]["bootstrap_resamples"])
    mc = int(cfg.get("max_candidates", 32))

    results: dict[str, Any] = {}
    rows = {k: [] for k in ("dataset", "controller", "sample_id", "order_idx", "stop_n", "answer",
                            "correct", "cost", "leading_support_fraction_at_stop", "fixed32_correct",
                            "full_pool_correct_fraction")}

    for spec in offline_cfg["data"]["models"]:
        dataset = str(spec["dataset"])
        has_vision = spec.get("type") == "vlm"
        qa, _ = build_question_arrays(offline_cfg, spec)
        split = make_splits(list(qa), seed=seed, dev_size=dev_size)
        dev_ids, eval_ids = split["dev"], split["eval"]
        orders = load_or_make_orders(dataset, qa, n_orders=n_orders, seed=order_seed)
        vz = load_or_build_visconf(offline_cfg, spec, qa, orders, budgets) if has_vision else None
        posfull = {sid: [np.array([{int(x): i for i, x in enumerate(qa[sid]["rollout_id"])}[int(r)]
                                   for r in orders[sid][o]], dtype=np.int64)
                         for o in range(n_orders)] for sid in (dev_ids + eval_ids)}

        ds_res: dict[str, Any] = {}
        fixed = _fixed_n_sc(qa, posfull, eval_ids, n_orders, [4, 8, 16, 32])
        ds_res["fixed_n_self_consistency"] = {
            str(N): {"accuracy": fixed[N]["accuracy"], "cost": fixed[N]["cost"]} for N in fixed}
        fixed32_pq = fixed[32]["per_q_acc"]
        idx = np.random.default_rng(order_seed).integers(0, len(eval_ids), size=(resamples,
                                                                                len(eval_ids)))

        for controller in ("agreement_based", "quality_aware"):
            frozen, grid_rows = select_thresholds(qa, posfull, dev_ids, controller, n_orders, cfg,
                                                  has_vision, vz)
            held = _eval_controller(qa, posfull, eval_ids, controller, frozen["params"], n_orders,
                                    cfg, has_vision, vz, dataset, rows)
            d = held["per_q_acc"] - fixed32_pq
            boot = {"mean": float(d.mean()),
                    "ci95": [float(np.percentile(d[idx].mean(axis=1), 2.5)),
                             float(np.percentile(d[idx].mean(axis=1), 97.5))]}
            ds_res[controller] = {
                "frozen": frozen,
                "dev_grid": grid_rows,
                "eval_accuracy": float(held["per_q_acc"].mean()),
                "eval_mean_token_cost": None,
                "stop_count_distribution": held["stop_count_distribution"],
                "stop_count_fractions": held["stop_count_fractions"],
                "incorrect_early_stops": held["incorrect_early_stops"],
                "recoverable_by_fixed32": held["recoverable_by_fixed32"],
                "accuracy_minus_fixed32_bootstrap": boot,
            }
        # mean token cost / rollouts for the frozen controllers on eval (recompute cheaply)
        for controller in ("agreement_based", "quality_aware"):
            res = _run_params(qa, posfull, eval_ids, controller,
                              ds_res[controller]["frozen"]["params"], n_orders, cfg, has_vision, vz)
            ds_res[controller]["eval_mean_token_cost"] = res["cost"]
            ds_res[controller]["eval_mean_rollouts"] = res["rollouts"]
        results[dataset] = ds_res

    out_dir = ensure_dir(REPO_ROOT / "results" / "task3")
    with open(os.fspath(out_dir / "adaptive.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_rows(out_dir, rows)
    print(f"[task3.adaptive] wrote {out_dir/'adaptive.json'} and adaptive_per_question.parquet "
          f"({len(rows['dataset'])} rows; {', '.join(results)}).")


def _write_rows(out_dir, rows: dict[str, list]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table(rows), os.fspath(out_dir / "adaptive_per_question.parquet"))


def main() -> None:
    parser = build_arg_parser("Task 3 adaptive sampling (dev-selected thresholds).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
