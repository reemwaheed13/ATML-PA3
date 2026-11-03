"""Task 3 analysis: figures + tables from the saved results files only. (handout p12-p15)

`python -m task3_ttc.analysis`

Reads results/task3/{offline.json, adaptive.json, signals.json, offline_per_question.parquet,
adaptive_per_question.parquet, live/live_summary.json, live/live_records.jsonl} and the committed
splits/task3/<dataset>_splits.json. Writes figures to results/task3/figures and CSV+markdown tables
to results/task3/tables, plus results/task3/qualitative_cases.json.

This module does not touch the raw data; it only summarizes what the pipeline already produced. It
optionally reads the rollout-scalars cache (DRIVE_ROOT/task3/cache) to annotate group sizes in the
qualitative cases, degrading gracefully if that cache is absent.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from common.paths import REPO_ROOT, SPLITS_DIR

RES = REPO_ROOT / "results" / "task3"
FIG = RES / "figures"
TAB = RES / "tables"
RWC_QUALITY = {"math": ["self_certainty_rwc"],
               "mathvista": ["visconf_rwc", "visconf_token_rwc", "vision_only_rwc"]}


# =================================================================================================
# IO helpers
# =================================================================================================

def _load_json(path) -> Optional[dict]:
    try:
        with open(os.fspath(path), encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return None


def _ensure_dirs():
    for d in (FIG, TAB):
        os.makedirs(os.fspath(d), exist_ok=True)


def _write_table(base: str, header: list[str], rows: list[list]) -> None:
    import csv
    with open(os.fspath(TAB / f"{base}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    with open(os.fspath(TAB / f"{base}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _rules_of(ds: dict) -> list[str]:
    skip = {"n_eval_questions", "budgets", "_accuracy_diffs"}
    return [k for k in ds if k not in skip]


def _best_rwc(ds: dict, dataset: str, budgets: list[int]) -> str:
    cands = [r for r in RWC_QUALITY.get(dataset, []) if r in ds]
    if not cands:
        return "self_consistency"
    N = str(max(budgets))
    return max(cands, key=lambda r: ds[r][N]["accuracy"])


# =================================================================================================
# Figures
# =================================================================================================

def _figures(offline: dict, adaptive: Optional[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset, ds in offline.items():
        budgets = [int(b) for b in ds["budgets"]]
        Ns = [str(b) for b in budgets]
        rules = _rules_of(ds)

        # 1) accuracy + oracle coverage vs N
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in ("first_ans", "self_consistency"):
            if r in ds:
                ax.plot(budgets, [ds[r][n]["accuracy"] for n in Ns], marker="o", label=r)
        if "self_consistency" in ds:
            ax.plot(budgets, [ds["self_consistency"][n]["oracle_coverage"] for n in Ns],
                    marker="s", linestyle="--", label="oracle coverage")
        ax.set_xscale("log", base=2); ax.set_xlabel("N (candidates)"); ax.set_ylabel("rate")
        ax.set_title(f"{dataset}: accuracy & oracle vs N"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.fspath(FIG / f"accuracy_vs_N_{dataset}.png"), dpi=130)
        plt.close(fig)

        # 2) accuracy vs mean generated-token cost, all rules + adaptive overlay
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rules:
            ax.plot([ds[r][n]["generated_token_cost"] for n in Ns],
                    [ds[r][n]["accuracy"] for n in Ns], marker=".", label=r)
        if adaptive and dataset in adaptive:
            for ctrl in ("agreement_based", "quality_aware"):
                c = adaptive[dataset].get(ctrl, {})
                if "eval_mean_token_cost" in c:
                    ax.scatter([c["eval_mean_token_cost"]], [c["eval_accuracy"]], s=80, marker="*",
                               zorder=5, label=f"adaptive:{ctrl}")
        ax.set_xlabel("mean generated-token cost"); ax.set_ylabel("accuracy")
        ax.set_title(f"{dataset}: accuracy vs cost"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.fspath(FIG / f"acc_vs_cost_{dataset}.png"), dpi=130)
        plt.close(fig)

        # 3) failure decomposition vs N (SC and best quality rule)
        best = _best_rwc(ds, dataset, budgets)
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in ("self_consistency", best):
            if r in ds:
                ax.plot(budgets, [ds[r][n]["fail_no_correct"] for n in Ns], marker="o",
                        label=f"{r}: no correct generated")
                ax.plot(budgets, [ds[r][n]["fail_missed"] for n in Ns], marker="s", linestyle="--",
                        label=f"{r}: correct missed")
        ax.set_xscale("log", base=2); ax.set_xlabel("N"); ax.set_ylabel("failure rate")
        ax.set_title(f"{dataset}: failure decomposition"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.fspath(FIG / f"failure_decomp_{dataset}.png"), dpi=130)
        plt.close(fig)

        # 4) distinct-group counts vs N
        fig, ax = plt.subplots(figsize=(6, 4))
        if "self_consistency" in ds:
            ax.plot(budgets, [ds["self_consistency"][n]["distinct_groups_incl_no_answer"] for n in Ns],
                    marker="o", label="incl NO_ANSWER")
            ax.plot(budgets, [ds["self_consistency"][n]["distinct_groups_excl_no_answer"] for n in Ns],
                    marker="s", linestyle="--", label="excl NO_ANSWER")
        ax.set_xscale("log", base=2); ax.set_xlabel("N"); ax.set_ylabel("mean distinct groups")
        ax.set_title(f"{dataset}: distinct answer groups"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.fspath(FIG / f"distinct_groups_{dataset}.png"), dpi=130)
        plt.close(fig)


# =================================================================================================
# Tables
# =================================================================================================

def _method_tables(offline: dict) -> None:
    for dataset, ds in offline.items():
        budgets = [int(b) for b in ds["budgets"]]
        rules = _rules_of(ds)
        for N in (8, 32):
            if N not in budgets:
                continue
            n = str(N)
            rows = []
            for r in rules:
                e = ds[r][n]
                ci = e["accuracy_ci95"]
                rows.append([r, e["accuracy"], f"[{ci[0]:.4f}, {ci[1]:.4f}]", e["oracle_coverage"],
                             e["generated_token_cost"], e["distinct_groups_excl_no_answer"]])
            _write_table(f"method_comparison_{dataset}_N{N}",
                         ["rule", "accuracy", "acc_ci95", "oracle", "cost", "distinct_excl"], rows)


def _signal_tables(signals: dict) -> None:
    for dataset, ds in signals.items():
        if not isinstance(ds, dict) or "spearman_with_correctness" not in ds:
            continue
        rows = []
        for s, v in ds["spearman_with_correctness"].items():
            lv = ds["spearman_with_length"].get(s, {})
            rows.append([s, v["mean"], v["count"], lv.get("mean"), lv.get("count")])
        _write_table(f"signals_spearman_{dataset}",
                     ["score", "spearman_correct", "n_correct", "spearman_length", "n_length"], rows)
        rows = []
        for s, v in ds["quartile_gap"].items():
            rows.append([s, v["all"]["mean"], v["all"]["count"], v["mixed"]["mean"],
                         v["mixed"]["count"]])
        _write_table(f"signals_quartile_gap_{dataset}",
                     ["score", "gap_all", "n_all", "gap_mixed", "n_mixed"], rows)
        if "token_c_v_pearson" in ds:
            t = ds["token_c_v_pearson"]
            _write_table(f"signals_token_cv_{dataset}",
                         ["metric", "value"],
                         [["mean", t["mean"]], ["median", t["median"]], ["iqr", t["iqr"]],
                          ["fraction_negative", t["fraction_negative"]],
                          ["n_questions", t["n_questions"]],
                          ["skipped_zero_variance_rollouts", t["skipped_zero_variance_rollouts"]]])
            cc = ds["covariance_check"]
            _write_table(f"signals_covariance_check_{dataset}",
                         ["metric", "value"],
                         [["max_abs", cc["max_abs"]], ["mean_abs", cc["mean_abs"]],
                          ["n_rollouts", cc["n_rollouts"]]])


def _live_table(summary: dict) -> None:
    strategies = ["single_long", "parallel", "sequential_refinement"]
    rows = []
    for s in strategies:
        a = summary.get(s, {})
        rows.append([s, a.get("accuracy"), a.get("extraction_rate"), a.get("truncation_rate"),
                     a.get("mean_output_tokens"), a.get("total_runtime_s"),
                     a.get("wrong_to_correct", ""), a.get("correct_to_wrong", "")])
    _write_table("live_summary",
                 ["strategy", "accuracy", "extraction_rate", "truncation_rate", "mean_output_tokens",
                  "total_runtime_s", "wrong_to_correct", "correct_to_wrong"], rows)


def _adaptive_tables(adaptive: dict) -> None:
    for dataset, ds in adaptive.items():
        frozen_rows, grid_rows, held_rows, stop_rows = [], [], [], []
        for ctrl in ("agreement_based", "quality_aware"):
            c = ds.get(ctrl)
            if not c:
                continue
            fp = c["frozen"]["params"]
            frozen_rows.append([ctrl, fp.get("tau", ""), fp.get("gamma", ""),
                                bool(fp.get("always_32", False)), c["frozen"]["dev_accuracy"],
                                c["frozen"]["dev_cost"], c["frozen"]["ref_fixed32_accuracy"]])
            for g in c.get("dev_grid", []):
                p = g["params"]
                grid_rows.append([ctrl, p.get("tau", ""), p.get("gamma", ""),
                                  bool(p.get("always_32", False)), g["dev_accuracy"], g["dev_cost"],
                                  g["dev_mean_rollouts"]])
            held_rows.append([ctrl, c["eval_accuracy"], c.get("eval_mean_token_cost"),
                              c.get("eval_mean_rollouts"), c["incorrect_early_stops"],
                              c["recoverable_by_fixed32"],
                              c["accuracy_minus_fixed32_bootstrap"]["mean"]])
            for k, frac in c.get("stop_count_fractions", {}).items():
                stop_rows.append([ctrl, k, frac, c["stop_count_distribution"].get(k)])
        _write_table(f"adaptive_frozen_{dataset}",
                     ["controller", "tau", "gamma", "always_32", "dev_acc", "dev_cost",
                      "ref_fixed32_acc"], frozen_rows)
        _write_table(f"adaptive_dev_grid_{dataset}",
                     ["controller", "tau", "gamma", "always_32", "dev_acc", "dev_cost",
                      "dev_rollouts"], grid_rows)
        _write_table(f"adaptive_heldout_{dataset}",
                     ["controller", "eval_acc", "eval_cost", "eval_rollouts", "incorrect_early",
                      "recoverable", "acc_minus_fixed32"], held_rows)
        _write_table(f"adaptive_stop_distribution_{dataset}",
                     ["controller", "stop_n", "fraction", "count"], stop_rows)
        fn = ds.get("fixed_n_self_consistency", {})
        _write_table(f"adaptive_fixed_n_sc_{dataset}", ["N", "accuracy", "cost"],
                     [[N, v["accuracy"], v["cost"]] for N, v in fn.items()])


def _bin(frac: float) -> str:
    if frac <= 0:
        return "0"
    if frac <= 0.25:
        return "(0,0.25]"
    if frac <= 0.5:
        return "(0.25,0.5]"
    if frac <= 0.75:
        return "(0.5,0.75]"
    return "(0.75,1]"


def _stop_by_pool_table() -> None:
    path = RES / "adaptive_per_question.parquet"
    if not path.exists():
        return
    import pyarrow.parquet as pq
    t = pq.read_table(os.fspath(path),
                      columns=["dataset", "controller", "full_pool_correct_fraction", "stop_n",
                               "correct"])
    d = {k: t.column(k).to_pylist() for k in t.column_names}
    agg: dict[tuple, list] = {}
    order = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1]"]
    for i in range(len(d["dataset"])):
        key = (d["dataset"][i], d["controller"][i], _bin(float(d["full_pool_correct_fraction"][i])))
        agg.setdefault(key, []).append((int(d["stop_n"][i]), bool(d["correct"][i])))
    rows = []
    for (dataset, ctrl, b) in sorted(agg, key=lambda k: (k[0], k[1], order.index(k[2]))):
        vals = agg[(dataset, ctrl, b)]
        stops = np.array([v[0] for v in vals]); corr = np.array([v[1] for v in vals], dtype=float)
        rows.append([dataset, ctrl, b, float(stops.mean()), float(corr.mean()), len(vals)])
    _write_table("adaptive_stop_by_pool_fraction",
                 ["dataset", "controller", "pool_correct_bin", "mean_stop_n", "accuracy", "count"],
                 rows)


# =================================================================================================
# Qualitative cases
# =================================================================================================

def _eval_order(dataset: str) -> list[str]:
    sp = _load_json(SPLITS_DIR / "task3" / f"{dataset}_splits.json")
    return [str(x) for x in sp["eval"]] if sp and "eval" in sp else []


def _group_sizes(dataset: str) -> dict[str, dict[int, int]]:
    """Optional: per-question {group_id -> full-pool support} from the rollout-scalars cache."""
    try:
        from .offline import _scalars_cache_path
        path = _scalars_cache_path(dataset)
    except Exception:
        path = None
    if path is None or not path.exists():
        return {}
    import pyarrow.parquet as pq
    t = pq.read_table(os.fspath(path), columns=["sample_id", "group"])
    sids = [str(s) for s in t.column("sample_id").to_pylist()]
    grps = t.column("group").to_pylist()
    out: dict[str, dict[int, int]] = {}
    for s, g in zip(sids, grps):
        out.setdefault(s, {})
        out[s][int(g)] = out[s].get(int(g), 0) + 1
    return out


def _selection_disagreements(offline: dict) -> dict:
    path = RES / "offline_per_question.parquet"
    cases: dict[str, list] = {}
    if not path.exists():
        return cases
    import pyarrow.parquet as pq
    for dataset, ds in offline.items():
        budgets = [int(b) for b in ds["budgets"]]
        best = _best_rwc(ds, dataset, budgets)
        t = pq.read_table(os.fspath(path),
                          columns=["dataset", "sample_id", "rule", "N", "order_idx",
                                   "selected_group", "selected_answer", "correct"],
                          filters=[("dataset", "=", dataset), ("N", "=", max(budgets)),
                                   ("order_idx", "=", 0),
                                   ("rule", "in", ["self_consistency", best])])
        d = {k: t.column(k).to_pylist() for k in t.column_names}
        by_sid: dict[str, dict] = {}
        for i in range(len(d["sample_id"])):
            by_sid.setdefault(str(d["sample_id"][i]), {})[d["rule"][i]] = (
                int(d["selected_group"][i]), d["selected_answer"][i], bool(d["correct"][i]))
        sizes = _group_sizes(dataset)
        out = []
        for sid in _eval_order(dataset):
            rec = by_sid.get(sid)
            if not rec or "self_consistency" not in rec or best not in rec:
                continue
            sc, q = rec["self_consistency"], rec[best]
            if sc[0] == q[0]:
                continue
            out.append({"sample_id": sid, "rule_a": "self_consistency", "rule_b": best,
                        "sc_answer": sc[1], "sc_correct": sc[2], "sc_group": sc[0],
                        "quality_answer": q[1], "quality_correct": q[2], "quality_group": q[0],
                        "group_sizes": sizes.get(sid, {})})
            if len(out) >= 3:
                break
        cases[dataset] = out
    return cases


def _premature_stops() -> dict:
    path = RES / "adaptive_per_question.parquet"
    cases: dict[str, list] = {}
    if not path.exists():
        return cases
    import pyarrow.parquet as pq
    t = pq.read_table(os.fspath(path),
                      columns=["dataset", "controller", "order_idx", "stop_n", "correct", "answer",
                               "fixed32_correct", "full_pool_correct_fraction", "sample_id"],
                      filters=[("controller", "=", "quality_aware"), ("order_idx", "=", 0)])
    d = {k: t.column(k).to_pylist() for k in t.column_names}
    by = {}
    for i in range(len(d["sample_id"])):
        by.setdefault(d["dataset"][i], {})[str(d["sample_id"][i])] = i
    for dataset, sid_idx in by.items():
        out = []
        for sid in _eval_order(dataset):
            i = sid_idx.get(sid)
            if i is None:
                continue
            if int(d["stop_n"][i]) < 32 and not d["correct"][i] and d["fixed32_correct"][i]:
                out.append({"sample_id": sid, "stop_n": int(d["stop_n"][i]),
                            "answer": d["answer"][i],
                            "full_pool_correct_fraction": float(d["full_pool_correct_fraction"][i])})
            if len(out) >= 3:
                break
        cases[dataset] = out
    return cases


def _revision_cases() -> dict:
    path = RES / "live" / "live_records.jsonl"
    w2c, c2w = [], []
    if path.exists():
        with open(os.fspath(path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sr = r.get("sequential_refinement", {})
                base = {"sample_id": r.get("sample_id"), "gold": r.get("gold"),
                        "initial_answer": sr.get("initial_answer"), "revised_answer": sr.get("answer")}
                if sr.get("wrong_to_correct"):
                    w2c.append(base)
                if sr.get("correct_to_wrong"):
                    c2w.append(base)
    return {"wrong_to_correct": w2c, "correct_to_wrong": c2w}


# =================================================================================================
# Run
# =================================================================================================

def run() -> None:
    _ensure_dirs()
    offline = _load_json(RES / "offline.json")
    adaptive = _load_json(RES / "adaptive.json")
    signals = _load_json(RES / "signals.json")
    live = _load_json(RES / "live" / "live_summary.json")

    if offline:
        _figures(offline, adaptive)
        _method_tables(offline)
    if signals:
        _signal_tables(signals)
    if live:
        _live_table(live)
    if adaptive:
        _adaptive_tables(adaptive)
    _stop_by_pool_table()

    qual = {"selection_disagreement_rule": "SC vs the best quality-RWC rule at N=32, order 0",
            "premature_stopping_rule": "quality_aware stopped early & wrong at order 0 while "
                                       "fixed-32 was correct",
            "revision_rule": "all live sequential-refinement wrong->correct and correct->wrong",
            "selection_disagreement": _selection_disagreements(offline) if offline else {},
            "premature_stopping": _premature_stops(),
            "revision": _revision_cases()}
    with open(os.fspath(RES / "qualitative_cases.json"), "w", encoding="utf-8") as f:
        json.dump(qual, f, indent=2, ensure_ascii=False)
    print(f"[task3.analysis] wrote figures to {FIG}, tables to {TAB}, and qualitative_cases.json.")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
