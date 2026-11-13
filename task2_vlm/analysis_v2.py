"""Task 2 follow-up analysis (CPU-only; no model runs, no re-generation). (handout p9, p15)

Produces three additional Required-Evidence artifacts from the *already saved* final per-example
dump and training log, reusing the scoring + bootstrap machinery in `task2_vlm.evaluate` and the
table/figure helpers in `task2_vlm.analysis`. It never re-scores and never overwrites an existing
result file -- it writes only:

  * results/task2/tables/final_cis.{csv,md}        -- per suite x condition pair-bootstrap CIs
                                                       (accuracy_overall, pair_accuracy) plus the
                                                       correct-minus-mismatched accuracy & NLL
                                                       advantages with CIs, all from the SAME
                                                       4,000 matched-pair resamples (seed 6304).
  * results/task2/tables/qualitative_v2.{csv,md}   -- qualitative table v2 (no pair reused across
                                                       categories; natural_vqa adds one (d) example
                                                       per question_type); raw generations shown.
  * results/task2/figures/learning_curves_2x2.pdf  -- the learning_curves.png panels in a compact
                                                       2x2 layout, 5.5in wide, all fonts >= 7pt.

    python -m task2_vlm.analysis_v2 --config configs/task2/vlm_stages.yaml

Inputs (run_id taken from the config; default run_id 'vlm_stages'):
  * per-example:   results/task2/<run_id>_final_examples.jsonl
  * training log:  results/task2/train/<run_id>.jsonl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless / CPU-only

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import SEED  # noqa: E402  -- handout seed 6304
from common.config import load_yaml  # noqa: E402
from common.logging_utils import read_jsonl  # noqa: E402
from common.paths import REPO_ROOT, ensure_dir  # noqa: E402
from .scoring import normalize_answer  # noqa: E402
# Reused, read-only, from the existing modules (do not change scoring):
from .analysis import (  # noqa: E402
    CONDITIONS,
    QUAL_CATEGORY_RULE,
    SUITES,
    _dedupe_last,
    _results_dir,
    _stage_boundaries,
    _write_table,
)
from .evaluate import _correct_image_advantage  # noqa: E402

RESAMPLES = 4000
BL, BH = 2.5, 97.5  # 95% percentile bootstrap bounds


def _inst_key(r: dict) -> tuple:
    """Identify an instance across conditions: (pair_id, question, target) is unique within a pair."""
    return (r.get("pair_id"), r.get("question"), r.get("target"))


def _nan(v: Any) -> float:
    return float(v) if v is not None and np.isfinite(v) else float("nan")


# =================================================================================================
# 1) Per suite x condition CIs + correct-minus-mismatched advantages (shared resamples)
# =================================================================================================

def _ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, BL)), float(np.percentile(samples, BH))


def final_cis(examples: list[dict], tables_dir: Path) -> list[Path]:
    """95% pair-bootstrap CIs, matched pair IDs across conditions, exactly as evaluate.py.

    For every suite we draw ONE sequence of 4,000 pair resamples with np.random.default_rng(6304)
    and apply the SAME draws to every condition and metric, so accuracy_overall / pair_accuracy
    per condition and the correct-minus-mismatched advantages all share the identical resamples.
    The correct-minus-mismatched advantage reuses evaluate._correct_image_advantage unchanged
    (its internal rng is re-seeded to the same SEED and the same pair count, so it replays the
    identical draw sequence).
    """
    rows: list[dict] = []
    for suite in SUITES:
        base = [r for r in examples if r.get("suite") == suite and r.get("condition") == "correct"]
        if not base:
            continue
        order = [_inst_key(r) for r in base]

        # Condition -> per-instance score / nll aligned to `order`.
        def aligned(cond: str, field: str) -> list:
            d = {_inst_key(r): r.get(field) for r in examples
                 if r.get("suite") == suite and r.get("condition") == cond}
            return [d[k] for k in order]

        scores = {cond: aligned(cond, "score") for cond in CONDITIONS}

        # Pair aggregation (complete pairs, matched pair IDs).
        idx_by_pair: dict[Any, list[int]] = {}
        for i, r in enumerate(base):
            idx_by_pair.setdefault(r["pair_id"], []).append(i)
        pair_ids = sorted(idx_by_pair, key=str)
        n = len(pair_ids)

        acc_pp = {c: np.array([float(np.mean([scores[c][i] for i in idx_by_pair[p]]))
                               for p in pair_ids]) for c in CONDITIONS}
        pc_pp = {c: np.array([1.0 if all(scores[c][i] >= 0.5 for i in idx_by_pair[p]) else 0.0
                              for p in pair_ids]) for c in CONDITIONS}

        rng = np.random.default_rng(SEED)
        acc_M = {c: np.empty(RESAMPLES) for c in CONDITIONS}
        pc_M = {c: np.empty(RESAMPLES) for c in CONDITIONS}
        for b in range(RESAMPLES):
            sel = rng.integers(0, n, n)  # one draw per iter -> matched across conditions/metrics
            for c in CONDITIONS:
                acc_M[c][b] = acc_pp[c][sel].mean()
                pc_M[c][b] = pc_pp[c][sel].mean()

        for cond in CONDITIONS:
            for metric, point, dist in (("accuracy_overall", acc_pp[cond].mean(), acc_M[cond]),
                                        ("pair_accuracy", pc_pp[cond].mean(), pc_M[cond])):
                lo, hi = _ci(dist)
                rows.append({"suite": suite, "scope": cond, "metric": metric,
                             "estimate": float(point), "ci95_low": lo, "ci95_high": hi,
                             "n_pairs": n})

        # Correct-minus-mismatched advantage (accuracy + NLL) via the evaluate.py helper, reused by
        # mapping the 'complementary_swap' slot to the mismatched condition (same resamples: the
        # helper re-seeds default_rng(SEED) over the same n pairs).
        cond_scores = {"correct": scores["correct"], "complementary_swap": scores["mismatched"]}
        cond_nlls = {"correct": [_nan(v) for v in aligned("correct", "reference_nll")],
                     "complementary_swap": [_nan(v) for v in aligned("mismatched", "reference_nll")]}
        instances = [{"pair_id": r["pair_id"]} for r in base]
        adv = _correct_image_advantage(instances, cond_scores, cond_nlls, RESAMPLES)
        acc_ci = adv["accuracy_advantage_ci95"]
        nll_ci = adv["nll_advantage_ci95"]
        rows.append({"suite": suite, "scope": "correct_minus_mismatched",
                     "metric": "accuracy_advantage", "estimate": adv["accuracy_advantage_mean"],
                     "ci95_low": acc_ci[0], "ci95_high": acc_ci[1], "n_pairs": adv["n_pairs"]})
        rows.append({"suite": suite, "scope": "correct_minus_mismatched",
                     "metric": "nll_advantage", "estimate": adv["nll_advantage_mean"],
                     "ci95_low": nll_ci[0], "ci95_high": nll_ci[1], "n_pairs": adv["n_pairs"]})

    caption = (
        f"95% percentile CIs from {RESAMPLES} matched-pair bootstrap resamples (complete pairs, "
        f"seed {SEED}), matched pair IDs across conditions -- one shared resample sequence per "
        f"suite drives every condition and metric. accuracy_advantage = correct - mismatched; "
        f"nll_advantage = mismatched - correct (positive => correct image lowers reference NLL). "
        f"Bootstrap mechanics identical to task2_vlm.evaluate._correct_image_advantage; scoring "
        f"reused unchanged from the saved per-example dump."
    )
    return _write_table(pd.DataFrame(rows), "final_cis", tables_dir, caption=caption)


# =================================================================================================
# 2) Qualitative table v2 (no pair reused across categories; natural_vqa per-question_type for (d))
# =================================================================================================

QUAL_RULE_V2 = (
    "Final-test complementary pairs sorted by pair_id. For each category (a)-(d) the first matching "
    "pair that has NOT already been used by an earlier category is taken (no pair reused across "
    "categories). Categories: (a) both images scored correct under correct-image input; "
    "(b) identical normalized prediction on both images under correct-image input; (c) under "
    "complementary-swap input, both images' predictions equal the partner image's canonical "
    "reference; (d) under correct-image input, one image is wrong (score<0.5) while its partner is "
    "correct (score>=0.5). For natural_vqa, after (a)-(d) one additional (d) example is added per "
    "question_type (yes_no, number, other): the first still-unused (d) pair of that type. Matching "
    "uses normalized predictions (scoring unchanged); raw_generation is reported for the correct "
    "and swap conditions."
)


def _pairs_by_suite_v2(examples: list[dict], suite: str) -> dict[Any, dict]:
    """Per-pair correct/swap raw+normalized predictions, scores, references, question_type."""
    pairs: dict[Any, dict[tuple, dict]] = {}
    for r in examples:
        if r.get("suite") != suite:
            continue
        cond = r.get("condition")
        if cond not in ("correct", "complementary_swap"):
            continue
        pid = r.get("pair_id")
        inst_key = (r.get("question"), r.get("target"))
        pair = pairs.setdefault(pid, {})
        inst = pair.setdefault(inst_key, {
            "question": r.get("question"), "target": r.get("target"),
            "question_type": r.get("question_type"),
            "references": list(r.get("references") or []),
            "partner_canonical": r.get("partner_canonical"),
        })
        if cond == "correct":
            inst["correct_image"] = r.get("image_id")
            inst["correct_pred"] = r.get("normalized_prediction")
            inst["correct_raw"] = r.get("raw_generation")
            inst["correct_score"] = r.get("score")
            inst["references"] = list(r.get("references") or [])
            inst["question_type"] = r.get("question_type")
        else:
            inst["swap_pred"] = r.get("normalized_prediction")
            inst["swap_raw"] = r.get("raw_generation")
    return pairs


def _lohi(pairs: dict, pid: Any) -> Optional[tuple[dict, dict]]:
    insts = list(pairs[pid].values())
    if len(insts) != 2 or any("correct_pred" not in i for i in insts):
        return None
    lo, hi = sorted(insts, key=lambda i: (str(i.get("target")), str(i.get("correct_image"))))
    return lo, hi


def _matches(key: str, lo: dict, hi: dict) -> bool:
    cs = [float(lo.get("correct_score") or 0.0), float(hi.get("correct_score") or 0.0)]
    if key == "a_both_correct":
        return cs[0] >= 0.5 and cs[1] >= 0.5
    if key == "b_identical_answer":
        return lo.get("correct_pred") == hi.get("correct_pred")
    if key == "c_follows_partner_on_swap":
        return all(i.get("swap_pred") is not None
                   and i.get("swap_pred") == normalize_answer(i.get("partner_canonical") or "")
                   for i in (lo, hi))
    return (cs[0] >= 0.5) != (cs[1] >= 0.5)  # d_wrong_vs_right_partner


def _first_match(pairs: dict, key: str, used: set, *, question_type: Optional[str] = None):
    for pid in sorted(pairs, key=str):
        if pid in used:
            continue
        lh = _lohi(pairs, pid)
        if lh is None:
            continue
        lo, hi = lh
        if question_type is not None and lo.get("question_type") != question_type:
            continue
        if _matches(key, lo, hi):
            return pid
    return None


def _qual_row(suite: str, category: str, question_type: Optional[str], pid: Any,
              lo: dict, hi: dict) -> dict:
    return {
        "suite": suite, "category": category,
        "question_type": question_type if question_type is not None else lo.get("question_type"),
        "pair_id": pid, "question": lo.get("question"),
        "image_left": lo.get("correct_image"), "image_right": hi.get("correct_image"),
        "ref_left": lo.get("target"), "ref_right": hi.get("target"),
        "references_left": "; ".join(lo.get("references") or []),
        "references_right": "; ".join(hi.get("references") or []),
        "raw_correct_left": lo.get("correct_raw"), "raw_correct_right": hi.get("correct_raw"),
        "raw_swap_left": lo.get("swap_raw"), "raw_swap_right": hi.get("swap_raw"),
        "score_left": lo.get("correct_score"), "score_right": hi.get("correct_score"),
        "selection_rule": QUAL_CATEGORY_RULE.get(category, category),
    }


def qualitative_v2(examples: list[dict], tables_dir: Path) -> list[Path]:
    rows: list[dict] = []
    for suite in SUITES:
        pairs = _pairs_by_suite_v2(examples, suite)
        if not pairs:
            continue
        used: set = set()
        for key in QUAL_CATEGORY_RULE:  # (a)-(d) in order
            pid = _first_match(pairs, key, used)
            if pid is None:
                print(f"[task2.analysis_v2] {suite}: no unused pair for category {key}.")
                continue
            used.add(pid)
            lo, hi = _lohi(pairs, pid)
            rows.append(_qual_row(suite, key, None, pid, lo, hi))
        if suite == "natural_vqa":
            for qt in ("yes_no", "number", "other"):
                pid = _first_match(pairs, "d_wrong_vs_right_partner", used, question_type=qt)
                if pid is None:
                    print(f"[task2.analysis_v2] natural_vqa: no unused (d) pair for {qt}.")
                    continue
                used.add(pid)
                lo, hi = _lohi(pairs, pid)
                rows.append(_qual_row(suite, "d_wrong_vs_right_partner", qt, pid, lo, hi))
    return _write_table(pd.DataFrame(rows), "qualitative_v2", tables_dir, caption=QUAL_RULE_V2)


# =================================================================================================
# 3) Compact 2x2 learning-curve figure (PDF, 5.5in wide, fonts >= 7pt)
# =================================================================================================

def _stage_lines(ax, boundaries: list[int], names: list[str]) -> None:
    ymax = ax.get_ylim()[1]
    for i, b in enumerate(boundaries):
        ax.axvline(b, color="0.4", ls="--", lw=0.8)
        label = names[i + 1] if i + 1 < len(names) else f"stage@{b}"
        ax.text(b, ymax, f" {label}\n @{b}", fontsize=7, va="top", ha="left", color="0.3")


def learning_curves_2x2(rows: list[dict], cfg: dict[str, Any], figures_dir: Path) -> list[Path]:
    boundaries, names = _stage_boundaries(cfg)
    loss_rows = _dedupe_last([r for r in rows if "loss" in r and "type" not in r])
    dev_rows = _dedupe_last([r for r in rows if r.get("type") == "dev_eval"])

    with plt.rc_context({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}):
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.4))

        # (1) training loss: raw faint + 50-update rolling mean.
        ax = axes[0][0]
        if loss_rows:
            ld = pd.DataFrame(loss_rows).sort_values("update")
            ax.plot(ld["update"], ld["loss"], color="tab:blue", alpha=0.25, lw=0.7, label="raw")
            roll = ld["loss"].rolling(50, min_periods=1).mean()
            ax.plot(ld["update"], roll, color="tab:blue", lw=1.3, label="50-update mean")
        ax.set_ylabel("training loss")
        ax.legend(loc="upper right")
        _stage_lines(ax, boundaries, names)

        # (2) dev caption NLL.
        ax = axes[0][1]
        if dev_rows:
            dd = pd.DataFrame(dev_rows).sort_values("update")
            if "caption_dev_nll" in dd:
                ax.plot(dd["update"], dd["caption_dev_nll"], marker="o", ms=2.5, color="tab:green",
                        label="caption dev NLL")
        ax.set_ylabel("caption dev NLL")
        ax.legend(loc="upper right")
        _stage_lines(ax, boundaries, names)

        # (3) dev answer NLL, both suites x {correct, complementary, no-image}.
        ax = axes[1][0]
        if dev_rows:
            dd = pd.DataFrame(dev_rows).sort_values("update")
            styles = {"correct": "-", "complementary": "--", "no_image": ":"}
            colors = {"grounding": "tab:red", "natural_vqa": "tab:purple"}
            for suite in SUITES:
                for cond, ls in styles.items():
                    col = f"{suite}_nll_{cond}"
                    if col in dd:
                        ax.plot(dd["update"], dd[col], ls=ls, color=colors[suite], lw=1.0,
                                label=f"{suite[:4]} {cond[:4]}")
        ax.set_ylabel("dev answer NLL")
        ax.set_xlabel("optimizer update")
        ax.legend(ncol=2, loc="upper right", fontsize=7)
        _stage_lines(ax, boundaries, names)

        # (4) image-swap penalty delta NLL, both suites.
        ax = axes[1][1]
        if dev_rows:
            dd = pd.DataFrame(dev_rows).sort_values("update")
            for suite, color in (("grounding", "tab:red"), ("natural_vqa", "tab:purple")):
                col = f"{suite}_delta_nll"
                if col in dd:
                    ax.plot(dd["update"], dd[col], marker="o", ms=2.5, color=color, label=suite)
            ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_ylabel("ΔNLL (compl. − correct)")
        ax.set_xlabel("optimizer update")
        ax.legend(loc="upper right")
        _stage_lines(ax, boundaries, names)

        fig.suptitle(f"Task 2 learning curves — {cfg.get('run_id', '')}", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = figures_dir / "learning_curves_2x2.pdf"
        fig.savefig(out)  # vector PDF
        plt.close(fig)
    return [out]


# =================================================================================================
# Driver
# =================================================================================================

def analyze_v2(cfg: dict[str, Any]) -> dict[str, list[Path]]:
    run_id = cfg.get("run_id", "vlm_stages")
    results = _results_dir()
    figures_dir = ensure_dir(results / "figures")
    tables_dir = ensure_dir(results / "tables")

    examples_path = results / f"{run_id}_final_examples.jsonl"
    if not examples_path.exists():
        raise SystemExit(f"missing per-example dump: {examples_path}")
    examples = read_jsonl(os.fspath(examples_path))

    log_path = results / "train" / f"{run_id}.jsonl"
    rows = read_jsonl(os.fspath(log_path)) if log_path.exists() else []
    if not rows:
        print(f"[task2.analysis_v2] WARNING: training log not found for {run_id}; figure limited.")

    written: dict[str, list[Path]] = {}
    written["final_cis"] = final_cis(examples, tables_dir)
    written["qualitative_v2"] = qualitative_v2(examples, tables_dir)
    written["learning_curves_2x2"] = learning_curves_2x2(rows, cfg, figures_dir)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 follow-up analysis (CPU-only).")
    parser.add_argument("--config", default="configs/task2/vlm_stages.yaml")
    parser.add_argument("--drive-root", default=None)
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    cfg.setdefault("run_id", "vlm_stages")

    written = analyze_v2(cfg)
    total = sum(len(v) for v in written.values())
    print(f"\n[task2.analysis_v2] wrote {total} files:")
    for section, paths in written.items():
        for p in paths:
            try:
                rel = Path(p).relative_to(REPO_ROOT)
            except ValueError:
                rel = Path(p)
            print(f"  [{section}] {rel}")


if __name__ == "__main__":
    main()
