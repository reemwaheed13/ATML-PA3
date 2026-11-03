"""Task 2 analysis: turn the saved run outputs into the handout's Required Evidence. (p9, p15)

CPU-only; matplotlib + pandas; no model runs; facts only (no interpretation prose). Reads a run's
training log, final results, final per-example dump, and feature-cache log, and writes figures
(results/task2/figures/*.png) and tables (results/task2/tables/*.csv, with a .md copy of each).

    python -m task2_vlm.analysis --config configs/task2/vlm_stages.yaml

Inputs (run_id taken from the config):
  * training log:  results/task2/train/<run_id>.jsonl
                   (fallback: DRIVE_ROOT/task2/results/<run_id>/train.jsonl)
  * final results: results/task2/<run_id>_final.json
  * per-example:   results/task2/<run_id>_final_examples.jsonl
  * feature cache: results/task2/feature_cache.jsonl

Each section degrades gracefully: a missing input is reported and that section is skipped rather
than crashing (e.g. a run that predates the calibration / mixture-summary / per-example rows).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless / CPU-only

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.config import build_arg_parser, load_yaml, run_id_from_config  # noqa: E402
from common.logging_utils import read_jsonl  # noqa: E402
from common.paths import REPO_ROOT, ensure_dir  # noqa: E402
from .scoring import normalize_answer  # noqa: E402

# COCO val2014 images (all Task-2 eval images live in val2014) are fetched on demand for the
# qualitative figures and cached locally so repeated runs do not re-download.
_COCO_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"

CONDITIONS = ["correct", "complementary_swap", "mismatched", "no_image"]
SUITES = ["grounding", "natural_vqa"]

# Deterministic qualitative-pair selection rule (stated in the CSV + figure caption).
QUAL_RULE = (
    "Final-test complementary pairs sorted by pair_id; first match per category: "
    "(a) both images scored correct under correct-image input; "
    "(b) identical normalized prediction on both images under correct-image input; "
    "(c) under complementary-swap input, both images' predictions equal the partner image's "
    "canonical reference; "
    "(d) under correct-image input, one image is wrong (score<0.5) while its partner is correct "
    "(score>=0.5)."
)
QUAL_CATEGORY_RULE = {
    "a_both_correct": "both images correct (correct-image input)",
    "b_identical_answer": "identical prediction on both images (correct-image input)",
    "c_follows_partner_on_swap": "both predictions equal the partner's canonical reference (swap input)",
    "d_wrong_vs_right_partner": "one image wrong, its partner right (correct-image input)",
}


# =================================================================================================
# Path resolution + small IO helpers
# =================================================================================================

def _results_dir() -> Path:
    return REPO_ROOT / "results" / "task2"


def _train_log_path(run_id: str) -> Optional[Path]:
    """Committed repo log first, then the DRIVE_ROOT working copy; None if neither exists."""
    repo = _results_dir() / "train" / f"{run_id}.jsonl"
    if repo.exists():
        return repo
    drive = os.environ.get("DRIVE_ROOT")
    if drive:
        alt = Path(drive) / "task2" / "results" / run_id / "train.jsonl"
        if alt.exists():
            return alt
    return repo if repo.exists() else None


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: Any) -> str:
    """Render one cell for a markdown table (floats to 6 sig figs; None/NaN blank)."""
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.6g}"
    return str(v).replace("|", r"\|")


def _df_to_md(df: pd.DataFrame) -> str:
    """GitHub-flavoured markdown table without a tabulate dependency."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _write_table(df: pd.DataFrame, name: str, tables_dir: Path, *, caption: str = "") -> list[Path]:
    """Write a DataFrame as <name>.csv and <name>.md (optional leading caption line in the .md)."""
    csv_path = tables_dir / f"{name}.csv"
    md_path = tables_dir / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md = (f"<!-- {caption} -->\n\n" if caption else "") + _df_to_md(df)
    md_path.write_text(md, encoding="utf-8")
    return [csv_path, md_path]


def _dedupe_last(rows: list[dict], key: str = "update") -> list[dict]:
    """Keep the LAST row for each value of `key` (a resume replays some updates), sorted by key."""
    by_key: dict[int, dict] = {}
    for r in rows:
        if key in r:
            by_key[int(r[key])] = r
    return [by_key[k] for k in sorted(by_key)]


def _stage_boundaries(cfg: dict[str, Any]) -> tuple[list[int], list[str]]:
    """Cumulative stage-end updates (excluding the final end) + stage names, from the config."""
    stages = cfg.get("stages", [])
    updates = [int(s["updates"]) for s in stages]
    names = [str(s["name"]) for s in stages]
    cum = np.cumsum(updates).tolist()
    return cum[:-1], names  # e.g. [2250, 3750]


def _draw_stage_lines(ax, boundaries: list[int], names: list[str]) -> None:
    ymax = ax.get_ylim()[1]
    for i, b in enumerate(boundaries):
        ax.axvline(b, color="0.4", ls="--", lw=0.8)
        label = names[i + 1] if i + 1 < len(names) else f"stage@{b}"
        ax.text(b, ymax, f" {label}\n @{b}", fontsize=6, va="top", ha="left", color="0.3")


# =================================================================================================
# 1) Learning curves + dev-metrics CSV
# =================================================================================================

def learning_curves(rows: list[dict], cfg: dict[str, Any], figures_dir: Path,
                    tables_dir: Path) -> list[Path]:
    written: list[Path] = []
    boundaries, names = _stage_boundaries(cfg)

    loss_rows = _dedupe_last([r for r in rows if "loss" in r and "type" not in r])
    dev_rows = _dedupe_last([r for r in rows if r.get("type") == "dev_eval"])

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)

    # (1) training loss: raw faint + 50-update rolling mean.
    ax = axes[0]
    if loss_rows:
        ld = pd.DataFrame(loss_rows).sort_values("update")
        ax.plot(ld["update"], ld["loss"], color="tab:blue", alpha=0.25, lw=0.7, label="loss (raw)")
        roll = ld["loss"].rolling(50, min_periods=1).mean()
        ax.plot(ld["update"], roll, color="tab:blue", lw=1.5, label="loss (50-update mean)")
    ax.set_ylabel("training loss")
    ax.legend(fontsize=7, loc="upper right")
    _draw_stage_lines(ax, boundaries, names)

    # (2) dev caption NLL.
    ax = axes[1]
    if dev_rows:
        dd = pd.DataFrame(dev_rows).sort_values("update")
        if "caption_dev_nll" in dd:
            ax.plot(dd["update"], dd["caption_dev_nll"], marker="o", ms=3, color="tab:green",
                    label="caption dev NLL")
    ax.set_ylabel("caption dev NLL")
    ax.legend(fontsize=7, loc="upper right")
    _draw_stage_lines(ax, boundaries, names)

    # (3) dev answer NLL, both suites x {correct, complementary, no-image}.
    ax = axes[2]
    if dev_rows:
        dd = pd.DataFrame(dev_rows).sort_values("update")
        styles = {"correct": "-", "complementary": "--", "no_image": ":"}
        colors = {"grounding": "tab:red", "natural_vqa": "tab:purple"}
        for suite in SUITES:
            for cond, ls in styles.items():
                col = f"{suite}_nll_{cond}"
                if col in dd:
                    ax.plot(dd["update"], dd[col], ls=ls, color=colors[suite], lw=1.1,
                            label=f"{suite} {cond}")
    ax.set_ylabel("dev answer NLL")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    _draw_stage_lines(ax, boundaries, names)

    # (4) image-swap penalty delta NLL, both suites.
    ax = axes[3]
    if dev_rows:
        dd = pd.DataFrame(dev_rows).sort_values("update")
        for suite, color in (("grounding", "tab:red"), ("natural_vqa", "tab:purple")):
            col = f"{suite}_delta_nll"
            if col in dd:
                ax.plot(dd["update"], dd[col], marker="o", ms=3, color=color, label=f"{suite} ΔNLL")
        ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_ylabel("ΔNLL (complementary − correct)")
    ax.set_xlabel("optimizer update")
    ax.legend(fontsize=7, loc="upper right")
    _draw_stage_lines(ax, boundaries, names)

    fig.suptitle(f"Task 2 learning curves — {cfg.get('run_id', '')}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = figures_dir / "learning_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(out)

    # Dev metrics CSV.
    if dev_rows:
        dd = pd.DataFrame(dev_rows).sort_values("update")
        dd = dd[[c for c in dd.columns if c != "type"]]
        written += _write_table(dd, "dev_metrics", tables_dir,
                                 caption="Development metrics vs optimizer update.")
    return written


# =================================================================================================
# 2) Mixture counts per stage (observed vs configured)
# =================================================================================================

def mixture_counts(rows: list[dict], cfg: dict[str, Any], tables_dir: Path) -> list[Path]:
    summaries = [r for r in rows if r.get("type") == "stage_mixture_summary"]
    if not summaries:
        print("[task2.analysis] no stage_mixture_summary rows; skipping mixture-counts table.")
        return []

    cfg_mix = {str(s["name"]): dict(s.get("mixture", {})) for s in cfg.get("stages", [])}
    records: list[dict] = []
    for s in summaries:
        stage = s.get("stage", "")
        draws = s.get("draws", {})
        total = int(s.get("total_draws", sum(draws.values()))) or 1
        configured = cfg_mix.get(stage, {})
        cfg_total = sum(configured.values()) or 1.0
        for comp, cnt in draws.items():
            records.append({
                "stage": stage,
                "component": comp,
                "draws": int(cnt),
                "observed_share": cnt / total,
                "configured_prob": configured.get(comp, 0.0) / cfg_total,
            })
    df = pd.DataFrame(records)
    return _write_table(df, "mixture_counts", tables_dir,
                        caption="Per-stage mixture draws: observed share vs configured probability.")


# =================================================================================================
# 3) Final results + advantages
# =================================================================================================

def _condition_row(cond: str, data: dict) -> dict:
    row: dict[str, Any] = {"condition": cond}
    row["accuracy_overall"] = data.get("accuracy_overall")
    for qtype, val in (data.get("accuracy_by_type") or {}).items():
        row[f"accuracy_{qtype}"] = val
    row["pair_accuracy"] = data.get("pair_accuracy")
    row["identical_answer_rate"] = data.get("identical_answer_rate")
    row["empty_answer_rate"] = data.get("empty_answer_rate")
    row["mean_reference_nll"] = data.get("mean_reference_nll")
    row["gen_seconds_per_answer"] = data.get("gen_seconds_per_answer")
    row["follows_partner_rate"] = data.get("follows_partner_rate")  # swap only; else None
    return row


def final_results(final: dict, tables_dir: Path) -> list[Path]:
    written: list[Path] = []
    for suite in SUITES:
        if suite not in final:
            continue
        rows = [_condition_row(cond, final[suite][cond])
                for cond in CONDITIONS if cond in final[suite]]
        df = pd.DataFrame(rows)
        written += _write_table(df, f"final_results_{suite}", tables_dir,
                                 caption=f"Final-test metrics for {suite} across the 4 image conditions.")

    # Advantages table (correct vs complementary): accuracy + NLL with 95% CIs, per suite.
    adv_rows: list[dict] = []
    for suite in SUITES:
        adv = final.get(f"{suite}_advantage")
        if not adv:
            continue
        acc_ci = adv.get("accuracy_advantage_ci95") or [None, None]
        nll_ci = adv.get("nll_advantage_ci95") or [None, None]
        adv_rows.append({
            "suite": suite,
            "accuracy_advantage_mean": adv.get("accuracy_advantage_mean"),
            "accuracy_ci95_low": acc_ci[0],
            "accuracy_ci95_high": acc_ci[1],
            "nll_advantage_mean": adv.get("nll_advantage_mean", adv.get("nll_advantage")),
            "nll_ci95_low": nll_ci[0] if nll_ci else None,
            "nll_ci95_high": nll_ci[1] if nll_ci else None,
            "n_pairs": adv.get("n_pairs"),
        })
    if adv_rows:
        written += _write_table(pd.DataFrame(adv_rows), "final_advantages", tables_dir,
                                caption="Correct-vs-complementary advantage (accuracy, NLL) with "
                                        "95% CIs from matched-pair bootstrap.")
    return written


# =================================================================================================
# 4) Qualitative complementary pairs
# =================================================================================================

def _coco_image(image_id: int, cache_dir: Path):
    """Return a PIL image for a val2014 id, fetching into `cache_dir` if absent; None on failure."""
    from PIL import Image
    fname = f"COCO_val2014_{int(image_id):012d}.jpg"
    path = cache_dir / fname
    if not path.exists():
        try:
            urllib.request.urlretrieve(_COCO_URL.format(image_id=int(image_id)), os.fspath(path))
        except Exception as exc:  # noqa: BLE001 — offline / fetch failure => placeholder
            print(f"[task2.analysis] could not fetch image {image_id}: {exc}")
            return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _pairs_by_suite(examples: list[dict], suite: str) -> dict[Any, dict]:
    """Assemble per-pair correct/swap predictions keyed by instance (pair_id, question, target)."""
    pairs: dict[Any, dict[str, dict]] = {}
    for r in examples:
        if r.get("suite") != suite:
            continue
        cond = r.get("condition")
        if cond not in ("correct", "complementary_swap"):
            continue
        pid = r.get("pair_id")
        inst_key = (r.get("question"), r.get("target"))
        pair = pairs.setdefault(pid, {})
        inst = pair.setdefault(inst_key, {"question": r.get("question"), "target": r.get("target"),
                                          "partner_canonical": r.get("partner_canonical")})
        if cond == "correct":
            inst["correct_image"] = r.get("image_id")
            inst["correct_pred"] = r.get("normalized_prediction")
            inst["correct_score"] = r.get("score")
        else:
            inst["swap_pred"] = r.get("normalized_prediction")
    return pairs


def _select_qualitative(pairs: dict[Any, dict]) -> list[dict]:
    """First final-test pair (sorted by pair_id) matching each of the four categories."""
    selected: list[dict] = []
    for key, rule in QUAL_CATEGORY_RULE.items():
        for pid in sorted(pairs, key=lambda x: str(x)):
            insts = list(pairs[pid].values())
            if len(insts) != 2 or any("correct_pred" not in i for i in insts):
                continue
            lo, hi = sorted(insts, key=lambda i: (str(i.get("target")), str(i.get("correct_image"))))
            cs = [float(i.get("correct_score") or 0.0) for i in (lo, hi)]
            if key == "a_both_correct":
                match = cs[0] >= 0.5 and cs[1] >= 0.5
            elif key == "b_identical_answer":
                match = lo.get("correct_pred") == hi.get("correct_pred")
            elif key == "c_follows_partner_on_swap":
                match = all(i.get("swap_pred") is not None
                            and i.get("swap_pred") == normalize_answer(i.get("partner_canonical") or "")
                            for i in (lo, hi))
            else:  # d_wrong_vs_right_partner
                match = (cs[0] >= 0.5) != (cs[1] >= 0.5)
            if match:
                selected.append({
                    "category": key, "selection_rule": rule, "pair_id": pid,
                    "question": lo.get("question"),
                    "image_left": lo.get("correct_image"), "image_right": hi.get("correct_image"),
                    "ref_left": lo.get("target"), "ref_right": hi.get("target"),
                    "pred_correct_left": lo.get("correct_pred"),
                    "pred_correct_right": hi.get("correct_pred"),
                    "pred_swap_left": lo.get("swap_pred"), "pred_swap_right": hi.get("swap_pred"),
                })
                break
    return selected


def qualitative(examples: list[dict], figures_dir: Path, tables_dir: Path,
                cache_dir: Path) -> list[Path]:
    written: list[Path] = []
    for suite in SUITES:
        pairs = _pairs_by_suite(examples, suite)
        selected = _select_qualitative(pairs)
        if not selected:
            print(f"[task2.analysis] no qualitative pairs found for {suite}; skipping.")
            continue

        df = pd.DataFrame(selected)
        written += _write_table(df, f"qualitative_{suite}", tables_dir, caption=QUAL_RULE)

        nrows = len(selected)
        fig, axes = plt.subplots(nrows, 2, figsize=(8, 3.4 * nrows), squeeze=False)
        for r, item in enumerate(selected):
            for c, side in enumerate(("left", "right")):
                ax = axes[r][c]
                img = _coco_image(item[f"image_{side}"], cache_dir)
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, f"image {item['image_' + side]}\n(unavailable)",
                            ha="center", va="center", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                title = (f"img {item['image_' + side]} | ref: {item[f'ref_{side}']}\n"
                         f"correct→{item[f'pred_correct_{side}']}  "
                         f"swap→{item[f'pred_swap_{side}']}")
                ax.set_title(title, fontsize=7)
            axes[r][0].set_ylabel(f"[{item['category']}]\nQ: {item['question']}", fontsize=7)
        fig.suptitle(f"{suite}: qualitative complementary pairs", fontsize=10)
        fig.text(0.5, 0.005, QUAL_RULE, wrap=True, fontsize=6, ha="center")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        out = figures_dir / f"qualitative_{suite}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
    return written


# =================================================================================================
# 5) Resources
# =================================================================================================

def resources(rows: list[dict], final: dict, feature_rows: list[dict],
              tables_dir: Path) -> list[Path]:
    written: list[Path] = []

    # Trainable parameters by stage (stage_start rows).
    stage_rows = [r for r in rows if r.get("type") == "stage_start"]
    if stage_rows:
        df = pd.DataFrame([{
            "stage_idx": r.get("stage_idx"), "stage": r.get("stage"), "update": r.get("update"),
            "connector_params": r.get("trainable_params", {}).get("connector"),
            "adapter_params": r.get("trainable_params", {}).get("adapters"),
            "total_trainable_params": r.get("trainable_params", {}).get("total"),
        } for r in stage_rows]).sort_values("stage_idx")
        written += _write_table(df, "resources_params_by_stage", tables_dir,
                                caption="Trainable parameters by stage.")

    # Training wall time + peak GPU (resources row; peak = max over resources/peak_gpu rows).
    res = next((r for r in rows if r.get("type") == "resources"), None)
    peak_vals = [int(r["peak_gpu_bytes"]) for r in rows
                 if r.get("type") == "peak_gpu" and "peak_gpu_bytes" in r]
    if res is not None:
        peak_vals.append(int(res.get("peak_gpu_bytes", 0)))
    if res is not None:
        peak = max(peak_vals) if peak_vals else int(res.get("peak_gpu_bytes", 0))
        df = pd.DataFrame([{
            "run_id": res.get("run_id"),
            "wall_seconds": res.get("wall_seconds"),
            "wall_hours": (res.get("wall_seconds") or 0) / 3600.0,
            "peak_gpu_bytes": peak,
            "peak_gpu_gb": peak / 1e9,
            "trainable_params": res.get("trainable_params"),
            "total_params": res.get("total_params"),
        }])
        written += _write_table(df, "resources_training", tables_dir,
                                caption="Training wall time and peak GPU memory.")

    # Connector calibration row.
    calib = next((r for r in rows if r.get("type") == "calibration" and "scale" in r), None)
    if calib is not None:
        df = pd.DataFrame([{
            "rule": calib.get("rule"),
            "text_norm": calib.get("text_norm"),
            "visual_norm_before": calib.get("visual_norm_before"),
            "scale": calib.get("scale"),
        }])
        written += _write_table(df, "resources_calibration", tables_dir,
                                caption="Connector magnitude calibration.")
    else:
        print("[task2.analysis] no calibration row; skipping calibration table.")

    # Feature caching: use the full-run row (max images_total), not the smoke row.
    cache_rows = [r for r in feature_rows if r.get("type") == "feature_cache"]
    if cache_rows:
        full = max(cache_rows, key=lambda r: int(r.get("images_total", 0)))
        df = pd.DataFrame([{
            "images_total": full.get("images_total"),
            "images_cached_now": full.get("images_cached_now"),
            "images_already_cached": full.get("images_already_cached"),
            "wall_seconds": full.get("wall_seconds"),
            "peak_gpu_bytes": full.get("peak_gpu_bytes"),
            "peak_gpu_gb": (full.get("peak_gpu_bytes") or 0) / 1e9,
            "cache_bytes": full.get("cache_bytes"),
            "cache_gb": (full.get("cache_bytes") or 0) / 1e9,
        }])
        written += _write_table(df, "resources_feature_caching", tables_dir,
                                caption="SigLIP feature caching (full-run row).")

    # Per-stage training time + peak GPU (from stage_timing.json, if present).
    st_path = _results_dir() / "stage_timing.json"
    if st_path.exists():
        st = _load_json(st_path)
        rows_st = []
        for s in st.get("stages", []):
            rows_st.append({
                "stage_idx": s.get("stage_idx"),
                "stage": s.get("stage"),
                "start_weights": s.get("start_weights"),
                "seconds_per_update": s.get("seconds_per_update"),
                "timed_updates": s.get("timed_updates"),
                "stage_updates": s.get("stage_updates"),
                "estimated_stage_seconds": s.get("estimated_stage_seconds"),
                "estimated_stage_hours": (s.get("estimated_stage_seconds") or 0) / 3600.0,
                "peak_gpu_bytes": s.get("peak_gpu_bytes"),
                "peak_gpu_gb": (s.get("peak_gpu_bytes") or 0) / 1e9,
            })
        if rows_st:
            cap = (st.get("note", "") + " Reference logged final-process wall seconds: "
                   + str(st.get("final_process_wall_seconds_logged")) + ".")
            written += _write_table(pd.DataFrame(rows_st), "resources_stage_timing", tables_dir,
                                    caption=cap)
    else:
        print("[task2.analysis] no stage_timing.json; skipping per-stage timing table.")

    # Generation time per answer, per suite x condition (from final results).
    gen_rows: list[dict] = []
    for suite in SUITES:
        if suite not in final:
            continue
        for cond in CONDITIONS:
            if cond in final[suite]:
                gen_rows.append({"suite": suite, "condition": cond,
                                 "gen_seconds_per_answer": final[suite][cond].get("gen_seconds_per_answer")})
    if gen_rows:
        written += _write_table(pd.DataFrame(gen_rows), "resources_generation_time", tables_dir,
                                caption="Generation time per answer, per suite and condition.")
    return written


# =================================================================================================
# Driver
# =================================================================================================

def analyze(cfg: dict[str, Any]) -> dict[str, list[Path]]:
    run_id = cfg.get("run_id", "task2_run")
    results = _results_dir()
    figures_dir = ensure_dir(results / "figures")
    tables_dir = ensure_dir(results / "tables")
    cache_dir = ensure_dir(results / "_coco_cache")

    written: dict[str, list[Path]] = {}

    # Training log (learning curves, mixture counts, resources).
    log_path = _train_log_path(run_id)
    rows = read_jsonl(os.fspath(log_path)) if log_path and log_path.exists() else []
    if not rows:
        print(f"[task2.analysis] WARNING: training log not found for {run_id}; "
              f"learning-curve / mixture / resource sections will be limited.")
    written["learning_curves"] = learning_curves(rows, cfg, figures_dir, tables_dir)
    written["mixture_counts"] = mixture_counts(rows, cfg, tables_dir)

    # Final results.
    final_path = results / f"{run_id}_final.json"
    final = _load_json(final_path) if final_path.exists() else {}
    if final:
        written["final_results"] = final_results(final, tables_dir)
    else:
        print(f"[task2.analysis] WARNING: {final_path.name} not found; skipping final-results tables.")

    # Qualitative (per-example dump).
    examples_path = results / f"{run_id}_final_examples.jsonl"
    if examples_path.exists():
        examples = read_jsonl(os.fspath(examples_path))
        written["qualitative"] = qualitative(examples, figures_dir, tables_dir, cache_dir)
    else:
        print(f"[task2.analysis] {examples_path.name} not found; skipping qualitative figures.")

    # Resources (needs feature-cache log too).
    fc_path = results / "feature_cache.jsonl"
    feature_rows = read_jsonl(os.fspath(fc_path)) if fc_path.exists() else []
    written["resources"] = resources(rows, final, feature_rows, tables_dir)

    return written


def main() -> None:
    parser = build_arg_parser("Task 2 analysis: Required Evidence figures + tables (CPU-only).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    cfg.setdefault("run_id", run_id_from_config(args.config, cfg))

    written = analyze(cfg)

    total = sum(len(v) for v in written.values())
    print(f"\n[task2.analysis] wrote {total} files for run '{cfg.get('run_id')}':")
    for section, paths in written.items():
        for p in paths:
            try:
                rel = Path(p).relative_to(REPO_ROOT)
            except ValueError:
                rel = Path(p)
            print(f"  [{section}] {rel}")


if __name__ == "__main__":
    main()
