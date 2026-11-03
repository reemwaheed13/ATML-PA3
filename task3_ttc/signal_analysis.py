"""Step 3 signal analysis on the eval split at N = 32 (full pool as the VisConf calibration set).

`python -m task3_ttc.signal_analysis --config configs/task3/offline.yaml`

Writes results/task3/signals.json:
  * within-question Spearman of each score (length, L, K; MathVista also V, Z, Z_token) with
    correctness and with length (averaged over questions where defined, with the count);
  * quartile gap: correct-rate(top 8) - correct-rate(bottom 8) per score (top = shortest 8 for
    length), averaged over all eval questions and over mixed-correctness questions (with counts);
  * MathVista: per-rollout Pearson of c_{i,t} and v_{i,t} over tokens (skipping zero-variance
    rollouts, counted), averaged within a question then summarized across questions (mean, median,
    IQR, fraction negative);
  * covariance check |(Z_token - Z) - Cov_t(c,v)| (population) over all rollouts (max, mean);
and results/task3/trajectories.json (per-token k/c/v/logp for the highest- and lowest-Z rollouts of
the first three mixed-correctness MathVista eval questions in permutation order).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from common.config import build_arg_parser, load_yaml
from common.paths import REPO_ROOT, ensure_dir
from . import data as t3data
from . import signals
from .offline import _sig_params
from .splits import make_splits

TRAJ_RULE = ("first 3 MathVista eval questions in split permutation order whose 32-rollout pool "
             "contains both correct and incorrect rollouts; within each, the rollouts with the "
             "highest and lowest Z_i (VisConf calibrated over the full pool)")


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1)
    for val in np.unique(a):
        m = a == val
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _full_pool_c(kl_list: list[np.ndarray], low: float, high: float, eps: float) -> list[np.ndarray]:
    pooled = np.concatenate(kl_list) if kl_list else np.zeros(0)
    if pooled.size:
        q_lo = float(np.percentile(pooled, low * 100.0, method="linear"))
        q_hi = float(np.percentile(pooled, high * 100.0, method="linear"))
    else:
        q_lo = q_hi = 0.0
    span = max(q_hi - q_lo, eps)
    return [np.clip((kl.astype(np.float64) - q_lo) / span, 0.0, 1.0) for kl in kl_list]


def _question_scores(q, by_rid, has_vision, low, high, eps):
    """Per-rollout arrays (rollout-id sorted) + scores + token arrays for one question."""
    rolls = q.rollouts
    order = np.argsort([int(r["rollout_id"]) for r in rolls], kind="stable")
    rolls = [rolls[int(i)] for i in order]
    rid = [int(r["rollout_id"]) for r in rolls]
    correct = np.array([by_rid[int(r)] for r in rid], dtype=float)
    length = np.array([int(r["length"]) for r in rolls], dtype=float)
    L = np.array([signals.mean_logprob(r) for r in rolls], dtype=float)
    K = np.array([signals.self_certainty(r) for r in rolls], dtype=float)
    scores = {"length": length, "L": L, "K": K}
    extra = {}
    if has_vision:
        kl = [np.asarray(r["kl_u_p"], dtype=np.float64) for r in rolls]
        vt = [signals._token_visual(r).astype(np.float64) for r in rolls]
        lp = [np.asarray(r["logp"], dtype=np.float64) for r in rolls]
        c = _full_pool_c(kl, low, high, eps)
        V = np.array([float(v.mean()) if v.size else 0.0 for v in vt])
        Z = np.array([float(v.mean() if v.size else 0.0) * float(ci.mean() if ci.size else 0.0)
                      for v, ci in zip(vt, c)])
        Ztok = np.array([float((ci * v).mean()) if ci.size else 0.0 for ci, v in zip(c, vt)])
        scores.update({"V": V, "Z": Z, "Ztok": Ztok})
        extra = {"kl": kl, "v": vt, "logp": lp, "c": c, "Z": Z, "rid": rid}
    return correct, length, scores, extra


def run(cfg: dict[str, Any]) -> None:
    low, high, eps = _sig_params(cfg)
    out: dict[str, Any] = {"trajectories_rule": TRAJ_RULE, "N": 32}
    traj: dict[str, Any] = {"rule": TRAJ_RULE, "cases": []}

    for spec in cfg["data"]["models"]:
        dataset = str(spec["dataset"])
        has_vision = spec.get("type") == "vlm"
        questions = t3data.load_joined(cfg, spec)
        qmap = {q.sample_id: q for q in questions}
        gmap = t3data.compute_groups(cfg, spec, questions, use_cache=True, progress_every=100)
        split = make_splits(list(qmap), seed=int(cfg["splits"]["seed"]),
                            dev_size=int(cfg["splits"]["dev_size"]))
        eval_ids = split["eval"]

        score_names = ["length", "L", "K"] + (["V", "Z", "Ztok"] if has_vision else [])
        sp_corr = {s: [] for s in score_names}
        sp_len = {s: [] for s in score_names}
        quart_all = {s: [] for s in score_names}
        quart_mixed = {s: [] for s in score_names}
        cv_per_q: list[float] = []
        cv_skipped = 0
        cov_abs: list[float] = []
        traj_found = 0

        for sid in eval_ids:
            q = qmap[sid]
            by_rid = {int(x): bool(c) for x, c in zip(gmap[sid]["rollout_ids"], gmap[sid]["correct"])}
            correct, length, scores, extra = _question_scores(q, by_rid, has_vision, low, high, eps)
            mixed = bool(correct.any() and not correct.all())

            for s in score_names:
                v = scores[s]
                c1 = _spearman(v, correct)
                c2 = _spearman(v, length)
                if not np.isnan(c1):
                    sp_corr[s].append(c1)
                if not np.isnan(c2):
                    sp_len[s].append(c2)
                asc = np.argsort(v, kind="stable")                 # ascending; ties keep rid order
                if s == "length":
                    top, bot = asc[:8], asc[-8:]                   # shortest 8 vs longest 8
                else:
                    top, bot = asc[-8:], asc[:8]                   # highest 8 vs lowest 8
                gap = float(correct[top].mean() - correct[bot].mean())
                quart_all[s].append(gap)
                if mixed:
                    quart_mixed[s].append(gap)

            if has_vision:
                cors = []
                for ci, vi in zip(extra["c"], extra["v"]):
                    if ci.std() == 0 or vi.std() == 0:
                        cv_skipped += 1
                        continue
                    cors.append(float(np.corrcoef(ci, vi)[0, 1]))
                if cors:
                    cv_per_q.append(float(np.mean(cors)))
                for ci, vi in zip(extra["c"], extra["v"]):
                    cov = float(((ci - ci.mean()) * (vi - vi.mean())).mean())  # population ddof=0
                    ztok = float((ci * vi).mean()) if ci.size else 0.0
                    z = (float(vi.mean()) if vi.size else 0.0) * (float(ci.mean()) if ci.size else 0.0)
                    cov_abs.append(abs((ztok - z) - cov))
                if mixed and traj_found < 3:
                    Z = extra["Z"]
                    hi, lo = int(np.argmax(Z)), int(np.argmin(Z))
                    traj["cases"].append({
                        "dataset": dataset, "sample_id": sid,
                        "highest_Z": _traj_entry(extra, hi), "lowest_Z": _traj_entry(extra, lo)})
                    traj_found += 1

        ds: dict[str, Any] = {
            "spearman_with_correctness": {s: _mean_count(sp_corr[s]) for s in score_names},
            "spearman_with_length": {s: _mean_count(sp_len[s]) for s in score_names},
            "quartile_gap": {s: {"all": _mean_count(quart_all[s]),
                                 "mixed": _mean_count(quart_mixed[s])} for s in score_names},
        }
        if has_vision:
            arr = np.array(cv_per_q) if cv_per_q else np.zeros(0)
            ds["token_c_v_pearson"] = {
                "n_questions": int(arr.size),
                "skipped_zero_variance_rollouts": int(cv_skipped),
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "median": float(np.median(arr)) if arr.size else float("nan"),
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if arr.size else float("nan"),
                "fraction_negative": float((arr < 0).mean()) if arr.size else float("nan")}
            ca = np.array(cov_abs) if cov_abs else np.zeros(0)
            ds["covariance_check"] = {"max_abs": float(ca.max()) if ca.size else 0.0,
                                      "mean_abs": float(ca.mean()) if ca.size else 0.0,
                                      "n_rollouts": int(ca.size)}
        out[dataset] = ds

    out_dir = ensure_dir(REPO_ROOT / "results" / "task3")
    with open(os.fspath(out_dir / "signals.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.fspath(out_dir / "trajectories.json"), "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2)
    print(f"[task3.signal_analysis] wrote signals.json and trajectories.json "
          f"({len(traj['cases'])} trajectory cases).")


def _traj_entry(extra: dict, i: int) -> dict:
    return {"rollout_id": int(extra["rid"][i]),
            "k": extra["kl"][i].astype(float).tolist(),
            "c": extra["c"][i].astype(float).tolist(),
            "v": extra["v"][i].astype(float).tolist(),
            "logp": extra["logp"][i].astype(float).tolist()}


def _mean_count(xs: list[float]) -> dict[str, Any]:
    a = np.array(xs, dtype=float) if xs else np.zeros(0)
    return {"mean": float(a.mean()) if a.size else float("nan"), "count": int(a.size)}


def main() -> None:
    parser = build_arg_parser("Task 3 Step-3 signal analysis.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
