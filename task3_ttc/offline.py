"""Offline selection experiments over candidate budgets. (handout p12-p13, Steps 1-3)

For budgets N in {1,2,4,8,16,32}, using ten seeded permutations of the 32 rollout IDs PER QUESTION
(nested prefixes, orders reused across methods and reused by the adaptive step), evaluate selection
rules and report accuracy, oracle coverage, and generated-token cost (sum of retained token counts;
FirstAns counts only its first candidate). Average over orders within a question, then across
questions. Ties resolve to the earliest supporting rollout in the current order; RWC ties by average
ranks.

Speed/caching (audit fixes):
  * groups + correctness come from data.compute_groups (cached under SCORER_VERSION).
  * build_question_arrays() returns per-question numpy scalar arrays (rollout_id, group, correct,
    length, L, K, and V for the VLM), cached to DRIVE_ROOT/task3/cache/<dataset>_rollout_scalars_
    <SCORER_VERSION>.parquet, plus the per-token kl/v_t arrays for VisConf.
  * VisConf Z and Z_token are computed once per (sample_id, order_idx, N) and cached to
    DRIVE_ROOT/task3/cache/<dataset>_visconf_<SCORER_VERSION>.npz; the adaptive checkpoints 4/8/16
    are the same nested prefixes and reuse this cache.

Both offline.py and adaptive.py (and signal_analysis.py) import build_question_arrays / the orders
helpers / the VisConf cache from here so there is one source of truth.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from common.config import build_arg_parser, load_yaml
from common.paths import REPO_ROOT, SPLITS_DIR, ensure_dir, task3_dir
from . import data as t3data
from . import signals
from .splits import make_splits

NO_ANSWER_GROUP = 0


# =================================================================================================
# Per-question orders (10 permutations per question, saved and reused)
# =================================================================================================

def make_question_orders(qa: dict[str, dict], *, n_orders: int = 10,
                         seed: int = 6304) -> dict[str, list[list[int]]]:
    """Ten rollout-id permutations per question, derived from SeedSequence([seed, q_index, k]).

    q_index is the question's position in the lexicographically sorted ID list; the permutation
    acts on the POSITIONS of the rollout-id-sorted rollouts and is emitted as actual rollout_ids.
    """
    ids_sorted = sorted(qa)                              # lexicographic (string) order
    out: dict[str, list[list[int]]] = {}
    for q_index, sid in enumerate(ids_sorted):
        rids = [int(x) for x in qa[sid]["rollout_id"]]   # already rollout-id sorted
        perms: list[list[int]] = []
        for k in range(n_orders):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(q_index), int(k)]))
            perm = rng.permutation(len(rids))
            perms.append([rids[int(p)] for p in perm])
        out[sid] = perms
    return out


def _orders_path(dataset: str):
    return SPLITS_DIR / "task3" / f"{dataset}_orders.json"


def load_or_make_orders(dataset: str, qa: dict[str, dict], *, n_orders: int,
                        seed: int) -> dict[str, list[list[int]]]:
    """Load splits/task3/<dataset>_orders.json if present (asserting it matches), else write it."""
    generated = make_question_orders(qa, n_orders=n_orders, seed=seed)
    path = _orders_path(dataset)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        saved = {str(k): [[int(r) for r in o] for o in v] for k, v in saved.items()}
        gen_cmp = {k: [[int(r) for r in o] for o in v] for k, v in generated.items()}
        assert saved == gen_cmp, f"saved orders {path} do not match the seeded regeneration"
        return saved
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(generated, f)
    return generated


# =================================================================================================
# Per-question arrays (scalars cached to parquet; tokens kept for VisConf)
# =================================================================================================

def _scalars_cache_path(dataset: str):
    try:
        return task3_dir("cache") / f"{dataset}_rollout_scalars_{t3data.SCORER_VERSION}.parquet"
    except Exception:
        return None


def _visconf_cache_path(dataset: str):
    try:
        return task3_dir("cache") / f"{dataset}_visconf_{t3data.SCORER_VERSION}.npz"
    except Exception:
        return None


def _group_maps(answers: list, groups: np.ndarray, correct: np.ndarray):
    rep = t3data.group_representatives(answers, [int(g) for g in groups])
    gc: dict[int, bool] = {}
    for g, c in zip(groups, correct):
        gc[int(g)] = gc.get(int(g), False) or bool(c)
    return rep, gc


def build_question_arrays(cfg: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, dict], bool]:
    """Return ({sample_id -> per-question numpy arrays}, has_vision); cache scalars to parquet."""
    dataset = str(spec["dataset"])
    has_vision = spec.get("type") == "vlm"
    cache = _scalars_cache_path(dataset)
    if cache is not None and cache.exists():
        return _load_scalars(cache, has_vision), has_vision

    questions = t3data.load_joined(cfg, spec)
    gmap = t3data.compute_groups(cfg, spec, questions, use_cache=True, progress_every=100)
    qa: dict[str, dict] = {}
    for q in questions:
        rolls = q.rollouts
        order = np.argsort([int(r["rollout_id"]) for r in rolls], kind="stable")
        rolls = [rolls[int(i)] for i in order]
        rid = np.array([int(r["rollout_id"]) for r in rolls], dtype=np.int64)

        g = gmap[q.sample_id]
        by_rid = {int(x): (int(gg), bool(cc))
                  for x, gg, cc in zip(g["rollout_ids"], g["groups"], g["correct"])}
        assert set(by_rid) == set(int(x) for x in rid), \
            f"compute_groups rollout_ids != loaded rollouts for {q.sample_id}"
        groups = np.array([by_rid[int(r)][0] for r in rid], dtype=np.int64)
        correct = np.array([by_rid[int(r)][1] for r in rid], dtype=bool)

        length = np.array([int(r["length"]) for r in rolls], dtype=np.int64)
        L = np.array([signals.mean_logprob(r) for r in rolls], dtype=np.float64)
        K = np.array([signals.self_certainty(r) for r in rolls], dtype=np.float64)
        answers = [r.get("extracted") for r in rolls]
        rep, gc = _group_maps(answers, groups, correct)
        entry = {"rollout_id": rid, "group": groups, "correct": correct, "length": length,
                 "L": L, "K": K, "answers": answers, "group_rep": rep, "group_correct": gc,
                 "gold": q.gold, "metadata": q.metadata, "kl_tokens": None, "v_tokens": None}
        if has_vision:
            entry["V"] = np.array([signals.visual_engagement(r) for r in rolls], dtype=np.float64)
            entry["kl_tokens"] = [np.asarray(r["kl_u_p"], dtype=np.float32) for r in rolls]
            entry["v_tokens"] = [signals._token_visual(r).astype(np.float32) for r in rolls]
        else:
            entry["V"] = np.zeros(len(rolls), dtype=np.float64)
        qa[q.sample_id] = entry

    if cache is not None:
        _save_scalars(cache, dataset, qa, has_vision)
    return qa, has_vision


def _save_scalars(path, dataset: str, qa: dict[str, dict], has_vision: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols: dict[str, list] = {k: [] for k in
                             ("sample_id", "rollout_id", "group", "correct", "length", "L", "K",
                              "V", "answer")}
    for sid in sorted(qa):
        e = qa[sid]
        for i in range(len(e["rollout_id"])):
            cols["sample_id"].append(sid)
            cols["rollout_id"].append(int(e["rollout_id"][i]))
            cols["group"].append(int(e["group"][i]))
            cols["correct"].append(bool(e["correct"][i]))
            cols["length"].append(int(e["length"][i]))
            cols["L"].append(float(e["L"][i]))
            cols["K"].append(float(e["K"][i]))
            cols["V"].append(float(e["V"][i]))
            cols["answer"].append(e["answers"][i] if e["answers"][i] is not None else "")
    try:
        ensure_dir(path.parent)
        pq.write_table(pa.table(cols), os.fspath(path))
    except Exception:
        pass


def _load_scalars(path, has_vision: bool) -> dict[str, dict]:
    import pyarrow.parquet as pq
    tbl = pq.read_table(path)
    d = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
    qa: dict[str, dict] = {}
    rows: dict[str, list[int]] = {}
    for i, sid in enumerate(d["sample_id"]):
        rows.setdefault(str(sid), []).append(i)
    for sid, idxs in rows.items():
        idxs.sort(key=lambda i: int(d["rollout_id"][i]))
        rid = np.array([int(d["rollout_id"][i]) for i in idxs], dtype=np.int64)
        groups = np.array([int(d["group"][i]) for i in idxs], dtype=np.int64)
        correct = np.array([bool(d["correct"][i]) for i in idxs], dtype=bool)
        length = np.array([int(d["length"][i]) for i in idxs], dtype=np.int64)
        L = np.array([float(d["L"][i]) for i in idxs], dtype=np.float64)
        K = np.array([float(d["K"][i]) for i in idxs], dtype=np.float64)
        V = np.array([float(d["V"][i]) for i in idxs], dtype=np.float64)
        answers = [(d["answer"][i] or None) for i in idxs]
        rep, gc = _group_maps(answers, groups, correct)
        qa[str(sid)] = {"rollout_id": rid, "group": groups, "correct": correct, "length": length,
                        "L": L, "K": K, "V": V, "answers": answers, "group_rep": rep,
                        "group_correct": gc, "gold": None, "metadata": None,
                        "kl_tokens": None, "v_tokens": None}
    return qa


def _attach_tokens(cfg: dict[str, Any], spec: dict[str, Any], qa: dict[str, dict]) -> None:
    """Reload the per-token kl/v_t arrays (needed for VisConf when scalars came from the cache)."""
    questions = t3data.load_joined(cfg, spec)
    for q in questions:
        if q.sample_id not in qa:
            continue
        rolls = q.rollouts
        order = np.argsort([int(r["rollout_id"]) for r in rolls], kind="stable")
        rolls = [rolls[int(i)] for i in order]
        qa[q.sample_id]["kl_tokens"] = [np.asarray(r["kl_u_p"], dtype=np.float32) for r in rolls]
        qa[q.sample_id]["v_tokens"] = [signals._token_visual(r).astype(np.float32) for r in rolls]


# =================================================================================================
# VisConf cache (Z and Z_token per (sample_id, order_idx, N))
# =================================================================================================

def _sig_params(cfg: dict[str, Any]) -> tuple[float, float, float]:
    s = cfg.get("signals", {})
    return (float(s.get("percentile_low", 0.05)), float(s.get("percentile_high", 0.95)),
            float(s.get("eps", 1e-8)))


def visconf_prefix(kl_list: list[np.ndarray], v_list: list[np.ndarray],
                   low: float, high: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Z_i = V_i*C_i and Z_token_i = mean_t c_{i,t} v_{i,t} for a candidate set (calibrated over it)."""
    if not kl_list:
        return np.zeros(0), np.zeros(0)
    pooled = np.concatenate(kl_list)
    if pooled.size:
        q_lo = float(np.percentile(pooled, low * 100.0, method="linear"))
        q_hi = float(np.percentile(pooled, high * 100.0, method="linear"))
    else:
        q_lo = q_hi = 0.0
    span = max(q_hi - q_lo, eps)
    z = np.zeros(len(kl_list), dtype=np.float64)
    zt = np.zeros(len(kl_list), dtype=np.float64)
    for i, (kl, v) in enumerate(zip(kl_list, v_list)):
        c = np.clip((kl.astype(np.float64) - q_lo) / span, 0.0, 1.0)
        z[i] = (float(v.mean()) if v.size else 0.0) * (float(c.mean()) if c.size else 0.0)
        zt[i] = float((c * v).mean()) if c.size else 0.0
    return z, zt


def load_or_build_visconf(cfg: dict[str, Any], spec: dict[str, Any], qa: dict[str, dict],
                          orders: dict[str, list[list[int]]], budgets: list[int]) -> dict[str, dict]:
    """Return {sample_id -> {"Z": {(o,N): arr}, "Ztok": {(o,N): arr}}}; build+cache npz if missing."""
    dataset = str(spec["dataset"])
    path = _visconf_cache_path(dataset)
    n_orders = len(next(iter(orders.values())))
    sids = sorted(qa)
    if path is not None and path.exists():
        return _decode_visconf(*_load_visconf_npz(path), budgets)

    if any(qa[s].get("kl_tokens") is None for s in sids):
        _attach_tokens(cfg, spec, qa)
    low, high, eps = _sig_params(cfg)
    flat_len = n_orders * int(sum(budgets))
    zflat = np.zeros((len(sids), flat_len), dtype=np.float32)
    ztflat = np.zeros((len(sids), flat_len), dtype=np.float32)
    for r, sid in enumerate(sids):
        e = qa[sid]
        posmap = {int(rid): i for i, rid in enumerate(e["rollout_id"])}
        kl, vt = e["kl_tokens"], e["v_tokens"]
        off = 0
        for o in range(n_orders):
            posfull = [posmap[int(rid)] for rid in orders[sid][o]]
            for N in budgets:
                pref = posfull[:N]
                z, zt = visconf_prefix([kl[p] for p in pref], [vt[p] for p in pref], low, high, eps)
                zflat[r, off:off + N] = z
                ztflat[r, off:off + N] = zt
                off += N
    if path is not None:
        try:
            ensure_dir(path.parent)
            np.savez(os.fspath(path), sample_ids=np.array(sids, dtype=object),
                     Z=zflat, Ztok=ztflat, budgets=np.array(budgets, dtype=np.int64),
                     n_orders=np.int64(n_orders))
        except Exception:
            pass
    return _decode_visconf(sids, zflat, ztflat, n_orders, budgets)


def _load_visconf_npz(path):
    z = np.load(os.fspath(path), allow_pickle=True)
    sids = [str(s) for s in z["sample_ids"].tolist()]
    return sids, z["Z"], z["Ztok"], int(z["n_orders"])


def _decode_visconf(sids, zflat, ztflat, n_orders, budgets=None) -> dict[str, dict]:
    # Signature tolerates both call shapes: (..., n_orders, budgets) and (..., budgets) from load.
    if budgets is None:
        raise ValueError("budgets required")
    out: dict[str, dict] = {}
    for r, sid in enumerate(sids):
        zd: dict = {}
        ztd: dict = {}
        off = 0
        for o in range(n_orders):
            for N in budgets:
                zd[(o, N)] = np.asarray(zflat[r, off:off + N], dtype=np.float64)
                ztd[(o, N)] = np.asarray(ztflat[r, off:off + N], dtype=np.float64)
                off += N
        out[sid] = {"Z": zd, "Ztok": ztd}
    return out


# =================================================================================================
# Selection rules (numpy-vectorized; ties -> earliest position in current order)
# =================================================================================================

def _sc_winner(groups: np.ndarray) -> int:
    """Self-consistency via np.bincount over group ids; ties -> earliest position in the prefix."""
    counts = np.bincount(groups)
    best = counts.max()
    is_leader = counts[groups] == best
    return int(groups[int(np.argmax(is_leader))])


def _rule_names(cfg: dict[str, Any], has_vision: bool) -> list[str]:
    rules = list(cfg["comparisons"]["both"])
    if has_vision:
        rules += list(cfg["comparisons"]["mathvista_extra"])
    return rules


def select(rule: str, e: dict, pos: np.ndarray, *, zpref=None, ztokpref=None) -> dict[str, Any]:
    """Apply a rule to the prefix positions `pos`; return group/cost/oracle/correct/distinct."""
    groups = e["group"][pos]
    length = e["length"][pos]
    gc = e["group_correct"]
    if rule == "first_ans":
        chosen = int(groups[0])
        cost = int(length[0])
    else:
        cost = int(length.sum())
        if rule == "self_consistency":
            chosen = _sc_winner(groups)
        elif rule == "shortest":
            chosen = int(groups[int(np.argmin(length))])          # shortest rollout; ties earliest
        elif rule == "highest_logprob":
            chosen = int(groups[int(np.argmax(e["L"][pos]))])
        elif rule == "highest_self_certainty":
            chosen = int(groups[int(np.argmax(e["K"][pos]))])
        elif rule == "self_certainty_rwc":
            chosen = int(signals.rwc_winner(groups.tolist(), e["K"][pos].tolist()))
        elif rule == "vision_only_highest":
            chosen = int(groups[int(np.argmax(e["V"][pos]))])
        elif rule == "vision_only_rwc":
            chosen = int(signals.rwc_winner(groups.tolist(), e["V"][pos].tolist()))
        elif rule == "visconf_highest":
            chosen = int(groups[int(np.argmax(zpref))])
        elif rule == "visconf_rwc":
            chosen = int(signals.rwc_winner(groups.tolist(), list(zpref)))
        elif rule == "visconf_token_highest":
            chosen = int(groups[int(np.argmax(ztokpref))])
        elif rule == "visconf_token_rwc":
            chosen = int(signals.rwc_winner(groups.tolist(), list(ztokpref)))
        else:
            raise ValueError(f"Unknown rule {rule!r}")
    oracle = bool(any(gc.get(int(g), False) for g in groups))
    return {"group": chosen, "correct": bool(gc.get(chosen, False)), "cost": cost, "oracle": oracle,
            "distinct_incl": int(len(set(int(g) for g in groups))),
            "distinct_excl": int(len(set(int(g) for g in groups if g != NO_ANSWER_GROUP)))}


# =================================================================================================
# Bootstrap (shared resample indices)
# =================================================================================================

def _boot_idx(q: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, q, size=(resamples, q))


def _ci(values: np.ndarray, idx: np.ndarray) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _diff(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    d = a - b
    return {"mean": float(d.mean()), "ci95": _ci(d, idx)}


# =================================================================================================
# Run
# =================================================================================================

def run(cfg: dict[str, Any]) -> None:
    """Run all rules x budgets for both datasets; write offline.json + offline_per_question.parquet."""
    budgets = [int(b) for b in cfg["offline"]["budgets"]]
    n_orders = int(cfg["offline"]["n_orders"])
    order_seed = int(cfg["offline"].get("order_seed", 6304))
    resamples = int(cfg["report"]["bootstrap_resamples"])

    results: dict[str, Any] = {}
    pq_cols: dict[str, list] = {k: [] for k in
                                ("dataset", "sample_id", "rule", "N", "order_idx",
                                 "candidate_rollout_ids", "selected_group", "selected_answer",
                                 "correct", "cost", "oracle", "n_distinct_groups_incl_no_answer",
                                 "n_distinct_groups_excl_no_answer")}

    for spec in cfg["data"]["models"]:
        dataset = str(spec["dataset"])
        qa, has_vision = build_question_arrays(cfg, spec)
        split = make_splits(list(qa), seed=int(cfg["splits"]["seed"]),
                            dev_size=int(cfg["splits"]["dev_size"]))
        eval_ids = split["eval"]
        limit_q = int(cfg["offline"].get("limit_questions", 0) or 0)
        if limit_q:
            eval_ids = eval_ids[:limit_q]
        orders = load_or_make_orders(dataset, qa, n_orders=n_orders, seed=order_seed)
        vz = load_or_build_visconf(cfg, spec, qa, orders, budgets) if has_vision else None
        rules = _rule_names(cfg, has_vision)

        # per-question (averaged over orders) matrices: metric -> rule -> {N: np.array over eval}
        acc = {r: {} for r in rules}
        cov = {r: {} for r in rules}
        cost = {r: {} for r in rules}
        dinc = {r: {} for r in rules}
        dexc = {r: {} for r in rules}
        fnc = {r: {} for r in rules}
        fmiss = {r: {} for r in rules}

        # position maps per sid/order
        posfull = {sid: [np.array([{int(x): i for i, x in enumerate(qa[sid]["rollout_id"])}[int(rid)]
                                    for rid in orders[sid][o]], dtype=np.int64)
                         for o in range(n_orders)] for sid in eval_ids}

        for N in budgets:
            per_q = {r: {m: [] for m in ("acc", "cov", "cost", "dinc", "dexc", "fnc", "fmiss")}
                     for r in rules}
            for sid in eval_ids:
                e = qa[sid]
                acc_o = {r: [] for r in rules}
                tallies = {r: {"cov": 0.0, "cost": 0.0, "dinc": 0.0, "dexc": 0.0,
                               "fnc": 0.0, "fmiss": 0.0} for r in rules}
                for o in range(n_orders):
                    pos = posfull[sid][o][:N]
                    zp = vz[sid]["Z"][(o, N)] if has_vision else None
                    ztp = vz[sid]["Ztok"][(o, N)] if has_vision else None
                    cand = e["rollout_id"][pos].tolist()
                    for r in rules:
                        out = select(r, e, pos, zpref=zp, ztokpref=ztp)
                        acc_o[r].append(1.0 if out["correct"] else 0.0)
                        t = tallies[r]
                        t["cov"] += out["oracle"]; t["cost"] += out["cost"]
                        t["dinc"] += out["distinct_incl"]; t["dexc"] += out["distinct_excl"]
                        if not out["correct"]:
                            t["fnc" if not out["oracle"] else "fmiss"] += 1.0
                        pq_cols["dataset"].append(dataset); pq_cols["sample_id"].append(sid)
                        pq_cols["rule"].append(r); pq_cols["N"].append(N)
                        pq_cols["order_idx"].append(o)
                        pq_cols["candidate_rollout_ids"].append(cand)
                        pq_cols["selected_group"].append(out["group"])
                        pq_cols["selected_answer"].append(e["group_rep"].get(out["group"],
                                                                             t3data.NO_ANSWER))
                        pq_cols["correct"].append(out["correct"]); pq_cols["cost"].append(out["cost"])
                        pq_cols["oracle"].append(out["oracle"])
                        pq_cols["n_distinct_groups_incl_no_answer"].append(out["distinct_incl"])
                        pq_cols["n_distinct_groups_excl_no_answer"].append(out["distinct_excl"])
                for r in rules:
                    k = n_orders
                    per_q[r]["acc"].append(float(np.mean(acc_o[r])))
                    per_q[r]["cov"].append(tallies[r]["cov"] / k)
                    per_q[r]["cost"].append(tallies[r]["cost"] / k)
                    per_q[r]["dinc"].append(tallies[r]["dinc"] / k)
                    per_q[r]["dexc"].append(tallies[r]["dexc"] / k)
                    per_q[r]["fnc"].append(tallies[r]["fnc"] / k)
                    per_q[r]["fmiss"].append(tallies[r]["fmiss"] / k)
            for r in rules:
                acc[r][N] = np.array(per_q[r]["acc"])
                cov[r][N] = np.array(per_q[r]["cov"])
                cost[r][N] = np.array(per_q[r]["cost"])
                dinc[r][N] = np.array(per_q[r]["dinc"])
                dexc[r][N] = np.array(per_q[r]["dexc"])
                fnc[r][N] = np.array(per_q[r]["fnc"])
                fmiss[r][N] = np.array(per_q[r]["fmiss"])

        idx = _boot_idx(len(eval_ids), resamples, order_seed)
        ds_res: dict[str, Any] = {"n_eval_questions": len(eval_ids), "budgets": budgets}
        for r in rules:
            ds_res[r] = {}
            for N in budgets:
                ds_res[r][N] = {
                    "accuracy": float(acc[r][N].mean()),
                    "accuracy_ci95": _ci(acc[r][N], idx),
                    "oracle_coverage": float(cov[r][N].mean()),
                    "generated_token_cost": float(cost[r][N].mean()),
                    "distinct_groups_incl_no_answer": float(dinc[r][N].mean()),
                    "distinct_groups_excl_no_answer": float(dexc[r][N].mean()),
                    "fail_no_correct": float(fnc[r][N].mean()),
                    "fail_missed": float(fmiss[r][N].mean()),
                }

        # paired question-level bootstrap on accuracy differences (shared idx), every N
        pairs: list[tuple[str, str]] = [(r, "self_consistency") for r in rules
                                        if r != "self_consistency"]
        pairs += [("self_certainty_rwc", "highest_self_certainty"),
                  ("highest_self_certainty", "highest_logprob")]
        if has_vision:
            pairs += [("visconf_rwc", "visconf_highest"),
                      ("vision_only_rwc", "vision_only_highest"),
                      ("visconf_highest", "vision_only_highest"),
                      ("visconf_rwc", "self_certainty_rwc"),
                      ("visconf_token_highest", "visconf_highest"),
                      ("visconf_token_rwc", "visconf_rwc")]
        diffs: dict[str, Any] = {}
        for a, b in pairs:
            if a not in acc or b not in acc:
                continue
            diffs[f"{a}__vs__{b}"] = {N: _diff(acc[a][N], acc[b][N], idx) for N in budgets}
        ds_res["_accuracy_diffs"] = diffs
        results[dataset] = ds_res

    out_dir = ensure_dir(REPO_ROOT / "results" / "task3")
    with open(os.fspath(out_dir / "offline.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_per_question(out_dir, pq_cols)
    print(f"[task3.offline] wrote {out_dir/'offline.json'} and offline_per_question.parquet "
          f"({len(pq_cols['dataset'])} rows; {', '.join(results)}).")


def _write_per_question(out_dir, cols: dict[str, list]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table(cols), os.fspath(out_dir / "offline_per_question.parquet"))


def main() -> None:
    parser = build_arg_parser("Task 3 offline selection experiments.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
