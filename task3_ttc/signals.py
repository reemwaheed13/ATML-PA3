"""Quality signals: log-prob, self-certainty, visual engagement, VisConf, RWC. (handout p12-p13)

Per rollout i with T_i tokens (token measurements read from logp and kl_u_p, raw model dists):
    L_i = (1/T_i) sum_t log p_{i,t}(y_{i,t})            mean sampled-token log-probability
    K_i = (1/T_i) sum_t k_{i,t},  k_{i,t} = KL(U || p_{i,t})   mean self-certainty

MathVista visual engagement (from group attention totals):
    v_{i,t} = A_image / (A_image + A_prompt + A_generated),   V_i = (1/T_i) sum_t v_{i,t}

VisConf: pool token KL across the CURRENT candidate set G, calibrate with its 5th/95th
percentiles (linear interpolation), recomputed for each candidate prefix:
    c_{i,t} = clip_[0,1]( (k_{i,t} - Q05(G)) / max(Q95(G) - Q05(G), 1e-8) )
    C_i = (1/T_i) sum_t c_{i,t},   Z_i = V_i * C_i

RWC (rank-weighted consensus) with descending average rank r_i among N>1 candidates (rank 1 best):
    w_i = (N - r_i) / (N - 1),   W(a) = sum_{i: a_i == a} w_i,   a_hat = argmax_a W(a)

Tokenwise fusion ablation (Step 3): Z_i^token = (1/T_i) sum_t c_{i,t} v_{i,t};
    Z_i^token - Z_i = Cov(c_{i,t}, v_{i,t}).

Per-rollout dict schema (produced by task3_ttc.data.load_joined), all per-token arrays aligned to
the generated tokens with termination excluded (handout p12):
    "logp":           sampled-token log-probability  log p_{i,t}(y_{i,t})     [T_i]
    "kl_u_p":         self-certainty  k_{i,t} = KL(U || p_{i,t})              [T_i]
    "attn_image" / "attn_prompt" / "attn_generated":  group attention totals  [T_i]  (VLM only)
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Default calibration percentiles / epsilon (overridable from configs/task3/offline.yaml:signals).
_P_LOW = 0.05
_P_HIGH = 0.95
_EPS = 1.0e-8


def _arr(x: Any) -> np.ndarray:
    """Coerce a per-token measurement into a flat float64 array."""
    return np.asarray(x, dtype=np.float64).ravel()


def mean_logprob(rollout: dict[str, Any]) -> float:
    """L_i: mean sampled-token log-probability over the rollout's T_i tokens."""
    lp = _arr(rollout["logp"])
    return float(lp.mean()) if lp.size else 0.0


def self_certainty(rollout: dict[str, Any]) -> float:
    """K_i: mean self-certainty (KL(U||p)) over the rollout's T_i tokens."""
    kl = _arr(rollout["kl_u_p"])
    return float(kl.mean()) if kl.size else 0.0


def _token_visual(rollout: dict[str, Any]) -> np.ndarray:
    """Per-token visual-engagement share v_{i,t} = A_image / (A_image + A_prompt + A_generated).

    Normalized as specified (p12) rather than using raw image-attention mass, since the group
    totals exclude some special positions and need not sum to one. Tokens with zero total mass
    contribute 0.
    """
    img = _arr(rollout["attn_image"])
    prm = _arr(rollout["attn_prompt"])
    gen = _arr(rollout["attn_generated"])
    total = img + prm + gen
    v = np.zeros_like(total)
    nz = total > 0
    v[nz] = img[nz] / total[nz]
    return v


def visual_engagement(rollout: dict[str, Any]) -> float:
    """V_i: mean per-token visual engagement over T_i tokens (MathVista)."""
    v = _token_visual(rollout)
    return float(v.mean()) if v.size else 0.0


def _calibrate_tokens(candidate_set: list[dict[str, Any]], low: float, high: float,
                      eps: float) -> list[np.ndarray]:
    """Per-rollout calibrated token confidence c_{i,t} using percentiles pooled over G.

    Calibration uses ONLY the supplied candidate set (the current prefix): pool every rollout's
    k_{i,t}, take the low/high percentiles with linear interpolation, and clip each token to [0,1].
    """
    if not candidate_set:
        return []
    pooled = np.concatenate([_arr(r["kl_u_p"]) for r in candidate_set])
    if pooled.size:
        q_lo = float(np.percentile(pooled, low * 100.0, method="linear"))
        q_hi = float(np.percentile(pooled, high * 100.0, method="linear"))
    else:
        q_lo = q_hi = 0.0
    span = max(q_hi - q_lo, eps)
    return [np.clip((_arr(r["kl_u_p"]) - q_lo) / span, 0.0, 1.0) for r in candidate_set]


def visconf(candidate_set: list[dict[str, Any]], *, percentile_low: float = _P_LOW,
            percentile_high: float = _P_HIGH, eps: float = _EPS) -> list[float]:
    """Compute Z_i = V_i * C_i for each candidate in G (calibration uses only this candidate set)."""
    cal = _calibrate_tokens(candidate_set, percentile_low, percentile_high, eps)
    out: list[float] = []
    for r, c in zip(candidate_set, cal):
        c_i = float(c.mean()) if c.size else 0.0
        out.append(visual_engagement(r) * c_i)
    return out


def visconf_tokenwise(candidate_set: list[dict[str, Any]], *, percentile_low: float = _P_LOW,
                      percentile_high: float = _P_HIGH, eps: float = _EPS) -> list[float]:
    """Z_i^token = (1/T_i) sum_t c_{i,t} v_{i,t} (per-token fusion ablation, Step 3)."""
    cal = _calibrate_tokens(candidate_set, percentile_low, percentile_high, eps)
    out: list[float] = []
    for r, c in zip(candidate_set, cal):
        v = _token_visual(r)
        out.append(float((c * v).mean()) if c.size else 0.0)
    return out


def _average_ranks_desc(scores: list[float]) -> np.ndarray:
    """Average ranks (1 = highest score) with ties sharing the mean of their positions."""
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    order = np.argsort(-s, kind="mergesort")       # stable, descending
    positional = np.empty(n, dtype=np.float64)
    positional[order] = np.arange(1, n + 1)
    ranks = positional.copy()
    for val in np.unique(s):
        idx = np.where(s == val)[0]
        if idx.size > 1:
            ranks[idx] = positional[idx].mean()
    return ranks


def _rwc_weights(answers: list[Any], scores: list[float]) -> dict[Any, float]:
    """Accumulate RWC vote weight w_i = (N - r_i)/(N - 1) per answer group."""
    n = len(answers)
    ranks = _average_ranks_desc(scores)
    w = (n - ranks) / (n - 1)
    weights: dict[Any, float] = {}
    for a, wi in zip(answers, w):
        weights[a] = weights.get(a, 0.0) + float(wi)
    return weights


def rwc_winner(answers: list[Any], scores: list[float]) -> Any:
    """RWC-selected answer using descending average ranks; ties broken by earliest supporting rollout.

    `answers` must be in the current candidate order so the earliest-occurrence tie-break matches
    "earliest supporting rollout in the current order" (handout p12). At N == 1 the single
    candidate is returned (handout: every rule returns the only candidate at N = 1).
    """
    n = len(answers)
    if n == 0:
        return None
    if n == 1:
        return answers[0]
    weights = _rwc_weights(answers, scores)
    first_pos: dict[Any, int] = {}
    for i, a in enumerate(answers):
        first_pos.setdefault(a, i)
    # Highest total weight wins; ties -> smallest first_pos (earliest supporting rollout).
    return max(weights.items(), key=lambda kv: (kv[1], -first_pos[kv[0]]))[0]


def rwc_margin(answers: list[Any], scores: list[float]) -> float:
    """(largest - second-largest vote) / total vote weight; runner-up vote is 0 with one group (p14)."""
    n = len(answers)
    if n <= 1:
        return 0.0
    weights = _rwc_weights(answers, scores)
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    votes = sorted(weights.values(), reverse=True)
    v1 = votes[0]
    v2 = votes[1] if len(votes) > 1 else 0.0
    return float((v1 - v2) / total)
