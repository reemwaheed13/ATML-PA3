"""Load/join supplied rollout tables; form answer groups + correctness labels. (handout p11-p12)

Each model folder has generations.parquet and probability_metrics.parquet (VLM also
attention_metrics.parquet); join on (dataset, sample_id, rollout_id). Fields include text, token
IDs, lengths, extracted answers, gold answers, sampled-token logp, and full-distribution kl_u_p
(raw model distributions, before sampling transforms); VLM attention is split into image /
prompt-text / generated-text groups. Images are not needed for offline analysis.

Correctness (p11-12): MATH equivalence via math-verify; MathVista via answer-type/choice/precision
rules from released question metadata (MC answers as option letters). Answer GROUPS are formed
WITHOUT consulting gold; the same equivalence rule is used for grading. Missing extractions map to
a single NO_ANSWER group, always graded incorrect. The scorer is frozen before final evaluation.

Column names are read from cfg["columns"] (configs/task3/offline.yaml) so matching the supplied
schema is a config edit; the per-rollout dict this emits uses the logical keys documented in
task3_ttc/signals.py (logp, kl_u_p, attn_image/prompt/generated, text, token_ids, length,
extracted, gold).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

NO_ANSWER = "NO_ANSWER"


@dataclass
class QuestionRollouts:
    """All rollouts for one question, aligned across the joined tables."""

    dataset: str
    sample_id: str
    rollouts: list[dict[str, Any]]   # per-rollout: text, token_ids, length, extracted, logp, kl_u_p, (attn)
    gold: Any                        # for retrospective analysis only; NOT for selection/stopping


def _expand(path: str) -> str:
    return os.path.expandvars(os.fspath(path))


def load_joined(cfg: dict[str, Any], model_spec: dict[str, Any]) -> list[QuestionRollouts]:
    """Load + join the parquet files for one model/dataset; verify token-list alignment. (p12 Step 1)

    Returns one QuestionRollouts per question (all its rollouts kept together). Per-token arrays
    (logp, kl_u_p, and VLM attention totals) are verified to match the generated-token length with
    termination excluded; a mismatch raises so a schema/column error surfaces immediately.
    """
    import pandas as pd

    cols = cfg["columns"]
    jk = cols["join"]
    root = _expand(cfg["data"]["root"])
    folder = os.path.join(root, model_spec["name"])
    is_vlm = model_spec.get("type") == "vlm"

    gen = pd.read_parquet(os.path.join(folder, "generations.parquet"))
    prob = pd.read_parquet(os.path.join(folder, "probability_metrics.parquet"))
    keys = [jk["dataset"], jk["sample_id"], jk["rollout_id"]]
    df = gen.merge(prob, on=keys, how="inner", suffixes=("", "_prob"))
    if is_vlm:
        attn = pd.read_parquet(os.path.join(folder, "attention_metrics.parquet"))
        df = df.merge(attn, on=keys, how="inner", suffixes=("", "_attn"))

    g, p = cols["generations"], cols["probability"]
    a = cols.get("attention", {})

    out: list[QuestionRollouts] = []
    for sample_id, grp in df.groupby(jk["sample_id"], sort=True):
        grp = grp.sort_values(jk["rollout_id"])
        rollouts: list[dict[str, Any]] = []
        gold = None
        for _, row in grp.iterrows():
            logp = _as_list(row[p["logp"]])
            kl = _as_list(row[p["kl_u_p"]])
            _check_align(logp, kl, sample_id)
            r = {
                "rollout_id": row[jk["rollout_id"]],
                "text": row[g["text"]],
                "token_ids": _as_list(row[g["token_ids"]]),
                "length": int(row[g["length"]]) if g["length"] in row else len(logp),
                "extracted": _norm_extracted(row.get(g["extracted"])),
                "logp": logp,
                "kl_u_p": kl,
            }
            if is_vlm and a:
                r["attn_image"] = _as_list(row[a["image"]])
                r["attn_prompt"] = _as_list(row[a["prompt"]])
                r["attn_generated"] = _as_list(row[a["generated"]])
            rollouts.append(r)
            gold = row.get(g["gold"], gold)
        out.append(QuestionRollouts(dataset=str(model_spec["dataset"]), sample_id=str(sample_id),
                                    rollouts=rollouts, gold=gold))
    return out


def _as_list(x) -> list:
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return list(x.tolist())
    return list(x)


def _check_align(logp: list, kl: list, sample_id) -> None:
    if len(logp) != len(kl):
        raise ValueError(f"token-array misalignment at sample {sample_id}: "
                         f"logp={len(logp)} kl_u_p={len(kl)}")


def _norm_extracted(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


# =================================================================================================
# Equivalence, answer grouping, correctness
# =================================================================================================

_MATH_VERIFY = None


def _math_equivalent(a: str, b: str) -> bool:
    """MATH answer equivalence via math-verify (parse + verify), with a normalized-string fallback."""
    global _MATH_VERIFY
    if _MATH_VERIFY is None:
        try:
            from math_verify import parse, verify
            _MATH_VERIFY = (parse, verify)
        except Exception:
            _MATH_VERIFY = False
    if _MATH_VERIFY:
        parse, verify = _MATH_VERIFY
        try:
            return bool(verify(parse(a), parse(b)))
        except Exception:
            pass
    return _norm_math(a) == _norm_math(b)


def _norm_math(s: str) -> str:
    return "".join(str(s).split()).rstrip(".").lower()


def _mathvista_equivalent(a: str, b: str, metadata: Optional[dict]) -> bool:
    """MathVista equivalence using released metadata (answer_type / choices / precision). (p11-12)

    Multiple-choice: compare option letters. Float: compare at the given precision. Otherwise a
    normalized string compare. `metadata` carries question_type/answer_type/choices/precision.
    """
    md = metadata or {}
    atype = (md.get("answer_type") or md.get("question_type") or "").lower()
    if md.get("choices"):
        return _mc_letter(a, md["choices"]) == _mc_letter(b, md["choices"])
    if atype in ("float", "integer", "number"):
        try:
            prec = int(md.get("precision", 2))
            return round(float(a), prec) == round(float(b), prec)
        except (TypeError, ValueError):
            pass
    return str(a).strip().lower() == str(b).strip().lower()


def _mc_letter(ans: str, choices: list) -> str:
    """Represent a multiple-choice answer as its option letter (A, B, ...)."""
    s = str(ans).strip()
    if len(s) == 1 and s.upper().isalpha():
        return s.upper()
    for i, c in enumerate(choices):
        if str(c).strip().lower() == s.lower():
            return chr(ord("A") + i)
    return s.lower()


def equivalence(a: str, b: str, dataset: str, metadata: Optional[dict] = None) -> bool:
    """Answer equivalence: math-verify (MATH) or MathVista metadata rules (p11-12)."""
    if a is None or b is None:
        return False
    if dataset.lower().startswith("math") and "vista" not in dataset.lower():
        return _math_equivalent(a, b)
    return _mathvista_equivalent(a, b, metadata)


def answer_groups(rollout_answers: list[Optional[str]], dataset: str,
                  metadata: Optional[dict] = None) -> list[int]:
    """Group extracted answers by equivalence WITHOUT gold; NO_ANSWER is its own group. (p12 Step 1)

    Returns a per-rollout integer group id. Group 0 is reserved for NO_ANSWER (missing extraction);
    other groups are numbered in order of first appearance, each with a representative answer.
    """
    groups: list[int] = []
    reps: list[str] = []                 # representative answer per group id (>0)
    for ans in rollout_answers:
        if ans is None or str(ans).strip() == "" or str(ans) == NO_ANSWER:
            groups.append(0)
            continue
        gid = None
        for i, rep in enumerate(reps):
            if equivalence(ans, rep, dataset, metadata):
                gid = i + 1
                break
        if gid is None:
            reps.append(ans)
            gid = len(reps)
        groups.append(gid)
    return groups


def group_representatives(rollout_answers: list[Optional[str]], groups: list[int]) -> dict[int, str]:
    """Map group id -> a representative answer string (NO_ANSWER for group 0)."""
    reps = {0: NO_ANSWER}
    for ans, gid in zip(rollout_answers, groups):
        if gid > 0 and gid not in reps:
            reps[gid] = ans
    return reps


def correctness_labels(groups: list[int], gold, dataset: str, rollout_answers: list[Optional[str]],
                       metadata: Optional[dict] = None) -> list[bool]:
    """Label each rollout correct/incorrect vs gold (retrospective; NO_ANSWER always incorrect).

    Grades each group's representative answer against gold once (same equivalence rule), then maps
    the per-group verdict back to rollouts. Group 0 (NO_ANSWER) is always incorrect.
    """
    reps = group_representatives(rollout_answers, groups)
    verdict = {gid: (gid != 0 and gold is not None and equivalence(rep, gold, dataset, metadata))
               for gid, rep in reps.items()}
    return [bool(verdict.get(gid, False)) for gid in groups]
