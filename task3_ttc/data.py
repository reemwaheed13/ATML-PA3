"""Load/join supplied rollout tables; form answer groups + correctness labels. (handout p11-p12)

Each model folder (configs/task3/offline.yaml: models[].folder) holds generations.parquet and
probability_metrics.parquet; the VLM folder also has attention_metrics.parquet. The three tables are
aligned on (dataset, sample_id, rollout_id). sample_id is a STRING ("test:4869" for MATH, "980" for
MathVista) and is always sorted as a string.

Speed/memory (p12 Step 1): we read only the columns we need with pyarrow and do NOT keep
`generated_text`/`generated_token_ids` in the per-rollout dicts (the qualitative step pulls text
on demand via `load_text`). Per-token list columns (logp, kl_u_p, VLM attention totals) are read as
flat float32 buffers with offsets; each rollout gets a numpy VIEW into the shared buffer. Attention
is read with an explicit column list that EXCLUDES its own per-token `n_generated_text_tokens` list
(name clashes with the generations scalar).

Per-rollout dict keys (the schema task3_ttc/signals.py documents): logp, kl_u_p,
attn_image/attn_prompt/attn_generated (VLM), length, extracted, rollout_id. Gold and MathVista
metadata live on the QuestionRollouts (not per rollout).

Correctness (p11-12): MATH via math-verify (answers wrapped so they parse; parse results cached;
grouped by normalized string first, then distinct groups compared; multiprocessing across questions);
MathVista via answer-type/choice/precision rules over the released question metadata (MC answers as
option letters). Answer GROUPS are formed WITHOUT consulting gold; the same rule grades. Missing
extraction -> single NO_ANSWER group (group 0), always incorrect, kept in voting and cost accounting.
The scorer is frozen before final evaluation; SCORER_VERSION keys the on-disk cache.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import numpy as np

from common.paths import REPO_ROOT, ensure_dir, task3_dir

NO_ANSWER = "NO_ANSWER"

# Bump when the scoring logic changes so stale group/correctness caches are ignored.
SCORER_VERSION = "t3-scorer-3"


@dataclass
class QuestionRollouts:
    """All rollouts for one question, aligned across the joined tables."""

    dataset: str
    sample_id: str
    rollouts: list[dict[str, Any]]   # per-rollout: logp, kl_u_p, (attn_*), length, extracted, rollout_id
    gold: Any                        # for retrospective analysis only; NOT for selection/stopping
    metadata: Any = None             # MathVista question_type/answer_type/choices/precision/unit/answer


def _expand(path: str) -> str:
    return os.path.expandvars(os.fspath(path))


# =================================================================================================
# Parquet reading / alignment (pyarrow; no iterrows, no pandas merge on list columns)
# =================================================================================================

def _single(chunked):
    """Concatenate a pyarrow ChunkedArray's chunks into one Array (robust across versions)."""
    import pyarrow as pa
    return pa.concat_arrays(chunked.chunks)


def _col_np(chunked, dtype=None) -> np.ndarray:
    """A parquet scalar column as a flat numpy array (via a single concatenated Array)."""
    arr = _single(chunked).to_numpy(zero_copy_only=False)
    return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)


def _list_flat(chunked, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a pyarrow list column to (flat values [dtype], offsets [int64, len n+1]).

    Offsets are absolute indices into the (shared) flat values buffer, so row i spans
    values[offsets[i]:offsets[i+1]] with no per-row copy.
    """
    arr = _single(chunked)
    offsets = np.asarray(arr.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    values = np.asarray(arr.values.to_numpy(zero_copy_only=False)).astype(dtype, copy=False)
    return values, offsets


def _read_model_tables(cfg: dict[str, Any], model_spec: dict[str, Any],
                       *, strict: bool = True) -> dict[str, Any]:
    """Read + align the parquet tables for one model; verify per-token length alignment. (p12 Step 1)

    Returns {"questions": list[QuestionRollouts], "diag": {...}}. With strict=True a membership or
    length mismatch raises (fail fast); strict=False fills the diagnostics for `--check` instead.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cols = cfg["columns"]
    jk = cols["join"]
    g, p = cols["generations"], cols["probability"]
    a = cols.get("attention", {})
    is_vlm = model_spec.get("type") == "vlm"
    dataset = str(model_spec["dataset"])

    root = _expand(cfg["data"]["root"])
    folder = os.path.join(root, model_spec["folder"])
    gen_path = os.path.join(folder, "generations.parquet")
    prob_path = os.path.join(folder, "probability_metrics.parquet")
    attn_path = os.path.join(folder, "attention_metrics.parquet")

    # ---- generations: scalar columns only (no text, no token-id values retained) ----
    gt = pq.read_table(gen_path, columns=[jk["sample_id"], jk["rollout_id"],
                                          g["length"], g["extracted"], g["gold"]])
    g_sid = [str(s) for s in gt.column(jk["sample_id"]).to_pylist()]
    g_rid = _col_np(gt.column(jk["rollout_id"]))
    g_len = _col_np(gt.column(g["length"]), dtype=np.int64)
    g_extracted = gt.column(g["extracted"]).to_pylist()
    g_gold = gt.column(g["gold"]).to_pylist()
    n_gen = len(g_sid)

    # Token-id lengths for the alignment check, computed with pyarrow (no python list materialization).
    tid_tbl = pq.read_table(gen_path, columns=[g["token_ids"]])
    g_tokid_len = _col_np(pc.list_value_length(tid_tbl.column(g["token_ids"])), dtype=np.int64)
    del tid_tbl

    # ---- probability metrics: flat logp / kl_u_p buffers + offsets ----
    pt = pq.read_table(prob_path, columns=[jk["sample_id"], jk["rollout_id"],
                                           p["logp"], p["kl_u_p"]])
    p_sid = [str(s) for s in pt.column(jk["sample_id"]).to_pylist()]
    p_rid = _col_np(pt.column(jk["rollout_id"]))
    logp_vals, logp_off = _list_flat(pt.column(p["logp"]))
    kl_vals, kl_off = _list_flat(pt.column(p["kl_u_p"]))
    len_logp = np.diff(logp_off)
    len_kl = np.diff(kl_off)
    p_pos = {k: i for i, k in enumerate(zip(p_sid, (int(r) for r in p_rid)))}
    n_prob = len(p_sid)

    # ---- attention metrics (VLM only): explicit column list EXCLUDES n_generated_text_tokens ----
    a_pos: dict = {}
    img_vals = prm_vals = gen_vals = None
    img_off = prm_off = gen_off = None
    len_img = len_prm = len_gen = None
    n_attn = 0
    if is_vlm:
        at = pq.read_table(attn_path, columns=[jk["sample_id"], jk["rollout_id"],
                                               a["image"], a["prompt"], a["generated"]])
        a_sid = [str(s) for s in at.column(jk["sample_id"]).to_pylist()]
        a_rid = _col_np(at.column(jk["rollout_id"]))
        img_vals, img_off = _list_flat(at.column(a["image"]))
        prm_vals, prm_off = _list_flat(at.column(a["prompt"]))
        gen_vals, gen_off = _list_flat(at.column(a["generated"]))
        len_img, len_prm, len_gen = np.diff(img_off), np.diff(prm_off), np.diff(gen_off)
        a_pos = {k: i for i, k in enumerate(zip(a_sid, (int(r) for r in a_rid)))}
        n_attn = len(a_sid)

    # ---- MathVista metadata (HF), joined on pid == sample_id ----
    metadata_map = load_mathvista_metadata(cfg) if is_vlm else {}

    # ---- align, length-check, build per-question rollouts ----
    diag = {
        "folder": model_spec["folder"], "dataset": dataset,
        "n_gen": n_gen, "n_prob": n_prob, "n_attn": n_attn,
        "n_null_extracted": 0, "n_matched": 0, "n_unmatched": 0,
        "mismatch_logp_kl_len": 0, "mismatch_tokid_len": 0, "mismatch_attn_len": 0,
        "n_questions": 0, "n_rollouts": 0,
    }
    diag["n_null_extracted"] = sum(1 for x in g_extracted if _norm_extracted(x) is None)

    # Canonical order: sample_id as STRING, then rollout_id ascending. Keeps a question together.
    order = sorted(range(n_gen), key=lambda i: (g_sid[i], int(g_rid[i])))

    out: list[QuestionRollouts] = []
    cur_sid: Optional[str] = None
    cur_rollouts: list[dict[str, Any]] = []
    cur_gold: Any = None
    for gi in order:
        sid = g_sid[gi]
        key = (sid, int(g_rid[gi]))
        pi = p_pos.get(key)
        ai = a_pos.get(key) if is_vlm else 0
        if pi is None or (is_vlm and ai is None):
            diag["n_unmatched"] += 1
            continue
        diag["n_matched"] += 1

        n = int(g_len[gi])
        if not (len_logp[pi] == len_kl[pi] == n):
            diag["mismatch_logp_kl_len"] += 1
        if g_tokid_len[gi] != n:
            diag["mismatch_tokid_len"] += 1
        if is_vlm and not (len_img[ai] == len_prm[ai] == len_gen[ai] == n):
            diag["mismatch_attn_len"] += 1

        if sid != cur_sid:
            if cur_sid is not None:
                out.append(QuestionRollouts(dataset=dataset, sample_id=cur_sid, rollouts=cur_rollouts,
                                            gold=cur_gold, metadata=metadata_map.get(cur_sid)))
            cur_sid, cur_rollouts, cur_gold = sid, [], g_gold[gi]

        r: dict[str, Any] = {
            "rollout_id": int(g_rid[gi]),
            "length": n,
            "extracted": _norm_extracted(g_extracted[gi]),
            "logp": logp_vals[logp_off[pi]:logp_off[pi + 1]],
            "kl_u_p": kl_vals[kl_off[pi]:kl_off[pi + 1]],
        }
        if is_vlm:
            r["attn_image"] = img_vals[img_off[ai]:img_off[ai + 1]]
            r["attn_prompt"] = prm_vals[prm_off[ai]:prm_off[ai + 1]]
            r["attn_generated"] = gen_vals[gen_off[ai]:gen_off[ai + 1]]
        cur_rollouts.append(r)

    if cur_sid is not None:
        out.append(QuestionRollouts(dataset=dataset, sample_id=cur_sid, rollouts=cur_rollouts,
                                    gold=cur_gold, metadata=metadata_map.get(cur_sid)))

    diag["n_questions"] = len(out)
    diag["n_rollouts"] = sum(len(q.rollouts) for q in out)

    problems = (diag["n_unmatched"] + diag["mismatch_logp_kl_len"]
                + diag["mismatch_tokid_len"] + diag["mismatch_attn_len"])
    if strict and problems:
        raise ValueError(
            f"Task 3 alignment failed for {model_spec['folder']}: "
            f"unmatched={diag['n_unmatched']} logp/kl/len={diag['mismatch_logp_kl_len']} "
            f"token_ids/len={diag['mismatch_tokid_len']} attn/len={diag['mismatch_attn_len']} "
            f"(gen={n_gen} prob={n_prob} attn={n_attn}).")
    return {"questions": out, "diag": diag}


def load_joined(cfg: dict[str, Any], model_spec: dict[str, Any]) -> list[QuestionRollouts]:
    """Load + join the parquet files for one model/dataset; verify alignment. (p12 Step 1)

    Returns one QuestionRollouts per question (all rollouts kept together). Raises on any membership
    or per-token length mismatch so a schema error surfaces immediately.
    """
    return _read_model_tables(cfg, model_spec, strict=True)["questions"]


def load_text(cfg: dict[str, Any], model_spec: dict[str, Any], sample_id, rollout_id) -> dict[str, Any]:
    """Lazily load generated_text + generated_token_ids for ONE rollout (qualitative step only).

    Uses parquet predicate pushdown so the heavy text/token-id columns are not read for the whole
    table. `cfg` is needed for the root path and the column names (both isolated in the config).
    """
    import pyarrow.parquet as pq

    cols = cfg["columns"]
    jk, g = cols["join"], cols["generations"]
    folder = os.path.join(_expand(cfg["data"]["root"]), model_spec["folder"])
    tbl = pq.read_table(
        os.path.join(folder, "generations.parquet"),
        columns=[jk["sample_id"], jk["rollout_id"], g["text"], g["token_ids"]],
        filters=[(jk["sample_id"], "=", str(sample_id)), (jk["rollout_id"], "=", int(rollout_id))],
    )
    if tbl.num_rows == 0:
        raise KeyError(f"no rollout {sample_id!r}/{rollout_id} in {folder}")
    row = tbl.slice(0, 1).to_pylist()[0]
    return {"text": row.get(g["text"]), "token_ids": list(row.get(g["token_ids"]) or [])}


# =================================================================================================
# MathVista question metadata (Hugging Face AI4Math/MathVista, testmini) — cached as small JSON
# =================================================================================================

def load_mathvista_metadata(cfg: dict[str, Any]) -> dict[str, dict]:
    """Return {pid(str) -> metadata dict}; build + cache from HF if the JSON is missing. (no images)"""
    ms = cfg.get("metadata_source") or {}
    cache_rel = ms.get("cache", "splits/task3/mathvista_metadata.json")
    cache_path = REPO_ROOT / cache_rel
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): v for k, v in raw.items()}
    md = _download_mathvista_metadata(ms)
    ensure_dir(cache_path.parent)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(md, f, indent=2, ensure_ascii=False, sort_keys=True)
    return md


def _download_mathvista_metadata(ms: dict[str, Any]) -> dict[str, dict]:
    """Fetch metadata columns from the testmini parquet WITHOUT the image column (pyarrow read)."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    repo = ms.get("dataset", "AI4Math/MathVista")
    split = ms.get("split", "testmini")
    join_key = ms.get("join_key", "pid")
    want = list(ms.get("columns", [join_key, "question_type", "answer_type", "choices",
                                   "precision", "unit", "answer"]))

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    cand = sorted(f for f in files if f.endswith(".parquet") and split in os.path.basename(f))
    if not cand:
        raise RuntimeError(f"no {split} parquet in {repo} (files: {files})")

    md: dict[str, dict] = {}
    for rel in cand:
        local = hf_hub_download(repo, rel, repo_type="dataset")
        have = pq.ParquetFile(local).schema_arrow.names
        read_cols = [c for c in want if c in have]          # read ONLY metadata cols (skip images)
        for rec in pq.read_table(local, columns=read_cols).to_pylist():
            pid = str(rec.get(join_key))
            md[pid] = {k: rec.get(k) for k in read_cols if k != join_key}
    return md


# =================================================================================================
# Answer normalization
# =================================================================================================

def _norm_extracted(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


def _missing(ans) -> bool:
    return ans is None or str(ans).strip() == "" or str(ans) == NO_ANSWER


def _norm_math(s: str) -> str:
    return "".join(str(s).split()).rstrip(".").lower()


# ---- MathVista normalization ----

_TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d)")
_MC_LETTER_RE = re.compile(r"^\(?([A-Za-z])\)?(?:\s*[.\:\)].*)?$")


def _mv_strip(s) -> str:
    """Strip LaTeX markup that is irrelevant to the value: \\text{...}, spacing, $, %, thousands sep."""
    s = str(s).strip()
    s = _TEXT_RE.sub(r"\1", s)
    for tok in ("\\,", "\\;", "\\!"):
        s = s.replace(tok, "")
    s = s.replace("\\ ", " ").replace("$", "").replace("\\%", "").replace("%", "")
    s = _THOUSANDS_RE.sub("", s)
    return s.strip()


def _mv_strip_unit(s: str, unit) -> str:
    if unit:
        u = _mv_strip(unit).strip()
        if u and s.lower().endswith(u.lower()):
            s = s[: len(s) - len(u)].strip()
    return s


# ---- MathVista numeric parsing (answer_type integer/float only) ----

_NUM = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_MATHRM_RE = re.compile(r"\\mathrm\s*\{([^{}]*)\}")

# Whole-string accepted numeric forms (spaces removed before matching).
_PLAIN_RE = re.compile(rf"^{_NUM}$")
_FRAC_RE = re.compile(rf"^\\d?frac\{{({_NUM})\}}\{{({_NUM})\}}$")
_RATIO_RE = re.compile(rf"^({_NUM})/({_NUM})$")
_TIMES10_RE = re.compile(rf"^({_NUM})\\times10\^\{{?([+-]?\d+)\}}?$")
_POW10_RE = re.compile(rf"^10\^\{{?([+-]?\d+)\}}?$")

# Leading numeric expression (spaces allowed), used when dropping a trailing unit. Ordered most
# specific first so "10^3 m" takes the power, not the bare "10".
_LEAD_PATTERNS = [
    re.compile(rf"^\s*{_NUM}\s*\\times\s*10\s*\^\s*\{{?\s*[+-]?\d+\s*\}}?"),   # a x 10^k
    re.compile(rf"^\s*\\d?frac\s*\{{\s*{_NUM}\s*\}}\s*\{{\s*{_NUM}\s*\}}"),     # \frac{a}{b}
    re.compile(rf"^\s*10\s*\^\s*\{{?\s*[+-]?\d+\s*\}}?"),                       # 10^k
    re.compile(rf"^\s*{_NUM}\s*/\s*{_NUM}"),                                    # a/b
    re.compile(rf"^\s*{_NUM}"),                                                 # plain number
]

# A remainder that is only a unit: letters, spaces, /, *, ., parens/braces, backslash-macros, and
# exponents like ^2 or ^{2}. Deliberately EXCLUDES bare digits (so "55"), signs and dashes (so a
# second number or a range), which keeps ranges / "a and b" / functions unparseable.
_UNIT_ONLY_RE = re.compile(
    r"^(?:[A-Za-z]+|\s+|/|\*|·|\.|\^\s*\{?\s*\d+\s*\}?|[{}()]|\\[A-Za-z]+)*$")


def _num_expr_value(expr: str) -> Optional[float]:
    """Value of a whole numeric expression (plain / frac / ratio / a x 10^k / 10^k), else None."""
    e = expr.replace(" ", "")
    if _PLAIN_RE.match(e):
        try:
            return float(e)
        except ValueError:
            return None
    m = _FRAC_RE.match(e) or _RATIO_RE.match(e)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
        except ValueError:
            return None
        return a / b if b != 0 else None
    m = _TIMES10_RE.match(e)
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except ValueError:
            return None
    m = _POW10_RE.match(e)
    if m:
        return 10.0 ** int(m.group(1))
    return None


def _mv_numeric_value(s, unit=None) -> Optional[float]:
    """Parse a MathVista integer/float answer to a float, else None (ranges/letters/infty -> None)."""
    t = _mv_strip_unit(_mv_strip(s), unit)
    t = _MATHRM_RE.sub(r"\1", t)                       # unwrap \mathrm{...}
    t = t.replace("^\\circ", "").replace("\\circ", "").replace("°", "")
    t = t.replace("~", " ").strip()
    if not t:
        return None
    v = _num_expr_value(t)                             # whole string is a number expression
    if v is not None:
        return v
    for pat in _LEAD_PATTERNS:                         # number expression followed only by a unit
        m = pat.match(t)
        if m and _UNIT_ONLY_RE.match(t[m.end():]):
            v = _num_expr_value(m.group(0))
            if v is not None:
                return v
    return None


def _mv_mc_letter(ans, choices: list) -> str:
    """Map a multiple-choice answer to its option letter, else return non-matching normalized text."""
    s = str(ans).strip()
    if not s:
        return s
    norm = _mv_strip(s).lower()
    choices = choices or []
    for i, c in enumerate(choices):                         # full choice-text match
        if _mv_strip(str(c)).lower() == norm:
            return chr(ord("A") + i)
    m = _MC_LETTER_RE.match(s)                              # "B", "(B)", "B.", "B: ..."
    if m:
        letter = m.group(1).upper()
        idx = ord(letter) - ord("A")
        if (not choices and 0 <= idx < 26) or (0 <= idx < len(choices)):
            return letter
    return norm


def _mathvista_canonical(ans, metadata: Optional[dict]) -> str:
    """Canonical comparable token for a MathVista answer (grouping key == grading key)."""
    md = metadata or {}
    choices = md.get("choices") or []
    qtype = str(md.get("question_type") or "").lower()
    if choices or qtype == "multi_choice":
        return _mv_mc_letter(ans, choices)
    atype = str(md.get("answer_type") or "").lower()
    unit = md.get("unit")
    s = _mv_strip_unit(_mv_strip(ans), unit)
    if atype in ("integer", "float"):
        val = _mv_numeric_value(ans, unit)
        if val is None:
            return s.lower()                           # unparseable -> normalized string
        if atype == "integer":
            return str(int(val))
        prec = md.get("precision")
        prec = int(prec) if prec is not None else 2
        return f"{round(val, max(prec, 0)):.{max(prec, 0)}f}"
    return s.lower()


# =================================================================================================
# MATH equivalence (math-verify, cached parse, string fallback)
# =================================================================================================

_MATH_VERIFY: Any = None
_MV_PARSE_KW: dict = {}                 # timeout kwarg for math_verify.parse (name resolved at load)
_MV_VERIFY_KW: dict = {}               # timeout kwarg for math_verify.verify
_MATH_TIMEOUT = 1                      # seconds; a timeout is treated as NOT equivalent
_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")

# Candidate parameter names math-verify has used across versions for the parse/verify timeout.
_TIMEOUT_PARAM_NAMES = ("timeout_seconds", "parsing_timeout", "timeout")


class _TimeoutCounter(logging.Handler):
    """Count math-verify log records that mention a timeout ("Timeout during comparison")."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = ""
        if "timeout" in msg.lower():
            self.count += 1


_TIMEOUT_HANDLER = _TimeoutCounter()


def _reset_timeout_count() -> None:
    _TIMEOUT_HANDLER.count = 0


def _get_timeout_count() -> int:
    return _TIMEOUT_HANDLER.count


def _install_timeout_handler() -> None:
    lg = logging.getLogger("math_verify")
    if _TIMEOUT_HANDLER not in lg.handlers:
        lg.addHandler(_TIMEOUT_HANDLER)
    if lg.level == logging.NOTSET or lg.level > logging.WARNING:
        lg.setLevel(logging.WARNING)       # timeouts are logged as warnings; let them through


def _timeout_kwargs(func, seconds) -> dict:
    """Return {param: seconds} using whichever timeout parameter `func` actually declares."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    for name in _TIMEOUT_PARAM_NAMES:
        if name in params:
            return {name: seconds}
    return {}


def _get_math_verify():
    """Return (parse, verify) from math-verify, or None if unavailable (string fallback then)."""
    global _MATH_VERIFY, _MV_PARSE_KW, _MV_VERIFY_KW
    if _MATH_VERIFY is None:
        try:
            from math_verify import parse, verify
            _MATH_VERIFY = (parse, verify)
            _MV_PARSE_KW = _timeout_kwargs(parse, _MATH_TIMEOUT)
            _MV_VERIFY_KW = _timeout_kwargs(verify, _MATH_TIMEOUT)
            _install_timeout_handler()
        except Exception:
            _MATH_VERIFY = False
    return _MATH_VERIFY or None


def _sanitize_math(s: str) -> str:
    """Normalize a MATH answer for comparison/parse ONLY (the stored raw answer is never changed).

    Removes currency ("\\$"/bare "$"), spacing macros ("\\!", "\\,"), unwraps \\text{...}, drops
    thousands separators inside digit groups ("15,\\!000" -> "15000"), and strips a trailing period
    and surrounding whitespace.
    """
    s = _TEXT_RE.sub(r"\1", str(s))                        # unwrap \text{...}
    s = s.replace("\\!", "").replace("\\,", "")
    s = s.replace("\\$", "").replace("$", "")
    s = _THOUSANDS_RE.sub("", s)                           # digit,digit thousands separators
    return s.strip().rstrip(".").strip()


@lru_cache(maxsize=None)
def _parse_math(s: str):
    """Parse the SANITIZED answer, always wrapped as $...$ so bare/currency answers parse."""
    mv = _get_math_verify()
    if not mv:
        return None
    san = _sanitize_math(s)
    return mv[0](f"${san}$" if san else san, **_MV_PARSE_KW)


def _math_equivalent(a: str, b: str) -> bool:
    """MATH equivalence: identical-string / plain-number fast paths, then math-verify (timeout)."""
    sa, sb = _sanitize_math(a), _sanitize_math(b)
    if _norm_math(sa) == _norm_math(sb):                   # identical after sanitizing -> equivalent
        return True
    if _NUM_RE.match(sa) and _NUM_RE.match(sb):            # both plain numbers -> float compare
        try:
            return math.isclose(float(sa), float(sb), rel_tol=1e-9, abs_tol=1e-12)
        except ValueError:
            pass
    mv = _get_math_verify()
    if mv:
        try:
            return bool(mv[1](_parse_math(a), _parse_math(b), **_MV_VERIFY_KW))
        except Exception:
            pass
    return _norm_math(a) == _norm_math(b)


def math_gold_self_verify(golds) -> tuple[float, list[str]]:
    """Fraction of DISTINCT gold answers that verify against themselves + the list of failures."""
    distinct = sorted({str(g) for g in golds if g is not None})
    if not distinct:
        return 1.0, []
    mv = _get_math_verify()
    if not mv:
        return 1.0, []                                     # string fallback: trivially self-equal
    fails: list[str] = []
    ok = 0
    for g in distinct:
        try:
            if mv[1](_parse_math(g), _parse_math(g), **_MV_VERIFY_KW):
                ok += 1
            else:
                fails.append(g)
        except Exception:
            fails.append(g)
    return ok / len(distinct), fails


# =================================================================================================
# Equivalence dispatch / grouping / correctness
# =================================================================================================

def _is_math(dataset: str) -> bool:
    d = str(dataset).lower()
    return d == "math" or (d.startswith("math") and "vista" not in d)


def equivalence(a: str, b: str, dataset: str, metadata: Optional[dict] = None) -> bool:
    """Answer equivalence: math-verify (MATH) or MathVista metadata rules (p11-12)."""
    if a is None or b is None:
        return False
    if _is_math(dataset):
        return _math_equivalent(a, b)
    return _mathvista_canonical(a, metadata) == _mathvista_canonical(b, metadata)


def answer_groups(rollout_answers: list[Optional[str]], dataset: str,
                  metadata: Optional[dict] = None) -> list[int]:
    """Group extracted answers by equivalence WITHOUT gold; NO_ANSWER is group 0. (p12 Step 1)

    Group ids (>0) are numbered by first appearance. MATH: bucket by normalized string first (cached),
    then compare only DISTINCT buckets pairwise via math-verify. MathVista: group by the canonical
    key (deterministic, no pairwise compare).
    """
    groups: list[int] = []
    if _is_math(dataset):
        reps: list[str] = []                 # one representative raw answer per group id (>0)
        norm_to_gid: dict[str, int] = {}     # normalized string -> group id (fast path)
        for ans in rollout_answers:
            if _missing(ans):
                groups.append(0)
                continue
            nk = _norm_math(ans)
            gid = norm_to_gid.get(nk)
            if gid is None:
                for i, rep in enumerate(reps):
                    if _math_equivalent(ans, rep):
                        gid = i + 1
                        break
                if gid is None:
                    reps.append(ans)
                    gid = len(reps)
                norm_to_gid[nk] = gid
            groups.append(gid)
    else:
        key_to_gid: dict[str, int] = {}
        for ans in rollout_answers:
            if _missing(ans):
                groups.append(0)
                continue
            key = _mathvista_canonical(ans, metadata)
            gid = key_to_gid.get(key)
            if gid is None:
                gid = len(key_to_gid) + 1
                key_to_gid[key] = gid
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

    Grades each group's representative against gold once (same equivalence rule), then maps the
    per-group verdict back to rollouts. Group 0 (NO_ANSWER) is always incorrect.
    """
    reps = group_representatives(rollout_answers, groups)
    verdict = {gid: (gid != 0 and gold is not None and equivalence(rep, gold, dataset, metadata))
               for gid, rep in reps.items()}
    return [bool(verdict.get(gid, False)) for gid in groups]


# =================================================================================================
# Cached, multiprocessing-backed scorer (groups + correctness per rollout)
# =================================================================================================

def _q_payload(q: QuestionRollouts) -> tuple:
    return (q.dataset, q.sample_id, [r.get("extracted") for r in q.rollouts], q.gold, q.metadata,
            [r.get("rollout_id") for r in q.rollouts])


def _score_question(payload: tuple):
    """Top-level (picklable) worker: groups + correctness + math-verify timeout count for one question."""
    dataset, sid, answers, gold, metadata, rollout_ids = payload
    _reset_timeout_count()
    groups = answer_groups(answers, dataset, metadata)
    correct = correctness_labels(groups, gold, dataset, answers, metadata)
    return sid, list(zip(rollout_ids, groups, correct)), _get_timeout_count()


def _scorer_cache_path(dataset: str):
    try:
        return task3_dir("cache") / f"{dataset}_groups_{SCORER_VERSION}.parquet"
    except Exception:
        return None                                         # DRIVE_ROOT unset -> compute in-memory


def _groups_from_table(tbl) -> dict[str, dict]:
    sids = [str(s) for s in tbl.column("sample_id").to_pylist()]
    rids = tbl.column("rollout_id").to_pylist()
    gs = tbl.column("group").to_pylist()
    cs = tbl.column("correct").to_pylist()
    mapping: dict[str, dict] = {}
    for sid, rid, grp, ok in zip(sids, rids, gs, cs):
        m = mapping.setdefault(sid, {"rollout_ids": [], "groups": [], "correct": []})
        m["rollout_ids"].append(int(rid))
        m["groups"].append(int(grp))
        m["correct"].append(bool(ok))
    return mapping


def compute_groups(cfg: dict[str, Any], model_spec: dict[str, Any], questions: list[QuestionRollouts],
                   *, use_cache: bool = True, processes: Optional[int] = None,
                   progress_every: int = 0, stats: Optional[dict] = None) -> dict[str, dict]:
    """Groups + correctness per (dataset, sample_id, rollout_id), cached to DRIVE_ROOT/task3/cache.

    Returns {sample_id -> {"rollout_ids", "groups", "correct"}}. The cache is a parquet keyed on
    SCORER_VERSION (reruns skip recomputation). MATH grouping is parallelized across questions. If
    `stats` is given it is filled with {"timeouts_total", "questions_with_timeout"}; `progress_every`
    > 0 prints progress that often during (re)computation.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset = str(model_spec["dataset"])
    path = _scorer_cache_path(dataset)
    if use_cache and path is not None and path.exists():
        tbl = pq.read_table(path)
        sv = (tbl.column("scorer_version")[0].as_py()
              if "scorer_version" in tbl.column_names and tbl.num_rows else None)
        if sv == SCORER_VERSION:
            return _groups_from_table(tbl)

    payloads = [_q_payload(q) for q in questions]
    total = len(payloads)
    results: list[tuple] = []
    timeouts_total = 0
    questions_with_timeout = 0

    def _consume(iterable) -> None:
        nonlocal timeouts_total, questions_with_timeout
        for i, (sid, per, n_to) in enumerate(iterable, 1):
            results.append((sid, per))
            timeouts_total += n_to
            if n_to:
                questions_with_timeout += 1
            if progress_every and i % progress_every == 0:
                print(f"  [grouping] {i}/{total} questions (timeouts so far: {timeouts_total})",
                      flush=True)

    if _is_math(dataset) and processes != 1 and total > 1:
        import multiprocessing as mp
        with mp.Pool(processes) as pool:
            _consume(pool.imap(_score_question, payloads, chunksize=8))
    else:
        _consume(_score_question(pl) for pl in payloads)

    if stats is not None:
        stats["timeouts_total"] = timeouts_total
        stats["questions_with_timeout"] = questions_with_timeout

    mapping: dict[str, dict] = {}
    cols: dict[str, list] = {k: [] for k in
                             ("dataset", "sample_id", "rollout_id", "group", "correct", "scorer_version")}
    for sid, per in results:
        m = mapping.setdefault(sid, {"rollout_ids": [], "groups": [], "correct": []})
        for rid, grp, ok in per:
            m["rollout_ids"].append(int(rid))
            m["groups"].append(int(grp))
            m["correct"].append(bool(ok))
            cols["dataset"].append(dataset)
            cols["sample_id"].append(sid)
            cols["rollout_id"].append(int(rid))
            cols["group"].append(int(grp))
            cols["correct"].append(bool(ok))
            cols["scorer_version"].append(SCORER_VERSION)

    if use_cache and path is not None:
        try:
            ensure_dir(path.parent)
            pq.write_table(pa.table(cols), os.fspath(path))
        except Exception:
            pass
    return mapping


# =================================================================================================
# `python -m task3_ttc.data --config ... --check`
# =================================================================================================

_EXPECTED_NULLS = {"math": 521, "mathvista": 3032}
_MC_LETTERS = set("ABCDE")


def run_checks(cfg: dict[str, Any]) -> None:
    """Print data-integrity + scorer self-checks for every configured model. (handout p11-12)"""
    for spec in cfg["data"]["models"]:
        dataset = str(spec["dataset"])
        is_vlm = spec.get("type") == "vlm"
        print(f"\n===== {spec['folder']}  (dataset={dataset}) =====")

        t0 = time.perf_counter()
        bundle = _read_model_tables(cfg, spec, strict=False)
        load_s = time.perf_counter() - t0
        questions, diag = bundle["questions"], bundle["diag"]

        print(f"rows: generations={diag['n_gen']} probability={diag['n_prob']} "
              f"attention={diag['n_attn']}")
        exp = _EXPECTED_NULLS.get(dataset)
        tag = "OK" if exp is not None and diag["n_null_extracted"] == exp else "DIFFERS"
        print(f"null extracted_answer: {diag['n_null_extracted']} "
              f"(expected {exp}) -> {tag}")
        align = ("OK" if (diag["n_unmatched"] + diag["mismatch_logp_kl_len"]
                          + diag["mismatch_tokid_len"] + diag["mismatch_attn_len"]) == 0 else "MISMATCH")
        print(f"alignment: {align}  matched={diag['n_matched']} unmatched={diag['n_unmatched']} "
              f"len(logp/kl)!={diag['mismatch_logp_kl_len']} len(token_ids)!="
              f"{diag['mismatch_tokid_len']} len(attn)!={diag['mismatch_attn_len']}")
        print(f"questions={diag['n_questions']} rollouts={diag['n_rollouts']}")

        if _is_math(dataset):
            rate, fails = math_gold_self_verify([q.gold for q in questions])
            print(f"MATH gold self-verify: {rate:.4f} ({len(fails)} distinct failures)")
            if fails:
                print(f"  sample failures: {fails[:5]}")

        if is_vlm:
            _report_mathvista_metadata(cfg, questions)

        t1 = time.perf_counter()
        gstats: dict[str, int] = {}
        gmap = compute_groups(cfg, spec, questions, use_cache=False,
                              progress_every=100, stats=gstats)
        group_s = time.perf_counter() - t1

        distinct_counts, no_answer = [], 0
        total_roll = 0
        for q in questions:
            m = gmap.get(q.sample_id, {"groups": []})
            grps = m["groups"]
            distinct_counts.append(len({g for g in grps if g != 0}))
            no_answer += sum(1 for g in grps if g == 0)
            total_roll += len(grps)
        mean_distinct = float(np.mean(distinct_counts)) if distinct_counts else 0.0
        na_rate = no_answer / total_roll if total_roll else 0.0
        print(f"mean distinct answer groups/question: {mean_distinct:.3f}")
        print(f"NO_ANSWER rate: {na_rate:.4f} ({no_answer}/{total_roll})")
        if _is_math(dataset):
            print(f"math-verify timeouts: {gstats.get('timeouts_total', 0)} total across "
                  f"{gstats.get('questions_with_timeout', 0)} questions "
                  f"(a timeout is treated as NOT equivalent)")
        print(f"timing: load={load_s:.1f}s grouping={group_s:.1f}s")


def _report_mathvista_metadata(cfg: dict[str, Any], questions: list[QuestionRollouts]) -> None:
    md_map = load_mathvista_metadata(cfg)
    present = [q for q in questions if q.sample_id in md_map]
    missing = [q.sample_id for q in questions if q.sample_id not in md_map]
    print(f"MathVista metadata join: {len(present)}/{len(questions)} pids found "
          f"({'all found' if not missing else f'MISSING {len(missing)}: {missing[:5]}'})")

    gold_diff = 0
    mc_gold_maps = 0
    mc_total = mc_letters = 0
    for q in present:
        md = md_map.get(q.sample_id) or {}
        meta_ans = md.get("answer")
        if str(q.gold) != str(meta_ans):
            gold_diff += 1
        choices = md.get("choices") or []
        is_mc = bool(choices) or str(md.get("question_type") or "").lower() == "multi_choice"
        if is_mc:
            gl = _mv_mc_letter(q.gold, choices)
            ml = _mv_mc_letter(str(meta_ans), choices)
            if gl in _MC_LETTERS and gl == ml:
                mc_gold_maps += 1
            for r in q.rollouts:
                ex = r.get("extracted")
                if ex is None:
                    continue
                mc_total += 1
                if _mv_mc_letter(ex, choices) in _MC_LETTERS:
                    mc_letters += 1
    print(f"supplied gold != metadata answer: {gold_diff}/{len(present)} "
          f"(MC gold letters mapping to metadata choice text: {mc_gold_maps})")
    frac = mc_letters / mc_total if mc_total else 0.0
    print(f"MathVista MC extracted answers that are letters: {frac:.4f} "
          f"({mc_letters}/{mc_total}; {mc_total - mc_letters} other)")

    # (a) numeric-type extractions that still do not parse; (b) non-MC gold vs metadata answer
    # canonicalization mismatches; (c) numeric gold answers that do not parse.
    num_unparse = Counter()
    nonmc_gold_diff = 0
    nonmc_gold_diff_ex: list = []
    gold_num_fail = 0
    gold_num_fail_ex: list = []
    for q in present:
        md = md_map.get(q.sample_id) or {}
        choices = md.get("choices") or []
        if bool(choices) or str(md.get("question_type") or "").lower() == "multi_choice":
            continue                                    # non-multiple-choice only
        atype = str(md.get("answer_type") or "").lower()
        unit = md.get("unit")
        meta_ans = md.get("answer")
        if meta_ans is not None and _mathvista_canonical(q.gold, md) != _mathvista_canonical(meta_ans, md):
            nonmc_gold_diff += 1
            if len(nonmc_gold_diff_ex) < 10:
                nonmc_gold_diff_ex.append((q.sample_id, q.gold, meta_ans))
        if atype in ("integer", "float"):
            if _mv_numeric_value(q.gold, unit) is None:
                gold_num_fail += 1
                if len(gold_num_fail_ex) < 20:
                    gold_num_fail_ex.append((q.sample_id, q.gold))
            for r in q.rollouts:
                ex = r.get("extracted")
                if ex is not None and _mv_numeric_value(ex, unit) is None:
                    num_unparse[str(ex)] += 1

    print(f"numeric (integer/float) extractions still unparseable: {sum(num_unparse.values())}")
    for ex, c in num_unparse.most_common(20):
        print(f"    {c:5d}  {ex!r}")
    print(f"non-MC gold != metadata answer (canonical): {nonmc_gold_diff}")
    for sid, g, ma in nonmc_gold_diff_ex:
        print(f"    {sid}: gold={g!r} meta={ma!r}")
    print(f"numeric gold answers that do not parse: {gold_num_fail}")
    for sid, g in gold_num_fail_ex:
        print(f"    {sid}: {g!r}")


def main() -> None:
    from common.config import build_arg_parser, load_yaml

    parser = build_arg_parser("Task 3 data loading / scorer self-checks.")
    parser.add_argument("--check", action="store_true",
                        help="Run data-integrity + scorer self-checks and exit.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)

    if args.check:
        run_checks(cfg)
        return
    for spec in cfg["data"]["models"]:
        qs = load_joined(cfg, spec)
        print(f"{spec['folder']}: {len(qs)} questions, "
              f"{sum(len(q.rollouts) for q in qs)} rollouts.")


if __name__ == "__main__":
    main()
