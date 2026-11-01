"""GSM8K answer-extraction + truncation audit across the seven conditions. (handout Step 5/p4)

Reads the per-example GSM8K outputs lm-eval wrote during evaluation
(DRIVE_ROOT/task1/eval/<label>/gsm8k.jsonl) and, per condition, counts:
  * correct               — lm-eval scored the flexible-extract answer exact_match == 1,
  * wrong-but-extracted   — a number was extracted but lm-eval scored it incorrect,
  * unextractable         — lm-eval's flexible-extract produced "[invalid]"/empty (scored incorrect, p3),
  * truncated             — the generation reached the 1,024-new-token cap, measured by re-tokenizing
                            the raw generation with the Qwen3 tokenizer.
It also reports, per condition, how often the extracted answer and the correctness flip versus the
shared baseline. Outputs results/task1/gsm8k_extraction.json plus a compact per-example CSV
(results/task1/gsm8k_per_example.csv: label, doc_id, extracted, correct, truncated, n_tokens).

Source of truth: lm-eval, not our own parser. Each gsm8k.jsonl holds one row per (doc_id, filter);
we keep only the flexible-extract rows (matching the "filter" field) and take correctness from the
per-sample `exact_match` field and the extracted answer from `filtered_resps[0]` ("[invalid]"/empty
means unextractable). This avoids our parser being more lenient than lm-eval's flexible-extract
filter on messy baseline outputs. The distinct filter names found in each file are printed.

Robustness note: the raw sample files live on DRIVE_ROOT (not in the repo). Truncation is still
measured ourselves by re-tokenizing the raw generation (read from the first non-empty of the known
lm-eval keys, "resps" first) with the Qwen3 tokenizer against the 1,024-new-token cap; samples are
de-duplicated by doc_id. The recomputed exact-match (mean of lm-eval's per-sample exact_match) is
cross-checked against the committed eval JSON's `exact_match,flexible-extract`; a mismatch is a hard
error and both numbers are recorded.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

from common.config import build_arg_parser, load_yaml
from common.paths import RESULTS_DIR, ensure_dir, task1_dir

from .evaluate import _resolve_conditions

log = logging.getLogger(__name__)

# Candidate keys for the raw generation text, in preference order (differs across lm-eval layouts).
# "resps" (the full, pre-filter generation) is first — that is what the truncation check re-tokenizes.
_RAW_KEYS = ("resps", "response", "resp", "generation")
_FLEXIBLE_FILTER = "flexible-extract"   # the only lm-eval filter rows we score from
_INVALID_EXTRACTION = ("", "[invalid]")  # filtered_resps[0] values that mean "unextractable"
_DEFAULT_MAX_NEW_TOKENS = 1024
_EM_TOL = 1e-6          # tolerance when cross-checking recomputed vs eval-JSON flexible exact_match


# ---------------------------------------------------------------------------------------------
# Field extraction (schema-tolerant)
# ---------------------------------------------------------------------------------------------
def _flatten_to_str(value: Any) -> Optional[str]:
    """Reduce a possibly-nested list/tuple (lm-eval resps are [[text]]) to its first string."""
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _raw_generation(sample: dict[str, Any]) -> Optional[str]:
    """The full, pre-filter raw generation text (needed for the truncation measurement)."""
    for key in _RAW_KEYS:
        if key in sample:
            text = _flatten_to_str(sample[key])
            if text:
                return text
    return None


def _extracted_answer(sample: dict[str, Any]) -> Optional[str]:
    """lm-eval's flexible-extract answer: filtered_resps[0], or None if "[invalid]"/empty."""
    text = _flatten_to_str(sample.get("filtered_resps"))
    if text is None:
        return None
    text = text.strip()
    return None if text in _INVALID_EXTRACTION else text


def _numbers_equal(a: Optional[str], b: Optional[str]) -> bool:
    """Compare two normalized number strings, tolerant of 72 vs 72.0; falls back to string equality."""
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return a == b


# ---------------------------------------------------------------------------------------------
# Per-condition processing
# ---------------------------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def process_condition(
    samples: list[dict[str, Any]], tokenizer, max_new_tokens: int
) -> tuple[dict[int, dict], set[str]]:
    """Return ({doc_id: record}, {distinct filter names seen}) for one condition.

    lm-eval writes one row per (doc_id, filter); we score only the flexible-extract rows,
    de-duplicated by doc_id (first occurrence). Correctness and extraction come straight from
    lm-eval (per-sample exact_match, filtered_resps[0]); only truncation is measured ourselves.
    record: {extracted, extractable, correct, n_tokens, truncated}.
    """
    filters_seen = {s.get("filter") for s in samples if s.get("filter") is not None}
    records: dict[int, dict[str, Any]] = {}
    for sample in samples:
        if sample.get("filter") != _FLEXIBLE_FILTER:
            continue
        doc_id = sample.get("doc_id")
        if doc_id is None or doc_id in records:
            continue
        extracted = _extracted_answer(sample)
        extractable = extracted is not None
        correct = bool(round(float(sample.get("exact_match", 0.0))))

        raw = _raw_generation(sample)
        n_tokens = len(tokenizer(raw, add_special_tokens=False).input_ids) if raw else 0
        truncated = n_tokens >= max_new_tokens

        records[doc_id] = {
            "extracted": extracted,
            "extractable": extractable,
            "correct": correct,
            "n_tokens": n_tokens,
            "truncated": truncated,
        }
    return records, filters_seen


def _eval_json_flexible_em(label: str) -> Optional[float]:
    """Read the committed eval JSON's gsm8k flexible-extract exact_match for cross-checking."""
    path = RESULTS_DIR / "task1" / "eval" / f"{label}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["benchmarks"]["gsm8k"]["metrics"]
        return metrics.get("exact_match,flexible-extract")
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _summarize(label: str, records: dict[int, dict], baseline: Optional[dict[int, dict]]) -> dict:
    """Aggregate counts for one condition + change-vs-baseline tallies."""
    n = len(records)
    correct = sum(r["correct"] for r in records.values())
    unextractable = sum(not r["extractable"] for r in records.values())
    wrong_extracted = sum(r["extractable"] and not r["correct"] for r in records.values())
    truncated = sum(r["truncated"] for r in records.values())

    recomputed_em = correct / n if n else None
    eval_em = _eval_json_flexible_em(label)
    if recomputed_em is not None and eval_em is not None and abs(recomputed_em - eval_em) > _EM_TOL:
        raise ValueError(
            f"{label}: recomputed flexible exact_match {recomputed_em:.4f} differs from eval JSON "
            f"{eval_em:.4f} (>{_EM_TOL:.0e}); the mean of lm-eval's per-sample exact_match disagrees "
            f"with the committed eval JSON — the gsm8k.jsonl and eval JSON are out of sync."
        )

    summary = {
        "n_examples": n,
        "correct": int(correct),
        "wrong_extracted": int(wrong_extracted),
        "unextractable": int(unextractable),
        "truncated": int(truncated),
        "exact_match_recomputed": recomputed_em,
        "exact_match_eval_json": eval_em,
    }

    if baseline is not None:
        shared = [doc_id for doc_id in records if doc_id in baseline]
        changed_answer = sum(
            not _numbers_equal(records[d]["extracted"], baseline[d]["extracted"])
            and not (records[d]["extracted"] is None and baseline[d]["extracted"] is None)
            for d in shared
        )
        changed_correct = sum(records[d]["correct"] != baseline[d]["correct"] for d in shared)
        gained = sum(records[d]["correct"] and not baseline[d]["correct"] for d in shared)
        lost = sum(not records[d]["correct"] and baseline[d]["correct"] for d in shared)
        summary.update({
            "n_shared_with_baseline": len(shared),
            "changed_answer_vs_baseline": int(changed_answer),
            "changed_correctness_vs_baseline": int(changed_correct),
            "correct_gained_vs_baseline": int(gained),
            "correct_lost_vs_baseline": int(lost),
        })
    return summary


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------
def run(eval_cfg: dict[str, Any]) -> dict:
    """Process every condition's gsm8k.jsonl; write the JSON summary + per-example CSV."""
    from transformers import AutoTokenizer

    jobs = _resolve_conditions(eval_cfg)
    baseline_label = eval_cfg.get("baseline", {}).get("label", "baseline")
    max_new_tokens = int(eval_cfg.get("benchmarks", {}).get("gsm8k", {})
                         .get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))

    # Qwen3 tokenizer for the truncation measurement (model name read from the baseline train config).
    base_cfg = load_yaml(eval_cfg["baseline"]["config"])
    tokenizer = AutoTokenizer.from_pretrained(base_cfg["model"]["name"])

    # First pass: load + process every available condition into per-doc records.
    per_condition: dict[str, dict[int, dict]] = {}
    for job in jobs:
        label = job["label"]
        path = task1_dir("eval") / label / "gsm8k.jsonl"
        if not path.exists():
            log.warning("Missing GSM8K samples for %s (%s); skipping.", label, path)
            continue
        records, filters_seen = process_condition(_load_jsonl(path), tokenizer, max_new_tokens)
        per_condition[label] = records
        print(f"[gsm8k-extraction] {label}: filters found = {sorted(filters_seen)}", flush=True)
        log.info("%s: %d flexible-extract examples.", label, len(records))

    baseline_records = per_condition.get(baseline_label)

    # Summaries (change-vs-baseline for every non-baseline condition).
    summary: dict[str, Any] = {
        "baseline": baseline_label,
        "max_new_tokens": max_new_tokens,
        "conditions": {},
    }
    for job in jobs:
        label = job["label"]
        if label not in per_condition:
            continue
        base = None if label == baseline_label else baseline_records
        summary["conditions"][label] = _summarize(label, per_condition[label], base)

    out_dir = ensure_dir(RESULTS_DIR / "task1")
    json_path = out_dir / "gsm8k_extraction.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Compact per-example CSV (all conditions, in job order then doc_id).
    csv_path = out_dir / "gsm8k_per_example.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "doc_id", "extracted", "correct", "truncated", "n_tokens"])
        for job in jobs:
            label = job["label"]
            records = per_condition.get(label)
            if not records:
                continue
            for doc_id in sorted(records):
                r = records[doc_id]
                writer.writerow([label, doc_id, r["extracted"] if r["extracted"] is not None else "",
                                 int(r["correct"]), int(r["truncated"]), r["n_tokens"]])

    print(f"[gsm8k-extraction] {len(per_condition)} condition(s); wrote {json_path} and {csv_path}.",
          flush=True)
    return summary


def main() -> None:
    """CLI: --config (eval config enumerating the seven conditions); --drive-root for the samples."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser("GSM8K extraction + truncation audit across the seven conditions.")
    args = parser.parse_args()
    if args.drive_root:
        import os
        os.environ["DRIVE_ROOT"] = args.drive_root

    eval_cfg = load_yaml(args.config)
    run(eval_cfg)


if __name__ == "__main__":
    main()
