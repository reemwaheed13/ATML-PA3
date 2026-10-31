"""Qualitative generations: 168 responses across prompts x conditions. (handout p5, Step 7)

Using the 24 prompts (8 questions x {plain, JSON, template}) from first_plot_questions.yaml, this
generates one greedy response per prompt (thinking disabled, 512-new-token cap) for each of the
seven evaluation conditions: baseline, Full FT @625, Full FT @1250, L2-SP, LoRA-4, LoRA-16, OPLoRA.
24 prompts x 7 conditions = 168 responses.

Reuse, not duplication: the seven conditions and every model-construction detail (FP16, eval mode,
adapters merged into a fresh backbone with the Step-5 merge check, OPLoRA projected factors) come
straight from evaluate.py — `_resolve_conditions` enumerates the jobs and `build_eval_model` builds
each model. This file only adds the prompting + greedy generation + annotation-subset bookkeeping.

Durability: each condition's 24 responses are written to their own file on DRIVE_ROOT the moment it
finishes, and a condition whose file already exists (and is complete) is skipped — so a Colab
disconnect loses at most the one condition in flight. Once all seven files exist they are combined
into results/task1/qualitative/responses.jsonl (168 rows, committed), and the 42-row
annotation_sheet.csv (first + last base question x 3 formats x 7 conditions) is written for manual
scoring (the scoring columns are left empty).

Prompt format (p5): user message = the entry's first paraphrase; the JSON variants additionally
carry a `system` message. Rendered with the Qwen3 chat template, enable_thinking=False,
add_generation_prompt=True; generated greedily. Batching all 24 per condition with left padding is
fine (the prompt region is then uniform width, so new tokens start at a single column for the batch).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from common.config import build_arg_parser, load_yaml
from common.paths import REPO_ROOT, RESULTS_DIR, ensure_dir, task1_dir

from .evaluate import _resolve_conditions, build_eval_model

log = logging.getLogger(__name__)

_FORMAT_ORDER = ["plain", "json", "template"]
_RESPONSE_FIELDS = [
    "condition", "id", "base_question_id", "format", "system", "prompt",
    "response", "n_new_tokens", "truncated",
]
# annotation_sheet.csv: the machine-filled columns, then the EMPTY ones to score by hand.
_ANNOTATION_AUTO = ["condition", "id", "format", "truncated", "response"]
_ANNOTATION_MANUAL = [
    "format_compliance", "coherence_relevance", "repetition",
    "unnecessary_math_framing", "inappropriate_content", "excerpt", "notes",
]


# ---------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------
def _format_of(entry_id: str) -> tuple[str, str]:
    """Map an entry id to (format, base_question_id) by its suffix (_json / _template / none)."""
    if entry_id.endswith("_json"):
        return "json", entry_id[: -len("_json")]
    if entry_id.endswith("_template"):
        return "template", entry_id[: -len("_template")]
    return "plain", entry_id


def load_prompts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the 24 prompt records from the configured first_plot_questions.yaml, in file order.

    Each record: {id, base_question_id, format, system (str|None), prompt}. `prompt` is the entry's
    first paraphrase (the user message); `system` is the entry's system field if present (only the
    _json variants have one). The configured path is used as-is if absolute/existing, else resolved
    against the repo root.
    """
    raw = cfg["prompts"]["file"]
    path = Path(raw)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / raw
    with open(path, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f)

    records: list[dict[str, Any]] = []
    for entry in entries:
        entry_id = entry["id"]
        fmt, base = _format_of(entry_id)
        records.append({
            "id": entry_id,
            "base_question_id": base,
            "format": fmt,
            "system": entry.get("system"),
            "prompt": entry["paraphrases"][0],
        })
    return records


# ---------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------
def _eot_ids(tokenizer) -> list[int]:
    """End-of-turn token ids that stop a Qwen3 turn: the tokenizer eos plus <|im_end|> if distinct."""
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return sorted(ids)


def _render(tokenizer, record: dict[str, Any]) -> str:
    """Render one record to a Qwen3 chat prompt string (system optional, non-thinking)."""
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": record["prompt"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _trim_new_tokens(new_ids: list[int], eot: set[int], max_new_tokens: int) -> tuple[int, bool]:
    """Return (n_new_tokens, truncated) for one sample's generated-token slice.

    Stops at the first end-of-turn token (counted, then stripped for the text); if none appears the
    response hit the new-token cap -> truncated. Robust to pad==eos (Qwen3 has no dedicated pad, so
    build_eval_model aliases pad to eos): trailing padding only ever follows the first eot.
    """
    for k, tok in enumerate(new_ids):
        if tok in eot:
            return k + 1, False
    return len(new_ids), len(new_ids) >= max_new_tokens


def generate_for_condition(model, tokenizer, device, records: list[dict[str, Any]], *,
                           max_new_tokens: int, batch_size: int) -> list[dict[str, Any]]:
    """Greedy-generate one response per record for an already-built condition model.

    Returns a list aligned to `records`, each {response, n_new_tokens, truncated}. Left padding
    makes the prompt region a uniform width per batch, so new tokens start at a single column.
    """
    tokenizer.padding_side = "left"
    eot = _eot_ids(tokenizer)
    eot_set = set(eot)

    out: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        prompts = [_render(tokenizer, r) for r in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,                       # greedy
                eos_token_id=eot,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i in range(len(chunk)):
            new_ids = gen[i, input_len:].tolist()
            n_new, truncated = _trim_new_tokens(new_ids, eot_set, max_new_tokens)
            text = tokenizer.decode(new_ids[:n_new], skip_special_tokens=True).strip()
            out.append({"response": text, "n_new_tokens": n_new, "truncated": truncated})
    return out


# ---------------------------------------------------------------------------------------------
# Per-condition files (durable) + combine
# ---------------------------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write rows as JSONL (temp file in the same dir, then os.replace)."""
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".jsonl.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_complete(path: Path, expected_rows: int) -> bool:
    """A condition file counts as done only if it exists and holds the full row count."""
    if not path.exists():
        return False
    try:
        return len(_read_jsonl(path)) == expected_rows
    except (OSError, json.JSONDecodeError):
        return False


def _condition_rows(label: str, records: list[dict[str, Any]],
                    gens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Zip records + generations into full response rows for one condition."""
    rows = []
    for rec, gen in zip(records, gens):
        rows.append({
            "condition": label,
            "id": rec["id"],
            "base_question_id": rec["base_question_id"],
            "format": rec["format"],
            "system": rec["system"] or "",
            "prompt": rec["prompt"],
            "response": gen["response"],
            "n_new_tokens": gen["n_new_tokens"],
            "truncated": gen["truncated"],
        })
    return rows


# ---------------------------------------------------------------------------------------------
# Annotation subset
# ---------------------------------------------------------------------------------------------
def _write_annotation_sheet(path: Path, rows: list[dict[str, Any]], *,
                            questions: list[str], formats: list[str],
                            condition_order: list[str]) -> int:
    """Write the annotation CSV for `questions` x `formats` x conditions, sorted q/format/condition.

    Carries machine-readable fields (condition, id, format, truncated, response) and leaves the
    manual scoring columns empty. Returns the number of rows written.
    """
    q_index = {q: i for i, q in enumerate(questions)}
    f_index = {fmt: i for i, fmt in enumerate(formats)}
    c_index = {c: i for i, c in enumerate(condition_order)}

    subset = [r for r in rows if r["base_question_id"] in q_index and r["format"] in f_index]
    subset.sort(key=lambda r: (q_index[r["base_question_id"]], f_index[r["format"]],
                               c_index.get(r["condition"], len(c_index))))

    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ANNOTATION_AUTO + _ANNOTATION_MANUAL)
        writer.writeheader()
        for r in subset:
            row = {k: r[k] for k in _ANNOTATION_AUTO}
            row.update({k: "" for k in _ANNOTATION_MANUAL})
            writer.writerow(row)
    return len(subset)


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------
def run(cfg: dict[str, Any], device, *, smoke: bool = False) -> None:
    """Produce the 168 generations (durable per-condition files) + combined outputs.

    --smoke: 1 condition, 2 prompts, 32 new tokens; outputs go to dedicated smoke paths and the
    42-row annotation sheet is skipped (it needs all conditions).
    """
    gen_cfg = cfg["generation"]
    max_new_tokens = 32 if smoke else int(gen_cfg["max_new_tokens"])
    batch_size = int(gen_cfg.get("batch_size", 24))

    records = load_prompts(cfg)
    expected = int(cfg["prompts"].get("n_entries", len(records)))
    if not smoke and len(records) != expected:
        log.warning("Loaded %d prompts but config expects %d.", len(records), expected)

    jobs = _resolve_conditions(cfg)
    if smoke:
        jobs = jobs[:1]
        records = records[:2]

    cond_dir = ensure_dir(task1_dir("generations") / ("qualitative_smoke" if smoke else "qualitative"))
    merge_check_cfg = cfg.get("merge_check", {})

    print(f"[qualitative] {len(jobs)} condition(s) x {len(records)} prompt(s) on {device}; "
          f"max_new_tokens={max_new_tokens}.", flush=True)

    for job in jobs:
        label = job["label"]
        cond_path = cond_dir / f"{label}.jsonl"
        if _is_complete(cond_path, len(records)):
            print(f"[qualitative] skip {label} (already complete: {cond_path.name}).", flush=True)
            continue

        log.info("Generating condition %s (update %d, ckpt=%s).",
                 label, job["update"], job["ckpt_path"] or "-")
        model, tokenizer, _ = build_eval_model(
            job["config"], job["ckpt_path"], device, merge_check_cfg=merge_check_cfg)
        gens = generate_for_condition(
            model, tokenizer, device, records,
            max_new_tokens=max_new_tokens, batch_size=batch_size)
        _write_jsonl(cond_path, _condition_rows(label, records, gens))
        print(f"[qualitative] wrote {label} -> {cond_path} ({len(records)} rows).", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Combine the per-condition files (in job order) into the committed outputs.
    all_rows: list[dict[str, Any]] = []
    for job in jobs:
        all_rows.extend(_read_jsonl(cond_dir / f"{job['label']}.jsonl"))

    results_dir = ensure_dir(RESULTS_DIR / "task1" / ("qualitative/smoke" if smoke else "qualitative"))
    responses_path = results_dir / "responses.jsonl"
    _write_jsonl(responses_path, all_rows)
    print(f"[qualitative] combined {len(all_rows)} rows -> {responses_path}.", flush=True)

    if smoke:
        print("[qualitative] smoke run: skipping annotation_sheet.csv.", flush=True)
        return

    ann_cfg = cfg.get("annotation", {})
    n_ann = _write_annotation_sheet(
        results_dir / "annotation_sheet.csv", all_rows,
        questions=list(ann_cfg.get("questions", [])),
        formats=list(ann_cfg.get("formats", _FORMAT_ORDER)),
        condition_order=[job["label"] for job in jobs],
    )
    print(f"[qualitative] wrote annotation_sheet.csv ({n_ann} rows).", flush=True)


def main() -> None:
    """CLI: parse --config (+ --smoke / --drive-root); generate the seven conditions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser("Task 1 qualitative generations (Step 7; seven conditions).")
    parser.add_argument("--smoke", action="store_true",
                        help="1 condition, 2 prompts, 32 new tokens (quick wiring check).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(cfg, device, smoke=args.smoke)


if __name__ == "__main__":
    main()
