"""MetaMathQA selection, filtering, and response-only loss masking. (handout p3)

Selection (p3 "Training data"): first 40,000 ELIGIBLE examples, in source order. Exclude
examples with empty questions/responses, formatted length > 2,048 tokens, or a
query/original_question matching a GSM8K test question after text normalization. Do NOT truncate
answers or shuffle retained examples.

The kept indices are frozen once into splits/task1/metamathqa_kept_indices.json and reused, so
training data is identical across runs and across resumes.

Loss masking (p3): format each example as a user question + supplied assistant response; compute
next-token cross-entropy only on assistant-response tokens and the end-of-turn token; mask
prompts and padding; normalize accumulated loss over all valid response tokens in the effective
batch.

Selection CLI (freezes the split; run once on Colab):

    python -m task1_llm.data --config configs/task1/full_ft.yaml
    python -m task1_llm.data --config configs/task1/full_ft.yaml --force   # overwrite existing

It writes the kept indices, the normalization rule used, and per-stage counts (total scanned,
removed by each filter, eligible, kept), then prints a couple of masked examples so the
response-only masking can be verified by eye.
"""

from __future__ import annotations

import json
import os
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import torch

from common import SEED
from common.config import build_arg_parser, load_yaml
from common.paths import SPLITS_DIR
from common.seeding import seed_all

IGNORE_INDEX = -100  # torch cross-entropy ignore_index: masks prompt + padding tokens.

# Where the frozen selection lives (committed; see splits/README.md).
KEPT_INDICES_PATH = SPLITS_DIR / "task1" / "metamathqa_kept_indices.json"

# Default dataset coordinates + column names (overridable via cfg["data"]).
DEFAULT_DATASET = "meta-math/MetaMathQA"
DEFAULT_SPLIT = "train"
DEFAULT_FIELDS = {"query": "query", "original_question": "original_question", "response": "response"}

# GSM8K test set used for decontamination. "openai/gsm8k" is the canonical parquet-based repo
# (no dataset loading script, so no trust_remote_code prompt on Colab).
GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"


@dataclass
class ExampleBatch:
    """A microbatch of tokenized examples with response-only loss labels.

    Attributes:
        input_ids, attention_mask: standard tokenizer tensors.
        labels: response + end-of-turn tokens kept; prompt and padding set to ignore index.
        n_response_tokens: count of valid response tokens (for effective-batch normalization).
    """

    input_ids: Any
    attention_mask: Any
    labels: Any
    n_response_tokens: int


# --------------------------------------------------------------------------------------------
# GSM8K-decontamination text normalization.
#
# The exact rule is a TA-question (configs/task1/_base.yaml `normalization`). We implement it as
# a registry of NAMED rules selected by cfg, so swapping the rule after the TAs answer is a
# one-line config change and the chosen name is frozen into the split file for reproducibility.
# --------------------------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _norm_lowercase_strip_punct_ws(text: str) -> str:
    """Default rule: lowercase, strip ASCII punctuation, collapse all whitespace to single spaces."""
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())


def _norm_identity(text: str) -> str:
    """No normalization (exact string match) — available for ablation / strict matching."""
    return text


NORMALIZATION_RULES = {
    "lowercase_strip_punct_ws": _norm_lowercase_strip_punct_ws,
    "identity": _norm_identity,
}
DEFAULT_NORMALIZATION_RULE = "lowercase_strip_punct_ws"


def normalize_text_for_decontam(text: str, rule: str) -> str:
    """Normalize a question string for GSM8K-match decontamination using a named rule.

    The same rule is applied to both the GSM8K test questions and the MetaMathQA
    query/original_question so the comparison is apples-to-apples. The chosen rule name is
    recorded alongside the frozen kept-indices so the selection is reproducible.
    """
    try:
        fn = NORMALIZATION_RULES[rule]
    except KeyError:
        raise ValueError(
            f"Unknown normalization rule {rule!r}; known rules: {sorted(NORMALIZATION_RULES)}"
        )
    return fn(text or "")


def build_gsm8k_test_question_set(rule: str) -> set[str]:
    """Return the set of normalized GSM8K test questions used to exclude contaminated examples."""
    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split="test")
    return {normalize_text_for_decontam(q, rule) for q in ds["question"]}


# --------------------------------------------------------------------------------------------
# Chat formatting + response-only masking.
# --------------------------------------------------------------------------------------------


def _render_token_ids(tokenizer, messages, *, add_generation_prompt: bool,
                      enable_thinking: bool) -> list[int]:
    """Render a chat conversation to a flat list[int] of token ids, version-robustly.

    apply_chat_template's return type varies across transformers versions and tokenize settings
    (list[int], a tensor, a BatchEncoding/dict, or — as seen on Colab's transformers with Python
    3.13 — the rendered *string*). To avoid depending on that, we render to text (tokenize=False,
    stable everywhere) and tokenize it ourselves with add_special_tokens=False (the template has
    already placed every special token). The result is coerced to a flat list[int].
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    if not isinstance(text, str):
        # Some templates return a 1-element list of strings even with tokenize=False.
        text = text[0] if isinstance(text, (list, tuple)) and text else str(text)

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):  # unwrap an accidental batch dimension
        ids = ids[0]
    return [int(t) for t in ids]


def build_chat_formatted_example(
    question: str,
    response: str,
    tokenizer,
    *,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """Format one example as user question + assistant response with response-only labels.

    Strategy (robust to Qwen3 chat-template quirks): render the prompt alone (with the assistant
    generation prefix) and the full conversation, tokenize both to id lists (see
    _render_token_ids), then mask up to the length of their longest common token prefix.
    Everything after that divergence point is the assistant response plus the end-of-turn token,
    which is what we train on; the shared prefix (system + user + assistant header, and any empty
    <think></think> block the template inserts) is masked. Using the common-prefix boundary
    rather than the raw prompt length keeps the mask correct even if the template renders the
    generation prompt slightly differently from an existing assistant turn.

    Returns a dict with python lists: input_ids, labels (IGNORE_INDEX on prompt+any pad later),
    and n_response_tokens.
    """
    user = {"role": "user", "content": question}
    assistant = {"role": "assistant", "content": response}

    prompt_ids = _render_token_ids(
        tokenizer, [user], add_generation_prompt=True, enable_thinking=enable_thinking
    )
    full_ids = _render_token_ids(
        tokenizer, [user, assistant], add_generation_prompt=False, enable_thinking=enable_thinking
    )

    boundary = 0
    limit = min(len(prompt_ids), len(full_ids))
    while boundary < limit and prompt_ids[boundary] == full_ids[boundary]:
        boundary += 1

    labels = [IGNORE_INDEX] * boundary + full_ids[boundary:]

    # Safety net: training labels must be int lists of equal length to input_ids (this is the
    # shared path that produces labels for both selection and the training loop).
    assert isinstance(full_ids, list) and isinstance(labels, list), "ids/labels must be lists"
    assert len(full_ids) == len(labels), "input_ids and labels length mismatch"
    assert not full_ids or isinstance(full_ids[0], int), "input_ids must be a list[int]"

    return {
        "input_ids": full_ids,
        "labels": labels,
        "n_response_tokens": len(full_ids) - boundary,
    }


def masked_preview(formatted: dict[str, Any], tokenizer) -> str:
    """Render one formatted example showing which tokens are masked vs trained, for eyeballing."""
    input_ids = formatted["input_ids"]
    labels = formatted["labels"]
    boundary = next((i for i, lab in enumerate(labels) if lab != IGNORE_INDEX), len(input_ids))

    masked_text = tokenizer.decode(input_ids[:boundary], skip_special_tokens=False)
    trained_text = tokenizer.decode(input_ids[boundary:], skip_special_tokens=False)
    return (
        f"  masked  ({boundary} tokens, loss=IGNORE):\n"
        f"    «{masked_text}»\n"
        f"  trained ({formatted['n_response_tokens']} tokens, response + end-of-turn):\n"
        f"    ⟪{trained_text}⟫"
    )


# --------------------------------------------------------------------------------------------
# Dataset loading + eligibility selection.
# --------------------------------------------------------------------------------------------


def _data_fields(cfg: dict[str, Any]) -> dict[str, str]:
    fields = dict(DEFAULT_FIELDS)
    fields.update(cfg.get("data", {}).get("fields", {}) or {})
    return fields


def load_metamathqa(cfg: dict[str, Any]):
    """Load the MetaMathQA split named by cfg (default meta-math/MetaMathQA train)."""
    from datasets import load_dataset

    data_cfg = cfg.get("data", {})
    name = data_cfg.get("dataset_hf", DEFAULT_DATASET)
    split = data_cfg.get("split", DEFAULT_SPLIT)
    return load_dataset(name, split=split)


def select_eligible_indices(
    cfg: dict[str, Any],
    tokenizer,
    *,
    dataset=None,
    gsm8k_set: set[str] | None = None,
) -> tuple[list[int], dict[str, int], str, list[int]]:
    """Scan MetaMathQA in source order and return the first N eligible example indices.

    Applies the eligibility filters in the order empty -> GSM8K-decontam -> over-length (the
    length check is last because it is the only one that needs tokenization, so cheap rejects
    happen first). Scanning stops as soon as `num_examples` eligible indices are found, so the
    returned counts describe the consumed source-order prefix (which is exactly what determines
    the frozen selection).

    Returns (kept_indices, counts, normalization_rule, kept_token_lengths). `counts` has:
    total_scanned, removed_empty, removed_decontam, removed_over_len, eligible, kept. The formatted
    token length is computed for every candidate that clears empty+decontam (so length stats are
    available for the kept set even when the over-length filter is disabled); the limit is only
    enforced when exclude_over_max_len is set. Pass `gsm8k_set` to reuse a prebuilt decontam set.
    """
    data_cfg = cfg.get("data", {})
    num_examples = int(data_cfg.get("num_examples", 40000))
    max_len = int(data_cfg.get("max_seq_len", 2048))
    rule = data_cfg.get("normalization", DEFAULT_NORMALIZATION_RULE)
    do_empty = bool(data_cfg.get("exclude_empty", True))
    do_overlen = bool(data_cfg.get("exclude_over_max_len", True))
    do_decontam = bool(data_cfg.get("gsm8k_decontaminate", True))
    enable_thinking = bool(cfg.get("model", {}).get("enable_thinking", False))
    fields = _data_fields(cfg)
    qf, of, rf = fields["query"], fields["original_question"], fields["response"]

    if gsm8k_set is None:
        gsm8k_set = build_gsm8k_test_question_set(rule) if do_decontam else set()
    if dataset is None:
        dataset = load_metamathqa(cfg)

    counts = {
        "total_scanned": 0,
        "removed_empty": 0,
        "removed_decontam": 0,
        "removed_over_len": 0,
        "eligible": 0,
        "kept": 0,
    }
    kept: list[int] = []
    kept_lengths: list[int] = []

    for i, ex in enumerate(dataset):
        counts["total_scanned"] += 1
        query = (ex.get(qf) or "").strip()
        response = (ex.get(rf) or "").strip()
        original = (ex.get(of) or "").strip()

        if do_empty and (not query or not response):
            counts["removed_empty"] += 1
            continue

        if do_decontam and (
            normalize_text_for_decontam(query, rule) in gsm8k_set
            or (original and normalize_text_for_decontam(original, rule) in gsm8k_set)
        ):
            counts["removed_decontam"] += 1
            continue

        formatted = build_chat_formatted_example(
            query, response, tokenizer, enable_thinking=enable_thinking
        )
        length = len(formatted["input_ids"])
        if do_overlen and length > max_len:
            counts["removed_over_len"] += 1
            continue

        kept.append(i)
        kept_lengths.append(length)
        if len(kept) >= num_examples:
            break

    counts["eligible"] = len(kept)
    counts["kept"] = len(kept)
    return kept, counts, rule, kept_lengths


def save_kept_indices(
    path,
    kept_indices: list[int],
    counts: dict[str, int],
    cfg: dict[str, Any],
    rule: str,
    *,
    force: bool = False,
) -> None:
    """Write the frozen selection to `path` as JSON. Refuses to overwrite unless force=True."""
    path = os.fspath(path)
    if os.path.exists(path) and not force:
        raise FileExistsError(
            f"{path} already exists; selection is write-once. Pass --force to overwrite "
            f"(this changes the training set — only do so deliberately)."
        )

    data_cfg = cfg.get("data", {})
    payload = {
        "dataset": data_cfg.get("dataset", "MetaMathQA"),
        "dataset_hf": data_cfg.get("dataset_hf", DEFAULT_DATASET),
        "split": data_cfg.get("split", DEFAULT_SPLIT),
        "seed": SEED,
        "num_examples_requested": int(data_cfg.get("num_examples", 40000)),
        "max_seq_len": int(data_cfg.get("max_seq_len", 2048)),
        "normalization_rule": rule,
        "fields": _data_fields(cfg),
        "counts": counts,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kept_indices": kept_indices,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_kept_indices(path=KEPT_INDICES_PATH) -> dict[str, Any]:
    """Load the frozen selection written by save_kept_indices (used by the training loop)."""
    with open(os.fspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------
# Resumable microbatch iteration.
# --------------------------------------------------------------------------------------------


def iterate_microbatches(
    cfg: dict[str, Any],
    tokenizer,
    kept_indices: list[int],
    *,
    start_position: int = 0,
    dataset=None,
) -> Iterator[ExampleBatch]:
    """Yield microbatches over the kept examples, resumable from `start_position`.

    `start_position` is a cursor **in examples** (not microbatches), so --resume continues from
    the exact data position stored in the checkpoint and advances by microbatch_size each yield.
    Order is the frozen source order (no shuffling). Each batch is right-padded to the batch's
    longest sequence; padding is masked in both attention_mask and labels.
    """
    data_cfg = cfg.get("data", {})
    mb = int(cfg.get("train", {}).get("microbatch_size", 2))
    enable_thinking = bool(cfg.get("model", {}).get("enable_thinking", False))
    fields = _data_fields(cfg)
    qf, rf = fields["query"], fields["response"]

    if dataset is None:
        dataset = load_metamathqa(cfg)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    pos = start_position
    n = len(kept_indices)
    while pos < n:
        batch_src = kept_indices[pos : pos + mb]
        formatted = [
            build_chat_formatted_example(
                (dataset[j].get(qf) or "").strip(),
                (dataset[j].get(rf) or "").strip(),
                tokenizer,
                enable_thinking=enable_thinking,
            )
            for j in batch_src
        ]
        width = max(len(e["input_ids"]) for e in formatted)

        input_ids, attention_mask, labels = [], [], []
        for e in formatted:
            ids = e["input_ids"]
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(e["labels"] + [IGNORE_INDEX] * pad)

        yield ExampleBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            n_response_tokens=sum(e["n_response_tokens"] for e in formatted),
        )
        pos += mb


# --------------------------------------------------------------------------------------------
# CLI helpers.
# --------------------------------------------------------------------------------------------


def decontam_self_check(gsm8k_set: set[str], rule: str, sample_question: str) -> bool:
    """Assert the decontam path works: a known GSM8K test question, treated as a candidate, is
    flagged for removal under `rule`, while an unrelated string is not. Prints and returns True."""
    injected_removed = normalize_text_for_decontam(sample_question, rule) in gsm8k_set
    unrelated_removed = (
        normalize_text_for_decontam("the capital of france is paris 42 zzz", rule) in gsm8k_set
    )
    print(f"Decontam self-check (rule={rule}): injected GSM8K question removed="
          f"{injected_removed}; unrelated string removed={unrelated_removed}")
    assert injected_removed, "Decontam self-check FAILED: known GSM8K test question not detected."
    assert not unrelated_removed, "Decontam self-check FAILED: unrelated string matched GSM8K set."
    return True


def _print_previews(dataset, kept: list[int], tokenizer, cfg: dict[str, Any], n: int) -> None:
    """Print `n` masked previews from the first kept examples (response-only masking by eye)."""
    if n <= 0 or not kept:
        return
    enable_thinking = bool(cfg.get("model", {}).get("enable_thinking", False))
    fields = _data_fields(cfg)
    qf, rf = fields["query"], fields["response"]
    print(f"\n--- {min(n, len(kept))} masked example(s) "
          f"(verify only the response + end-of-turn is trained) ---")
    for idx, src in enumerate(kept[:n]):
        formatted = build_chat_formatted_example(
            (dataset[src].get(qf) or "").strip(),
            (dataset[src].get(rf) or "").strip(),
            tokenizer,
            enable_thinking=enable_thinking,
        )
        print(f"\n[example {idx} — source index {src}]")
        print(masked_preview(formatted, tokenizer))


# --------------------------------------------------------------------------------------------
# CLI: freeze the selection.
# --------------------------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser("Freeze the MetaMathQA kept-indices selection for Task 1.")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing splits/ selection file (changes the training set).",
    )
    parser.add_argument(
        "--preview", type=int, default=2,
        help="Print this many masked examples (0 to skip).",
    )
    parser.add_argument(
        "--preview-only", action="store_true",
        help="Load the existing split file and print masked previews without rewriting anything.",
    )
    args = parser.parse_args()

    import transformers
    print(f"transformers {transformers.__version__}")

    cfg = load_yaml(args.config)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    # --preview-only: read the frozen split and show previews; change nothing on disk.
    if args.preview_only:
        existing = load_kept_indices()
        dataset = load_metamathqa(cfg)
        print(f"Loaded split: {len(existing['kept_indices'])} kept indices, "
              f"rule={existing.get('normalization_rule')}")
        _print_previews(dataset, existing["kept_indices"], tokenizer, cfg, args.preview or 2)
        return

    seed_all(SEED)  # selection is deterministic (source order); seed set for reproducibility.

    out_path = os.fspath(KEPT_INDICES_PATH)
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(
            f"{out_path} already exists; selection is write-once. Re-run with --force to "
            f"overwrite (only if you intend to change the frozen training set)."
        )

    data_cfg = cfg.get("data", {})
    rule = data_cfg.get("normalization", DEFAULT_NORMALIZATION_RULE)
    max_len = int(data_cfg.get("max_seq_len", 2048))
    do_decontam = bool(data_cfg.get("gsm8k_decontaminate", True))

    dataset = load_metamathqa(cfg)

    # Build the GSM8K decontam set once, run the self-check, then reuse the set for selection.
    gsm8k_set: set[str] = set()
    if do_decontam:
        from datasets import load_dataset
        gsm = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split="test")
        gsm8k_set = {normalize_text_for_decontam(q, rule) for q in gsm["question"]}
        decontam_self_check(gsm8k_set, rule, gsm["question"][0])

    kept, counts, rule, kept_lengths = select_eligible_indices(
        cfg, tokenizer, dataset=dataset, gsm8k_set=gsm8k_set if do_decontam else None
    )
    save_kept_indices(out_path, kept, counts, cfg, rule, force=args.force)

    print(f"Wrote {len(kept)} kept indices to {out_path}")
    print(f"Normalization rule: {rule}")
    print("Filter-stage counts (over the consumed source-order prefix):")
    for key in ("total_scanned", "removed_empty", "removed_decontam", "removed_over_len",
                "eligible", "kept"):
        print(f"  {key:>16}: {counts[key]}")

    if kept_lengths:
        print(f"Kept token length: max={max(kept_lengths)}, "
              f"mean={sum(kept_lengths) / len(kept_lengths):.1f}, limit={max_len}")

    _print_previews(dataset, kept, tokenizer, cfg, args.preview)


if __name__ == "__main__":
    main()
