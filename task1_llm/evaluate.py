"""Task 1 evaluation via lm-evaluation-harness. (handout p3, Step 5; p4 table)

Zero-shot evaluation of the seven conditions over four benchmarks:
  * GSM8K (gsm8k_cot, generation, 1,024-new-token cap, greedy, non-thinking chat template,
    flexible-extraction exact match, unextractable => incorrect) -- the expensive part.
  * HellaSwag / BoolQ / ARC-Easy (candidate likelihood, stock completion prompts, NO chat
    template).
All in FP16 with dropout disabled. Reports raw + normalized accuracy where available, per-example
outputs, extracted GSM8K answers + truncation flags (saved to DRIVE_ROOT), and changes-from-
baseline in percentage points.

Two prompt paths (README "Evaluation prompt format"): only `gsm8k_cot` is wrapped in the Qwen3
chat template (enable_thinking=False -> empty <think></think>), matching training; the three
multiple-choice tasks use lm-eval's stock completion prompts with no chat template. This is driven
per-benchmark by `use_chat_template` in the config, realized as the `apply_chat_template` flag of a
separate `simple_evaluate` call per benchmark.

Conditions (p4 table) are enumerated in the eval config: a shared `baseline` (pretrained
Qwen3-0.6B, update 0) plus one entry per training run, each evaluated at the `eval_checkpoints`
declared in *its own* train config (Full FT at midpoint 625 + final 1250; everyone else at 1250).

Merge check (Step 5): before evaluating an adapter run, merge the adapter into a fresh backbone
(projected factors for OPLoRA) and verify merged vs unmerged logits agree on a small batch; eval
then runs the merged weights with no adapter path.

Backend: `hf` (default) is fully implemented. `vllm` is a config-selectable fast path for GSM8K
generation that is not wired up yet (see `_make_lm`); HF is the fallback the handout assumes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import torch

from common.checkpoint import checkpoint_path, load_checkpoint
from common.config import build_arg_parser, load_yaml, run_id_from_config
from common.paths import RESULTS_DIR, ensure_dir, task1_dir

from .adapters import merge_into_backbone
from .model import configure_method, load_base_model

log = logging.getLogger(__name__)

_FULL_METHODS = ("full_ft", "full_ft_l2sp")


# ---------------------------------------------------------------------------------------------
# Model construction for a single (run, checkpoint) condition
# ---------------------------------------------------------------------------------------------
def _prepare_for_eval(model) -> None:
    """Put a model into inference mode: no grad, caching on, grad-checkpointing off, dropout off."""
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        model.gradient_checkpointing_disable()
    except Exception:  # noqa: BLE001 — not all models expose this; harmless if absent
        pass
    model.config.use_cache = True
    model.eval()


def _load_trained_weights(model, method: str, ckpt_path: str) -> None:
    """Load a training checkpoint into `model`.

    Full-FT checkpoints hold the whole state_dict (strict load). Adapter checkpoints hold only the
    trainable adapter tensors over a backbone reloaded fresh from HF (strict=False); we assert no
    trainable tensor is silently missing, mirroring train.py's resume guard.
    """
    ck = load_checkpoint(ckpt_path)
    is_full = method in _FULL_METHODS
    if not is_full:
        trainable_names = {nm for nm, p in model.named_parameters() if p.requires_grad}
        missing = trainable_names - set(ck.weights.keys())
        if missing:
            shown = sorted(missing)
            raise RuntimeError(
                f"Eval checkpoint {ckpt_path} is missing {len(missing)} trainable parameter(s): "
                f"{shown[:10]}{' ...' if len(shown) > 10 else ''}"
            )
    model.load_state_dict(ck.weights, strict=is_full)


def _merge_check(unmerged, merged, tokenizer, device, *, batch_size: int, tol: float) -> dict:
    """Verify the merged backbone reproduces the adapter model's predictions on a small batch.

    Runs the unmerged model under FP16 autocast (the adapter factors are FP32 over an FP16 backbone,
    exactly as at training) and the already-FP16 merged model plainly, then compares logits. The
    gate is argmax (top-1) agreement — logit magnitudes differ at the FP16 noise floor, but the
    predicted token must not change. Returns stats for logging; raises if tokens disagree.
    """
    prompts = [
        "Question: If there are 3 cars and 2 more arrive, how many cars are there? Answer:",
        "The capital of France is",
        "2 + 2 =",
        "Once upon a time,",
    ][:batch_size]
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    amp = device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            lo = unmerged(**enc).logits.float()
        hi = merged(**enc).logits.float()

    mask = enc["attention_mask"].bool().unsqueeze(-1)
    diff = (lo - hi).abs()
    max_abs_diff = float(diff.masked_select(mask).max())
    tok_lo = lo.argmax(-1)
    tok_hi = hi.argmax(-1)
    valid = enc["attention_mask"].bool()
    agree = float((tok_lo.eq(tok_hi) & valid).sum()) / float(valid.sum())

    stats = {"merge_check_top1_agreement": agree, "merge_check_max_abs_logit_diff": max_abs_diff}
    if agree < 1.0:
        raise RuntimeError(
            f"Merge check failed: merged vs unmerged top-1 predictions agree on only {agree:.3%} "
            f"of tokens (max |Δlogit|={max_abs_diff:.3g}, tol={tol}). The merge is wrong."
        )
    if max_abs_diff > tol:
        log.warning("Merge check: top-1 agrees but max |Δlogit|=%.3g exceeds tol=%.3g (FP16 noise?).",
                    max_abs_diff, tol)
    return stats


def build_eval_model(train_cfg: dict[str, Any], ckpt_path: Optional[str], device, *,
                     merge_check_cfg: dict[str, Any]) -> tuple[Any, Any, dict]:
    """Build an FP16 inference model for one condition; returns (model, tokenizer, extra_stats).

    * baseline (ckpt_path None, method None/full): pretrained Qwen3-0.6B, no fine-tuning.
    * full_ft / full_ft_l2sp: load the full checkpoint, cast to FP16.
    * lora / oplora: attach adapters, load trained factors, merge into a fresh backbone (with the
      Step-5 merge check), then eval the merged FP16 weights with no adapter path.
    """
    model, tokenizer = load_base_model(train_cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen3 has no pad; eval batching needs one

    method = train_cfg.get("method")
    extra: dict = {}

    if ckpt_path is None:
        # Baseline: pristine pretrained model, no method wiring needed.
        model.to(device)
        _prepare_for_eval(model)
        return model.to(torch.float16), tokenizer, extra

    model, adapters, _ = configure_method(model, train_cfg)
    model.to(device)
    _load_trained_weights(model, method, ckpt_path)

    if adapters is not None:
        _prepare_for_eval(model)
        merged = merge_into_backbone(model, adapters).to(device)
        _prepare_for_eval(merged)
        merged = merged.to(torch.float16)
        if merge_check_cfg.get("enabled", True):
            extra.update(_merge_check(
                model, merged, tokenizer, device,
                batch_size=int(merge_check_cfg.get("batch_size", 8)),
                tol=float(merge_check_cfg.get("logit_atol", 0.5)),
            ))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return merged, tokenizer, extra

    # Full-FT: params loaded as FP32 masters; eval in FP16.
    _prepare_for_eval(model)
    return model.to(torch.float16), tokenizer, extra


# ---------------------------------------------------------------------------------------------
# lm-eval wrapper + per-benchmark run
# ---------------------------------------------------------------------------------------------
def _patch_non_thinking_chat_template(tokenizer) -> None:
    """Force enable_thinking=False on every chat-template render (empty <think></think> block).

    lm-eval renders chat prompts via tokenizer.apply_chat_template; Qwen3 defaults enable_thinking
    to True, so we intercept at the tokenizer to default it False regardless of the lm-eval version.
    Harmless for the multiple-choice tasks, which never apply the chat template.
    """
    if getattr(tokenizer, "_atml_non_thinking_patched", False):
        return
    orig = tokenizer.apply_chat_template

    def patched(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return orig(*args, **kwargs)

    tokenizer.apply_chat_template = patched
    tokenizer._atml_non_thinking_patched = True


def _make_lm(model, tokenizer, backend: str, batch_size, device):
    """Wrap the in-memory model in an lm-eval LM object for `backend` ('hf' | 'vllm')."""
    if backend == "hf":
        from lm_eval.models.huggingface import HFLM

        _patch_non_thinking_chat_template(tokenizer)
        return HFLM(
            pretrained=model, tokenizer=tokenizer,
            batch_size=batch_size, device=str(device), dtype=torch.float16,
        )
    if backend == "vllm":
        # vLLM loads from a path, not an in-memory module, so this path must first serialize the
        # merged/cast model + tokenizer to a temp dir on DRIVE_ROOT and point lm-eval's "vllm"
        # model at it (and re-apply the non-thinking chat template). Deferred: HF is the assumed
        # fallback and is correct; wire vLLM only if it measurably helps GSM8K on the L4.
        raise NotImplementedError(
            "backend: vllm is not implemented yet; use backend: hf (see evaluate.py)."
        )
    raise ValueError(f"Unknown backend {backend!r}")


def _simple_evaluate(**kwargs):
    """Import-tolerant handle on lm-eval's simple_evaluate across 0.4.x layouts."""
    try:
        from lm_eval import simple_evaluate
    except ImportError:
        from lm_eval.evaluator import simple_evaluate
    return simple_evaluate(**kwargs)


def _collect_metrics(task_result: dict[str, Any], primary_metric: str) -> dict[str, Any]:
    """Pull scalar metrics out of an lm-eval task result and pick the primary one.

    lm-eval keys scalars as "<metric>,<filter>" (e.g. "exact_match,flexible-extract",
    "acc,none", "acc_norm,none"). We keep them all, and choose the primary by metric name,
    preferring the flexible-extraction filter for GSM8K exact_match.
    """
    scalars = {
        k: v for k, v in task_result.items()
        if isinstance(v, (int, float)) and "," in k and not k.endswith("stderr")
    }
    candidates = [k for k in scalars if k.split(",")[0] == primary_metric]
    if primary_metric == "exact_match":
        flexible = [k for k in candidates if "flexible" in k]
        chosen = (flexible or candidates or [None])[0]
    else:
        chosen = (candidates or [None])[0]
    return {
        "primary_metric": primary_metric,
        "primary_filter": chosen,
        "primary_value": scalars.get(chosen) if chosen else None,
        "metrics": scalars,
    }


def run_benchmark(lm, name: str, bench_cfg: dict[str, Any], *, num_fewshot: int,
                  limit: Optional[int], save_samples: bool) -> tuple[dict[str, Any], list]:
    """Run one benchmark through lm-eval; return (aggregate_metrics, per_example_samples).

    gsm8k is a `generation` task (greedy, capped new tokens, chat template on); the three MC tasks
    are `likelihood` (chat template off). Each benchmark is its own simple_evaluate call so the
    chat-template flag is set per the two prompt paths.
    """
    task = bench_cfg["lm_eval_task"]
    use_chat = bool(bench_cfg.get("use_chat_template", False))
    kwargs: dict[str, Any] = dict(
        model=lm,
        tasks=[task],
        num_fewshot=num_fewshot,
        limit=limit,
        apply_chat_template=use_chat,
        log_samples=save_samples,
        bootstrap_iters=0,          # we don't need bootstrapped stderr; skip for speed
    )
    if bench_cfg.get("method") == "generation":
        max_new = int(bench_cfg.get("max_new_tokens", 1024))
        greedy = bool(bench_cfg.get("greedy", True))
        # lm-eval parses gen_kwargs from a "k=v,k=v" string; greedy => do_sample=False.
        kwargs["gen_kwargs"] = f"max_gen_toks={max_new},do_sample={'False' if greedy else 'True'}"

    out = _simple_evaluate(**kwargs)
    task_result = out["results"][task]
    agg = _collect_metrics(task_result, bench_cfg.get("primary_metric", "acc"))
    agg["task"] = task
    agg["n_examples"] = int(out.get("n-samples", {}).get(task, {}).get("effective", limit or -1)) \
        if isinstance(out.get("n-samples"), dict) else (limit or -1)
    samples = out.get("samples", {}).get(task, []) if save_samples else []
    return agg, samples


# ---------------------------------------------------------------------------------------------
# One condition across all benchmarks
# ---------------------------------------------------------------------------------------------
def evaluate_condition(eval_cfg: dict[str, Any], label: str, train_cfg: dict[str, Any],
                       update: int, ckpt_path: Optional[str], device) -> dict[str, Any]:
    """Evaluate one (run, checkpoint) condition across all four benchmarks; write results.

    Aggregate metrics -> repo results/task1/eval/<label>.json (small, committed). Per-example
    samples (raw generations, extracted GSM8K answers, truncation info) -> DRIVE_ROOT/task1/eval/
    <label>/<task>.jsonl (large, on Drive).
    """
    backend = eval_cfg.get("backend", "hf")
    num_fewshot = int(eval_cfg.get("num_fewshot", 0))
    batch_size = eval_cfg.get("batch_size", "auto")
    limit = eval_cfg.get("limit")
    save_samples = bool(eval_cfg.get("report", {}).get("save_per_example", True))
    benchmarks = eval_cfg["benchmarks"]

    log.info("Evaluating condition %s (update %d, ckpt=%s).", label, update, ckpt_path or "-")
    model, tokenizer, extra = build_eval_model(
        train_cfg, ckpt_path, device, merge_check_cfg=eval_cfg.get("merge_check", {}))
    lm = _make_lm(model, tokenizer, backend, batch_size, device)

    samples_dir = ensure_dir(task1_dir("eval") / label)
    results: dict[str, Any] = {
        "label": label,
        "run_id": train_cfg.get("run_id"),
        "method": train_cfg.get("method"),
        "update": update,
        "backend": backend,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "benchmarks": {},
    }
    results.update(extra)  # merge-check stats for adapter runs

    for name, bench_cfg in benchmarks.items():
        agg, samples = run_benchmark(
            lm, name, bench_cfg, num_fewshot=num_fewshot, limit=limit, save_samples=save_samples)
        results["benchmarks"][name] = agg
        if save_samples and samples:
            with open(samples_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, default=str) + "\n")
        pv = agg["primary_value"]
        log.info("  %-10s %s=%s", name, agg["primary_metric"],
                 f"{pv:.4f}" if isinstance(pv, (int, float)) else pv)

    out_dir = ensure_dir(RESULTS_DIR / "task1" / "eval")
    with open(out_dir / f"{label}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Free the model before the next (heavy) condition is built.
    del model, lm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------------------------
# Condition enumeration + baseline deltas
# ---------------------------------------------------------------------------------------------
def _resolve_conditions(eval_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the eval config's `baseline` + `conditions` into a flat list of eval jobs.

    Each job: {label, config (train cfg dict), update, ckpt_path or None, is_baseline}. A run's
    checkpoints come from its own train config's `eval_checkpoints`, so the schedule lives in one
    place. Checkpoint paths are DRIVE_ROOT/task1/checkpoints/<run_id>/ckpt_step<update>.pt.

    Condition `config` paths are given relative to the invocation directory (repo root), e.g.
    `configs/task1/full_ft.yaml` — the same paths the train commands use.
    """
    def _load_train_cfg(rel: str) -> dict[str, Any]:
        cfg = load_yaml(rel)
        cfg.setdefault("run_id", run_id_from_config(rel, cfg))
        return cfg

    jobs: list[dict[str, Any]] = []

    baseline = eval_cfg.get("baseline", {})
    if baseline.get("enabled", True):
        bcfg = _load_train_cfg(baseline["config"])
        jobs.append({"label": baseline.get("label", "baseline"), "config": bcfg,
                     "update": 0, "ckpt_path": None, "is_baseline": True})

    for cond in eval_cfg.get("conditions", []):
        tcfg = _load_train_cfg(cond["config"])
        run_id = tcfg["run_id"]
        updates = cond.get("eval_checkpoints") or tcfg.get("eval_checkpoints", [])
        ckpt_dir = task1_dir("checkpoints") / run_id
        for upd in updates:
            jobs.append({
                "label": f"{run_id}_u{int(upd)}",
                "config": tcfg,
                "update": int(upd),
                "ckpt_path": checkpoint_path(str(ckpt_dir), int(upd)),
                "is_baseline": False,
            })
    return jobs


def _write_summary(all_results: list[dict[str, Any]], baseline_label: str) -> None:
    """Write a cross-condition summary with primary metrics and pp-deltas vs the baseline."""
    baseline = next((r for r in all_results if r["label"] == baseline_label), None)
    base_primary = {}
    if baseline is not None:
        base_primary = {name: b.get("primary_value")
                        for name, b in baseline["benchmarks"].items()}

    rows = []
    for r in all_results:
        row = {"label": r["label"], "update": r["update"], "method": r["method"]}
        for name, b in r["benchmarks"].items():
            pv = b.get("primary_value")
            row[f"{name}"] = pv
            base = base_primary.get(name)
            if isinstance(pv, (int, float)) and isinstance(base, (int, float)):
                row[f"{name}_pp_vs_baseline"] = round((pv - base) * 100.0, 3)
        rows.append(row)

    out_dir = ensure_dir(RESULTS_DIR / "task1" / "eval")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline_label, "conditions": rows}, f, indent=2)


def main() -> None:
    """CLI: evaluate every condition enumerated in the eval config, then write the summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser("Task 1 evaluation (lm-eval; seven conditions).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root

    eval_cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jobs = _resolve_conditions(eval_cfg)
    baseline_label = eval_cfg.get("baseline", {}).get("label", "baseline")
    print(f"[eval] {len(jobs)} condition(s) on {device}; backend={eval_cfg.get('backend', 'hf')}.",
          flush=True)

    all_results = []
    for job in jobs:
        res = evaluate_condition(
            eval_cfg, job["label"], job["config"], job["update"], job["ckpt_path"], device)
        all_results.append(res)
        print(f"[eval] done {job['label']}.", flush=True)

    _write_summary(all_results, baseline_label)
    print(f"[eval] wrote {len(all_results)} condition file(s) + summary.json under "
          f"results/task1/eval/.", flush=True)


if __name__ == "__main__":
    main()
