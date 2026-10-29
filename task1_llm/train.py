"""Task 1 training loop: one-epoch SFT, resumable, config-driven. (handout p3-p4)

Usage (on Colab):
    python -m task1_llm.train --config configs/task1/full_ft.yaml
    python -m task1_llm.train --config configs/task1/full_ft.yaml --resume

Implements the shared recipe for all five methods: microbatch 2 x grad-accum 16 = effective
batch 32, 1,250 updates, AdamW + cosine (38 warmup), global-norm 1.0 clip, FP16 AMP with dynamic
scaling, FP32 master/adapter params. Saves recovery checkpoints at updates 312/625/937/1250
(preserving optimizer, scheduler, scaler, RNG, data position). Logs SFT loss + LR per update, and
resource stats. For L2-SP additionally logs Omega / lambda*Omega / total loss; for Full FT/L2-SP
records full-model displacement at midpoint/final.

Loss normalization (p3): the response-only cross-entropy is normalized over ALL valid response
tokens in the *effective* batch, not per-microbatch. We prefetch the 16 microbatches of an update,
count the total response tokens (the shifted, non-ignored labels — exactly HF's cross-entropy
denominator), then scale each microbatch's token-SUM loss (mean_loss * n_tokens) by 1/total before
backward. The accumulated gradient is therefore exactly the gradient of
(sum of per-token losses) / (total response tokens), with no post-hoc gradient surgery. The L2-SP
penalty is added once per update, after the data terms, so it is not diluted by the normalization.

Step counting (see README "Step counting"): the 1,250 updates and the 312/625/937/1250 milestones
count ATTEMPTED optimizer steps (one per 32-example accumulation window); a dynamic-scaler skip
still counts and still advances the data cursor. Only the LR scheduler advances on applied steps
(common.precision.optimizer_step).
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Allocator config MUST be set before torch initializes CUDA. expandable_segments lets the caching
# allocator grow/shrink segments instead of reserving fixed blocks, which cuts the fragmentation
# that made Full-FT+L2-SP OOM right after a checkpoint save (the failing 1.24 GiB alloc at update
# 11). setdefault so an explicit override in the environment still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from common import SEED
from common.checkpoint import (
    CheckpointState,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from common.config import build_arg_parser, load_yaml, run_id_from_config
from common.logging_utils import JsonlLogger
from common.paths import ensure_dir, task1_dir
from common.precision import PrecisionConfig, autocast_context, make_grad_scaler, optimizer_step
from common.resources import Timer, count_params, file_size_bytes, peak_gpu_bytes
from common.seeding import capture_rng_state, restore_rng_state, seed_all

from .adapters import export_initial_factors
from .data import IGNORE_INDEX, iterate_microbatches, load_kept_indices, load_metamathqa
from .l2sp import build_l2sp
from .model import (
    build_optimizer_and_scheduler,
    build_param_groups,
    configure_method,
    full_model_displacement,
    load_base_model,
    snapshot_params,
)

log = logging.getLogger(__name__)

_FULL_METHODS = ("full_ft", "full_ft_l2sp")

# Default row-chunk for the response-only lm_head: logits are [rows, vocab] (vocab ~152k), so we
# cap the transient logits tensor by projecting at most this many response positions at a time.
# Response tokens per microbatch are usually below this, so it is typically a single chunk.
_DEFAULT_LOGITS_CHUNK_ROWS = 512

# -----------------------------------------------------------------------------------------------
# Response-only loss (OOM fix; see README "Memory (OOM avoidance on the L4)").
#
# HF's model(..., labels=...) materializes full [B, T, vocab] logits (2 x 2048 x 151936 fp16 =
# 1.24 GiB, plus a same-shape fp32 upcast inside the cross-entropy) for EVERY position, then masks
# all but the response tokens. On the L4 that transient is what OOMs Full-FT+L2-SP. Instead we run
# the decoder to final hidden states, select only the positions whose shifted label != -100, and
# apply lm_head to just those rows (chunked). The math is unchanged: token-SUM cross-entropy in
# FP32 over exactly the same response tokens (verified against the HF loss by _verify_response_loss
# to ~1e-4). Normalization over the effective batch still happens in the caller (scale by
# 1/total_tokens before backward), exactly as before.
# -----------------------------------------------------------------------------------------------


def _decoder_and_head(model):
    """Return (decoder, lm_head): the backbone that yields final hidden states and the output head.

    Uses the standard HF accessors so this stays correct across Causal-LM architectures: the loss
    path is `lm_head(decoder(...).last_hidden_state)`, the same two modules model(..., labels=...)
    composes internally.
    """
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    head = model.get_output_embeddings()
    if head is None:
        head = model.lm_head
    return decoder, head


def _select_response_rows(decoder, input_ids, attn, labels, *, pcfg, device, amp_enabled):
    """Run the decoder and return (sel_hidden, sel_labels) for response positions only.

    Mirrors HF's shift: the hidden state at position t predicts the token at t+1, so we keep hidden
    positions [:-1] whose shifted label (labels[:, 1:]) is not the ignore index. sel_hidden stays
    attached to the backbone graph (grad flows) unless the caller is under no_grad.
    """
    with autocast_context(pcfg, device_type=device.type, enabled=amp_enabled):
        hidden = decoder(input_ids=input_ids, attention_mask=attn)[0]  # [B, T, H]
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    sel_hidden = hidden[:, :-1, :][mask]  # [N, H]
    sel_labels = shift_labels[mask]       # [N]
    return sel_hidden, sel_labels


def _backward_response_loss(decoder, head, batch, total_tokens, *, scaler, pcfg, device,
                            amp_enabled, chunk_rows):
    """Backward the response-only token-SUM loss for one microbatch; return its float token-SUM.

    Two-stage to bound peak memory to a single logits chunk: (1) detach the selected hidden states
    to a leaf and backward each chunk's lm_head->CE graph into leaf.grad (freeing the [rows, vocab]
    logits after each chunk); (2) push the accumulated leaf.grad through the backbone exactly once.
    lm_head's own gradient (and, under weight tying, the shared embedding's) accumulates from the
    chunk backwards, so the result equals a single full backward. The chunk losses are scaled by
    1/total_tokens (effective-batch normalization) and by the AMP scale, identically to before.
    """
    input_ids = batch.input_ids.to(device)
    attn = batch.attention_mask.to(device)
    labels = batch.labels.to(device)

    sel_hidden, sel_labels = _select_response_rows(
        decoder, input_ids, attn, labels, pcfg=pcfg, device=device, amp_enabled=amp_enabled
    )
    n = int(sel_labels.numel())
    if n == 0:
        return 0.0

    leaf = sel_hidden.detach().requires_grad_(True)
    loss_sum = 0.0
    for s in range(0, n, chunk_rows):
        e = min(s + chunk_rows, n)
        with autocast_context(pcfg, device_type=device.type, enabled=amp_enabled):
            logits = head(leaf[s:e])  # [rows, vocab], fp16 under autocast
        # FP32 token-SUM CE over this chunk — HF upcasts logits to fp32 before the softmax too.
        chunk = F.cross_entropy(logits.float(), sel_labels[s:e], reduction="sum")
        scaler.scale(chunk / total_tokens).backward()
        loss_sum += float(chunk.detach())
    # leaf.grad now holds the summed (scaled) hidden gradient; backprop it through the backbone once.
    sel_hidden.backward(leaf.grad)
    return loss_sum


@torch.no_grad()
def _response_loss_sum_value(decoder, head, input_ids, attn, labels, *, pcfg, device,
                             amp_enabled, chunk_rows):
    """Value-only (no backward) response-only token-SUM CE; returns (loss_sum, n_tokens)."""
    sel_hidden, sel_labels = _select_response_rows(
        decoder, input_ids, attn, labels, pcfg=pcfg, device=device, amp_enabled=amp_enabled
    )
    n = int(sel_labels.numel())
    loss_sum = 0.0
    for s in range(0, n, chunk_rows):
        e = min(s + chunk_rows, n)
        with autocast_context(pcfg, device_type=device.type, enabled=amp_enabled):
            logits = head(sel_hidden[s:e])
        loss_sum += float(F.cross_entropy(logits.float(), sel_labels[s:e], reduction="sum"))
    return loss_sum, n


@torch.no_grad()
def _verify_response_loss(model, decoder, head, batch, *, pcfg, device, amp_enabled, chunk_rows,
                          tol=1e-4):
    """Assert the response-only loss equals HF's model(..., labels=...) loss on one microbatch.

    Compares the per-token MEAN loss (O(1), so `tol` is a plain absolute tolerance). Runs the full
    HF path once (the only place the full [B,T,vocab] logits are ever built); done at the first
    update where GPU headroom is highest. Raises if they disagree beyond `tol`.
    """
    input_ids = batch.input_ids.to(device)
    attn = batch.attention_mask.to(device)
    labels = batch.labels.to(device)
    n_i = int((labels[:, 1:] != IGNORE_INDEX).sum())
    if n_i == 0:
        return None

    with autocast_context(pcfg, device_type=device.type, enabled=amp_enabled):
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    old_mean = float(out.loss)  # HF mean over non-ignored response tokens
    new_sum, n_new = _response_loss_sum_value(
        decoder, head, input_ids, attn, labels,
        pcfg=pcfg, device=device, amp_enabled=amp_enabled, chunk_rows=chunk_rows,
    )
    new_mean = new_sum / n_new if n_new else 0.0
    diff = abs(old_mean - new_mean)
    result = {"type": "loss_equivalence_check", "hf_mean_loss": old_mean,
              "response_only_mean_loss": new_mean, "abs_diff": diff,
              "n_response_tokens": n_i, "tol": tol, "ok": diff <= tol}
    if not result["ok"]:
        raise RuntimeError(
            f"Response-only loss disagrees with HF loss: hf={old_mean:.6f} "
            f"new={new_mean:.6f} abs_diff={diff:.3e} > tol={tol:.3e}"
        )
    return result


def train(cfg: dict[str, Any], *, resume: bool = False, stop_after: int | None = None) -> None:
    """Run (or resume) one SFT run defined by `cfg`.

    stop_after: if set, exit cleanly after this many optimizer updates (after any checkpoint that
    lands at that update), to simulate a Colab disconnect for resume testing.
    """
    run_id = cfg.get("run_id") or "task1_run"
    method = cfg["method"]
    is_full = method in _FULL_METHODS
    is_smoke = "smoke" in run_id.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_all(SEED)

    # --- model + method ------------------------------------------------------------------
    model, tokenizer = load_base_model(cfg)
    model, adapters, _ = configure_method(model, cfg)
    model.to(device)

    # theta0 for displacement (pretrained init, captured BEFORE any resume-weight load so it is
    # the same constant every run); L2-SP reference likewise snapshots the pretrained init.
    record_disp = bool(cfg.get("logging", {}).get("record_displacement", False))
    theta0 = snapshot_params(model) if record_disp else None
    l2sp = build_l2sp(model, cfg)

    param_groups = build_param_groups(model, cfg)
    optimizer, scheduler = build_optimizer_and_scheduler(param_groups, cfg)

    prec = cfg["precision"]
    pcfg = PrecisionConfig(
        compute_dtype=prec.get("compute_dtype", "float16"),
        fp32_master=bool(prec.get("fp32_master", False)),
        grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
    )
    amp_enabled = device.type == "cuda"
    scaler = make_grad_scaler(enabled=amp_enabled, device_type=device.type)

    micro = int(cfg["train"]["microbatch_size"])
    accum = int(cfg["train"]["grad_accum_steps"])
    total_updates = int(cfg["train"]["total_updates"])
    save_at = {int(u) for u in cfg.get("checkpoint", {}).get("save_at_updates", [])}
    eval_ckpts = {int(u) for u in cfg.get("eval_checkpoints", [])}

    ckpt_dir = ensure_dir(task1_dir("checkpoints") / run_id)
    # Per-run logs live on Drive beside the checkpoints so they survive a Colab disconnect; on
    # --resume JsonlLogger appends to the existing train.jsonl. Small final/aggregate results are
    # committed to the repo's results/ separately.
    results = ensure_dir(task1_dir("results") / run_id)

    # --- resume --------------------------------------------------------------------------
    start_update, start_cursor = 0, 0
    resumed_from: str | None = None
    if resume:
        latest = latest_checkpoint(str(ckpt_dir))
        if latest is None:
            log.warning("--resume set but no checkpoint in %s; starting fresh.", ckpt_dir)
        else:
            resumed_from = latest
            ck = load_checkpoint(latest)
            # Fail loudly if any trainable parameter is absent from the checkpoint. For adapter
            # runs we load with strict=False (the checkpoint deliberately omits the frozen
            # backbone), so without this check a missing adapter tensor would silently stay at its
            # fresh init instead of resuming.
            trainable_names = {nm for nm, p in model.named_parameters() if p.requires_grad}
            missing = trainable_names - set(ck.weights.keys())
            if missing:
                shown = sorted(missing)
                raise RuntimeError(
                    f"Resume checkpoint {latest} is missing {len(missing)} trainable "
                    f"parameter(s): {shown[:10]}{' ...' if len(shown) > 10 else ''}"
                )
            model.load_state_dict(ck.weights, strict=is_full)
            model.to(device)
            optimizer.load_state_dict(ck.optimizer)
            scheduler.load_state_dict(ck.scheduler)
            scaler.load_state_dict(ck.scaler)
            restore_rng_state(ck.rng)
            start_update = int(ck.step)
            start_cursor = int(ck.data_position.get("cursor", 0))
            log.info("Resumed %s at update %d (data cursor %d).", run_id, start_update, start_cursor)

    # Announce start state on stdout (Colab) so a resume is unambiguous.
    if resumed_from is not None:
        print(f"[train] RESUMING {run_id} from {resumed_from} at update "
              f"{start_update}/{total_updates} (data cursor {start_cursor}).", flush=True)
    elif resume:
        print(f"[train] --resume set but no checkpoint found; STARTING FRESH {run_id}: "
              f"0/{total_updates} updates.", flush=True)
    else:
        print(f"[train] STARTING FRESH {run_id}: 0/{total_updates} updates.", flush=True)

    # If a resumed checkpoint is already at (or past) the target, there is nothing to do.
    if start_update >= total_updates:
        print(f"[train] {run_id} already complete ({start_update}/{total_updates} updates); "
              f"nothing to do.", flush=True)
        return

    # Export pristine adapter factors on a fresh adapter start so OPLoRA-16 can match LoRA-16.
    if adapters is not None and start_update == 0:
        export_initial_factors(
            adapters, str(task1_dir("checkpoints") / f"{run_id}_initial_factors.pt")
        )

    trainable = [p for p in model.parameters() if p.requires_grad]

    # Response-only loss plumbing (see helpers above): the decoder/head we drive directly instead
    # of model(..., labels=...), and the row-chunk for the lm_head projection.
    decoder, head = _decoder_and_head(model)
    chunk_rows = int(cfg.get("train", {}).get("logits_chunk_rows", _DEFAULT_LOGITS_CHUNK_ROWS))
    verify_loss = bool(cfg.get("logging", {}).get("verify_response_loss", True))

    # --- data ----------------------------------------------------------------------------
    try:
        kept = load_kept_indices()["kept_indices"]
    except FileNotFoundError:
        raise SystemExit(
            "No frozen kept indices found. Run `python -m task1_llm.data --config "
            f"{run_id}` first to write splits/task1/metamathqa_kept_indices.json."
        )
    dataset = load_metamathqa(cfg)
    n = len(kept)

    def microbatch_stream(start: int):
        """Yield (batch, cursor_after) cycling the frozen order; cursor is the within-epoch index.

        Real runs are exactly one epoch (no wrap); the cycling only matters for the tiny smoke
        run, and is still exactly resumable because the data order is deterministic and dropout
        RNG is checkpointed.
        """
        pos = start
        while True:
            for batch in iterate_microbatches(cfg, tokenizer, kept, start_position=pos,
                                              dataset=dataset):
                pos += micro
                yield batch, pos % n
            pos = 0

    stream = microbatch_stream(start_cursor)
    cursor = start_cursor
    update = start_update

    logger = JsonlLogger(str(results / "train.jsonl"), append=resume)

    def save_recovery(upd: int, disp: float | None = None) -> str:
        # Full FT keeps the whole model; adapter runs store only trainable tensors (the frozen
        # backbone is reloaded from HF on resume), which is what keeps adapter checkpoints small.
        if is_full:
            weights = model.state_dict()
        else:
            weights = {nm: p.detach() for nm, p in model.named_parameters() if p.requires_grad}
        extra = {"run_id": run_id, "method": method}
        if disp is not None:
            extra["displacement"] = disp
        state = CheckpointState(
            weights=weights,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            rng=capture_rng_state(),
            data_position={"cursor": cursor},
            step=upd,
            extra=extra,
        )
        path = checkpoint_path(str(ckpt_dir), upd)
        # save_checkpoint reads the file back and raises if the write is corrupt, so we only reach
        # the prune below once this checkpoint is known good. Keep this update + every eval
        # checkpoint (evaluate.py needs e.g. Full-FT's midpoint 625 and final 1250); drop any
        # superseded pure-recovery checkpoint to save Drive space.
        save_checkpoint(path, state)
        prune_checkpoints(str(ckpt_dir), keep_steps={upd} | eval_ckpts)
        return path

    # --- training loop -------------------------------------------------------------------
    # Always snapshot peak GPU at update 11 (the step Full-FT+L2-SP used to OOM on the L4); the
    # smoke run also samples early updates. Peak is also logged after every checkpoint save below.
    peak_log_updates = {1, 2, 5, 11} if is_smoke else {11}
    verified = not verify_loss  # run the one-time loss-equivalence check at most once
    stopped_early = False
    with Timer() as timer:
        while update < total_updates:
            optimizer.zero_grad(set_to_none=True)

            # Prefetch the whole effective batch so the TOTAL response-token count is known before
            # any backward; response tokens are the shifted, non-ignored labels (HF's exact loss
            # denominator). cursor tracks the within-epoch data position for the checkpoint.
            micro_batches = []
            for _ in range(accum):
                batch, cursor = next(stream)
                micro_batches.append(batch)
            token_counts = [int((b.labels[:, 1:] != -100).sum()) for b in micro_batches]
            total_tokens = sum(token_counts)

            if total_tokens == 0:
                update += 1
                continue

            # One-time check (first real update, when GPU headroom is highest) that the memory-lean
            # response-only loss equals HF's model(..., labels=...) loss to ~1e-4. This is the ONLY
            # place the full [B,T,vocab] logits are ever built.
            if not verified:
                first = next((b for b, c in zip(micro_batches, token_counts) if c > 0), None)
                if first is not None:
                    chk = _verify_response_loss(
                        model, decoder, head, first, pcfg=pcfg, device=device,
                        amp_enabled=amp_enabled, chunk_rows=chunk_rows,
                    )
                    if chk is not None:
                        logger.log(chk)
                        print(f"[train] loss-equivalence check OK: hf={chk['hf_mean_loss']:.6f} "
                              f"response_only={chk['response_only_mean_loss']:.6f} "
                              f"abs_diff={chk['abs_diff']:.2e}", flush=True)
                        verified = True

            # Response-only token-SUM loss per microbatch; scaled by 1/total_tokens so the
            # accumulated gradient is exactly grad of (sum per-token losses)/(eff-batch response
            # tokens) — the same normalization and math as the old full-logits path.
            sft_sum = 0.0
            for batch, n_i in zip(micro_batches, token_counts):
                if n_i == 0:
                    continue
                sft_sum += _backward_response_loss(
                    decoder, head, batch, total_tokens, scaler=scaler, pcfg=pcfg,
                    device=device, amp_enabled=amp_enabled, chunk_rows=chunk_rows,
                )

            # L2-SP once per update, after the data terms: add scale*lambda*(theta-theta0) straight
            # into each grad (no autograd graph, theta0 streamed from pinned CPU). Undiluted by the
            # 1/total_tokens normalization above.
            l2sp_terms = None
            if l2sp is not None:
                l2sp_terms = l2sp.accumulate_grad(model, scale=scaler.get_scale())

            info = optimizer_step(optimizer, scaler, trainable, pcfg,
                                  scheduler=scheduler, step=update + 1)
            update += 1

            eff_loss = sft_sum / total_tokens
            record = {
                "update": update,
                "sft_loss": eff_loss,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": info["grad_norm"],
                "skipped": info["skipped"],
                "scale": info["scale"],
                "response_tokens": total_tokens,
            }
            if l2sp_terms is not None:
                record["omega"] = l2sp_terms["omega"]
                record["lambda_omega"] = l2sp_terms["lambda_omega"]
                record["total_loss"] = eff_loss + l2sp_terms["lambda_omega"]
                # Per-update host->device cost of streaming theta0 (reference-transfer overhead, p4).
                record["l2sp_transfer_seconds"] = l2sp_terms["transfer_seconds"]
            logger.log(record)
            print(f"[train] {run_id} {update}/{total_updates} loss={eff_loss:.4f} "
                  f"lr={record['lr']:.2e} scale={info['scale']:g} skipped={info['skipped']}",
                  flush=True)

            if update in peak_log_updates:
                logger.log({"type": "peak_gpu", "update": update,
                            "peak_gpu_bytes": peak_gpu_bytes()})

            disp = None
            if theta0 is not None and update in eval_ckpts:
                disp = full_model_displacement(model, theta0)
                logger.log({"update": update, "displacement": disp})

            if update in save_at:
                save_recovery(update, disp)
                # Log peak GPU right after every checkpoint save (save + any CPU offload can shift
                # the high-water mark; this is where the old OOM surfaced).
                logger.log({"type": "peak_gpu", "update": update, "after": "checkpoint_save",
                            "peak_gpu_bytes": peak_gpu_bytes()})

            # Simulated-disconnect stop: break after finishing this update (and its checkpoint).
            if stop_after is not None and update >= stop_after:
                stopped_early = True
                print(f"[train] --stop-after {stop_after} reached; stopping {run_id} at update "
                      f"{update}/{total_updates}. Re-run with --resume to continue.", flush=True)
                break

    # A simulated-disconnect stop leaves the run unfinished: don't force a final checkpoint and
    # don't write the completion-implying resource summary. The stop was already announced above.
    if stopped_early:
        logger.close()
        log.info("Stopped %s early at update %d/%d (%.1fs).",
                 run_id, update, total_updates, timer.wall_seconds)
        return

    # Ensure a final checkpoint exists even if total_updates wasn't in save_at_updates.
    if total_updates not in save_at:
        save_recovery(update)
        logger.log({"type": "peak_gpu", "update": update, "after": "checkpoint_save",
                    "peak_gpu_bytes": peak_gpu_bytes()})

    # --- resource summary ----------------------------------------------------------------
    trainable_params, total_params = count_params(model)
    final_ckpt = latest_checkpoint(str(ckpt_dir))
    summary = {
        "type": "resources",
        "run_id": run_id,
        "method": method,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "wall_seconds": timer.wall_seconds,
        "peak_gpu_bytes": timer.peak_gpu_bytes,
        "final_checkpoint_bytes": file_size_bytes(final_ckpt) if final_ckpt else 0,
    }
    if l2sp is not None:
        summary["l2sp_reference_bytes"] = l2sp.reference_overhead_bytes()
    logger.log(summary)
    logger.close()
    log.info("Finished %s: %d updates, %.1fs, peak GPU %d bytes.",
             run_id, update, timer.wall_seconds, timer.peak_gpu_bytes)
    print(f"[train] FINISHED {run_id}: {update}/{total_updates} updates, "
          f"{timer.wall_seconds:.1f}s, peak GPU {timer.peak_gpu_bytes} bytes.", flush=True)


def main() -> None:
    """CLI entry point: parse --config/--resume/--stop-after, load YAML, call train."""
    parser = build_arg_parser("Task 1 SFT training (resumable).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    cfg.setdefault("run_id", run_id_from_config(args.config, cfg))
    train(cfg, resume=args.resume, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
