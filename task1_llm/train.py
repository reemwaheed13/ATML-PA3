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

import torch

from common import SEED
from common.checkpoint import (
    CheckpointState,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from common.config import build_arg_parser, load_yaml, run_id_from_config
from common.logging_utils import JsonlLogger
from common.paths import RESULTS_DIR, ensure_dir, task1_dir
from common.precision import PrecisionConfig, autocast_context, make_grad_scaler, optimizer_step
from common.resources import Timer, count_params, file_size_bytes, peak_gpu_bytes
from common.seeding import capture_rng_state, restore_rng_state, seed_all

from .adapters import export_initial_factors
from .data import iterate_microbatches, load_kept_indices, load_metamathqa
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


def train(cfg: dict[str, Any], *, resume: bool = False) -> None:
    """Run (or resume) one SFT run defined by `cfg`."""
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
    results = ensure_dir(RESULTS_DIR / "task1" / run_id)

    # --- resume --------------------------------------------------------------------------
    start_update, start_cursor = 0, 0
    if resume:
        latest = latest_checkpoint(str(ckpt_dir))
        if latest is None:
            log.warning("--resume set but no checkpoint in %s; starting fresh.", ckpt_dir)
        else:
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

    # Export pristine adapter factors on a fresh adapter start so OPLoRA-16 can match LoRA-16.
    if adapters is not None and start_update == 0:
        export_initial_factors(
            adapters, str(task1_dir("checkpoints") / f"{run_id}_initial_factors.pt")
        )

    trainable = [p for p in model.parameters() if p.requires_grad]

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
        save_checkpoint(path, state)
        return path

    # --- training loop -------------------------------------------------------------------
    peak_log_updates = {1, 2, 5} if is_smoke else set()
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

            sft_sum = 0.0
            for batch, n_i in zip(micro_batches, token_counts):
                if n_i == 0:
                    continue
                input_ids = batch.input_ids.to(device)
                attn = batch.attention_mask.to(device)
                labels = batch.labels.to(device)

                with autocast_context(pcfg, device_type=device.type, enabled=amp_enabled):
                    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)

                # out.loss is the per-token mean; (mean * n_i) is this microbatch's token-SUM.
                # Scaling by 1/total_tokens up front makes the accumulated gradient exactly the
                # gradient of (sum of per-token losses) / (effective-batch response tokens).
                token_sum_loss = out.loss * n_i
                scaler.scale(token_sum_loss / total_tokens).backward()
                sft_sum += float(token_sum_loss.detach())

            # L2-SP once per update, after the data terms (added undiluted by the normalization).
            if l2sp is not None:
                scaler.scale(l2sp.lam * l2sp.penalty(model)).backward()

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
            if l2sp is not None:
                terms = l2sp.terms(model)
                record["omega"] = terms["omega"]
                record["lambda_omega"] = terms["lambda_omega"]
                record["total_loss"] = eff_loss + terms["lambda_omega"]
            logger.log(record)

            if update in peak_log_updates:
                logger.log({"type": "peak_gpu", "update": update,
                            "peak_gpu_bytes": peak_gpu_bytes()})

            disp = None
            if theta0 is not None and update in eval_ckpts:
                disp = full_model_displacement(model, theta0)
                logger.log({"update": update, "displacement": disp})

            if update in save_at:
                save_recovery(update, disp)

    # Ensure a final checkpoint exists even if total_updates wasn't in save_at_updates.
    if total_updates not in save_at:
        save_recovery(update)

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


def main() -> None:
    """CLI entry point: parse --config/--resume, load YAML, call train."""
    parser = build_arg_parser("Task 1 SFT training (resumable).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    cfg.setdefault("run_id", run_id_from_config(args.config, cfg))
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
