"""Task 2 three-stage curriculum training, resumable. (handout p8-p9)

Usage:
    python -m task2_vlm.train --config configs/task2/vlm_stages.yaml
    python -m task2_vlm.train --config configs/task2/vlm_stages.yaml --resume

One model carried through Stage 1 (caption alignment, connector only, 2,250 updates), Stage 2
(visual instruction tuning, connector + LoRA, 1,500 updates), Stage 3 (natural VQA, 750 updates):
4,500 updates / 72,000 presentations total. Per-stage 5% warmup then cosine decay to 10% of peak
LR; microbatch 8 x accum 2; global-norm 1.0; FP16 AMP + dynamic scaling; FP32 connector/adapters.
Example-weighted causal CE. Saves resumable checkpoints every 250 updates (+ connector/adapter
weights after each stage), preserving optimizer/scheduler/scaler/sampler position/stage/RNG.
Runs development evaluation before training, every 500 updates, and at stage endpoints.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from common import SEED
from common.checkpoint import (CheckpointState, checkpoint_path, latest_checkpoint,
                               load_checkpoint, prune_checkpoints, save_checkpoint)
from common.config import build_arg_parser, load_yaml, run_id_from_config
from common.logging_utils import JsonlLogger
from common.paths import ensure_dir, task2_dir
from common.precision import PrecisionConfig, make_grad_scaler, optimizer_step
from common.resources import Timer, count_params, peak_gpu_bytes
from common.seeding import capture_rng_state, restore_rng_state, seed_all

from .data import CurriculumSampler
from .features import load_cached_features
from .model import SigLIPQwenVLM, sanity_checks


def _stage_bounds(stages: list[dict]) -> list[tuple[int, int]]:
    """Return [(start, end), ...] global-update ranges for each stage."""
    bounds, acc = [], 0
    for s in stages:
        n = int(s["updates"])
        bounds.append((acc, acc + n))
        acc += n
    return bounds


def _build_optimizer(model: SigLIPQwenVLM, stage_cfg: dict[str, Any], cfg: dict[str, Any]):
    """AdamW param groups for a stage: connector (always) + LoRA (if enabled), per-stage LRs.

    Weight decay follows the Task-1 rule: applied to >=2-dim weights, excluded from biases/norms.
    """
    wd = float(cfg["train"]["optimizer"].get("weight_decay", 0.01))

    def split(params):
        decay = [p for p in params if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in params if p.requires_grad and p.ndim < 2]
        return decay, no_decay

    groups = []
    c_decay, c_nodecay = split(model.connector_parameters())
    groups.append({"params": c_decay, "weight_decay": wd, "lr": float(stage_cfg["connector_lr"])})
    groups.append({"params": c_nodecay, "weight_decay": 0.0, "lr": float(stage_cfg["connector_lr"])})

    if stage_cfg.get("lora_enabled", False) and model.adapters:
        a_decay, a_nodecay = split(model.adapter_parameters())
        lora_lr = float(stage_cfg["lora_lr"])
        groups.append({"params": a_decay, "weight_decay": wd, "lr": lora_lr})
        groups.append({"params": a_nodecay, "weight_decay": 0.0, "lr": lora_lr})

    groups = [g for g in groups if g["params"]]
    opt_cfg = cfg["train"]["optimizer"]
    optimizer = torch.optim.AdamW(groups, betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
                                  eps=float(opt_cfg.get("eps", 1e-8)))
    return optimizer


def _build_scheduler(optimizer, stage_updates: int, cfg: dict[str, Any]):
    """Per-stage warmup -> cosine decay to min_lr_fraction of peak (scales each group's own lr)."""
    sched = cfg["train"]["lr_schedule"]
    warmup = int(round(float(sched.get("warmup_fraction", 0.05)) * stage_updates))
    min_frac = float(sched.get("min_lr_fraction", 0.10))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, stage_updates - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_frac + (1.0 - min_frac) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _collect_weights(model: SigLIPQwenVLM) -> dict[str, Any]:
    """Trainable state to checkpoint: connector + adapter factors (frozen Qwen reloaded on resume)."""
    return {
        "connector": model.connector.state_dict(),
        "adapters": {name: {"A": ad.A.detach().cpu(), "B": ad.B.detach().cpu()}
                     for name, ad in model.adapters.items()},
    }


def _load_weights(model: SigLIPQwenVLM, weights: dict[str, Any]) -> None:
    model.connector.load_state_dict(weights["connector"])
    with torch.no_grad():
        for name, ad in model.adapters.items():
            saved = weights["adapters"][name]
            ad.A.copy_(saved["A"].to(ad.A.device, ad.A.dtype))
            ad.B.copy_(saved["B"].to(ad.B.device, ad.B.dtype))


def train(cfg: dict[str, Any], *, resume: bool = False, stop_after: int | None = None) -> None:
    run_id = cfg.get("run_id") or "task2_run"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(SEED)

    model = SigLIPQwenVLM(cfg)
    model.to(device)

    pcfg = PrecisionConfig(compute_dtype=cfg["precision"].get("compute_dtype", "float16"),
                           fp32_master=False, grad_clip_norm=float(cfg["train"]["grad_clip_norm"]))
    amp_enabled = device.type == "cuda"
    scaler = make_grad_scaler(enabled=amp_enabled, device_type=device.type)

    micro = int(cfg["train"]["microbatch_size"])
    accum = int(cfg["train"]["grad_accum_steps"])
    draws_per_micro = max(1, micro // 2)                 # each draw = 2 examples
    stages = cfg["stages"]
    bounds = _stage_bounds(stages)
    total_updates = int(cfg["train"]["total_updates"])
    save_every = int(cfg.get("checkpoint", {}).get("save_every_updates", 250))
    # Dev-eval schedule lives in the referenced eval config (development.at_updates).
    eval_ref = cfg.get("eval", {})
    dev_cfg = load_yaml(eval_ref["config"]) if isinstance(eval_ref, dict) and "config" in eval_ref else {}
    dev_at = set(int(u) for u in dev_cfg.get("development", {}).get("at_updates", []))

    ckpt_dir = ensure_dir(task2_dir("checkpoints") / run_id)
    results = ensure_dir(task2_dir("results") / run_id)
    logger = JsonlLogger(os.fspath(results / "train.jsonl"), append=resume)

    # --- resume -------------------------------------------------------------------------------
    start_update, resume_sampler_state, resume_opt, resume_sched = 0, None, None, None
    if resume:
        latest = latest_checkpoint(os.fspath(ckpt_dir))
        if latest is None:
            print(f"[task2.train] --resume but no checkpoint; STARTING FRESH {run_id}.", flush=True)
        else:
            ck = load_checkpoint(latest)
            _load_weights(model, ck.weights)
            scaler.load_state_dict(ck.scaler)
            restore_rng_state(ck.rng)
            start_update = int(ck.step)
            resume_sampler_state = ck.data_position.get("sampler")
            resume_opt = ck.extra.get("optimizer")
            resume_sched = ck.extra.get("scheduler")
            print(f"[task2.train] RESUMING {run_id} at update {start_update}/{total_updates}.",
                  flush=True)
    if start_update == 0:
        print(f"[task2.train] STARTING FRESH {run_id}: 0/{total_updates} updates.", flush=True)

    if start_update >= total_updates:
        print(f"[task2.train] {run_id} already complete ({start_update}/{total_updates}).", flush=True)
        logger.close()
        return

    # Sanity checks before Stage 1 on every run (real + smoke): one real forward/backward asserting
    # connector grads + feature/visual shapes + frozen backbone. Abort on failure; log the report.
    try:
        sanity_report = sanity_checks(model, cfg)
    except Exception as exc:
        logger.log({"type": "sanity", "ok": False, "update": start_update, "error": repr(exc)})
        logger.close()
        raise SystemExit(f"[task2.train] sanity_checks FAILED: {exc!r}")
    logger.log({"type": "sanity", "ok": True, "update": start_update, "report": sanity_report})
    print(f"[task2.train] sanity_checks OK: visual={sanity_report.get('visual_shape')} "
          f"connector_grad_ok={sanity_report.get('connector_grad_ok')}", flush=True)

    feat_cache: dict[int, torch.Tensor] = {}

    def features_for(examples):
        feats = []
        for ex in examples:
            t = feat_cache.get(ex.image_id)
            if t is None:
                t = load_cached_features(cfg, ex.image_id, device=device)
                feat_cache[ex.image_id] = t
            feats.append(t)
        return torch.stack(feats)

    def maybe_dev_eval(update: int) -> None:
        if update not in dev_at:
            return
        try:
            from .evaluate import run_development
            metrics = run_development(model, cfg, update)
            logger.log({"type": "dev_eval", "update": update, **metrics})
        except Exception as exc:  # dev eval must never crash a long training run
            logger.log({"type": "dev_eval_error", "update": update, "error": repr(exc)})

    def save(update: int, stage_idx: int, sampler, optimizer, scheduler, tag: str = "recovery") -> None:
        state = CheckpointState(
            weights=_collect_weights(model),
            optimizer={}, scheduler={}, scaler=scaler.state_dict(),
            rng=capture_rng_state(),
            data_position={"sampler": sampler.get_state(), "stage_idx": stage_idx},
            step=update,
            extra={"run_id": run_id, "stage_idx": stage_idx, "tag": tag,
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        )
        path = checkpoint_path(os.fspath(ckpt_dir), update)
        save_checkpoint(path, state)
        prune_checkpoints(os.fspath(ckpt_dir), keep_steps={update})
        logger.log({"type": "peak_gpu", "update": update, "after": "checkpoint_save",
                    "peak_gpu_bytes": peak_gpu_bytes()})

    # --- stage loop ---------------------------------------------------------------------------
    update = start_update
    stopped_early = False
    maybe_dev_eval(0) if start_update == 0 else None

    with Timer() as timer:
        for stage_idx, (stage_cfg, (g0, g1)) in enumerate(zip(stages, bounds)):
            if update >= g1:
                continue                                    # whole stage already done (resume)

            # Freeze/unfreeze adapters for this stage (Stage 1 connector-only), then log the
            # requires_grad-derived trainable counts before building the optimizer.
            model.set_adapter_trainable(bool(stage_cfg.get("lora_enabled", False)))
            counts = model.trainable_param_counts()
            logger.log({"type": "stage_start", "stage": stage_cfg["name"], "stage_idx": stage_idx,
                        "update": update, "trainable_params": counts})
            print(f"[task2.train] stage {stage_cfg['name']}: trainable params "
                  f"connector={counts['connector']} adapters={counts['adapters']} "
                  f"total={counts['total']}", flush=True)

            sampler = CurriculumSampler(cfg, stage_cfg, model.tokenizer,
                                        seed=int(SEED + 1000 * (stage_idx + 1)))
            optimizer = _build_optimizer(model, stage_cfg, cfg)
            scheduler = _build_scheduler(optimizer, int(stage_cfg["updates"]), cfg)

            # Resume into the middle of this stage: restore sampler + opt/sched for the stage.
            if update > g0 and resume_sampler_state is not None:
                sampler.set_state(resume_sampler_state)
                if resume_opt is not None:
                    optimizer.load_state_dict(resume_opt)
                if resume_sched is not None:
                    scheduler.load_state_dict(resume_sched)
                resume_sampler_state = resume_opt = resume_sched = None

            trainable = [p for g in optimizer.param_groups for p in g["params"]]
            remaining = g1 - update
            draw_iter = sampler.draws(remaining * accum * draws_per_micro)

            for _ in range(remaining):
                optimizer.zero_grad(set_to_none=True)
                micro_loss_sum = 0.0
                n_examples = 0
                for _m in range(accum):
                    examples = []
                    for _d in range(draws_per_micro):
                        examples.extend(next(draw_iter))
                    if not examples:
                        continue
                    feats = features_for(examples)
                    out = model(examples, feats, pcfg=pcfg, amp_enabled=amp_enabled,
                                with_image=True)
                    loss = out["loss"] / accum
                    scaler.scale(loss).backward()
                    micro_loss_sum += float(out["loss"].detach())
                    n_examples += out["n_examples"]

                info = optimizer_step(optimizer, scaler, trainable, pcfg,
                                      scheduler=scheduler, step=update + 1)
                update += 1

                logger.log({"update": update, "stage": stage_cfg["name"], "stage_idx": stage_idx,
                            "loss": micro_loss_sum / accum, "lr": scheduler.get_last_lr()[0],
                            "grad_norm": info["grad_norm"], "skipped": info["skipped"],
                            "scale": info["scale"], "examples": n_examples})
                if update % 50 == 0 or update == g0 + 1:
                    print(f"[task2.train] {run_id} {update}/{total_updates} "
                          f"stage={stage_cfg['name']} loss={micro_loss_sum / accum:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

                maybe_dev_eval(update)
                if update % save_every == 0:
                    save(update, stage_idx, sampler, optimizer, scheduler)

                if stop_after is not None and update >= stop_after:
                    stopped_early = True
                    break

            # End of stage: dev eval + save connector/adapter weights.
            if not stopped_early:
                maybe_dev_eval(update)
                save(update, stage_idx, sampler, optimizer, scheduler, tag=f"stage{stage_idx}_end")
            if stopped_early:
                break

    if stopped_early:
        print(f"[task2.train] --stop-after {stop_after} reached at {update}/{total_updates}; "
              f"re-run with --resume.", flush=True)
        logger.close()
        return

    trainable_params, total_params = count_params(model)
    logger.log({"type": "resources", "run_id": run_id, "wall_seconds": timer.wall_seconds,
                "peak_gpu_bytes": timer.peak_gpu_bytes, "trainable_params": trainable_params,
                "total_params": total_params})
    logger.close()
    print(f"[task2.train] FINISHED {run_id}: {update}/{total_updates} updates, "
          f"{timer.wall_seconds:.1f}s, peak GPU {timer.peak_gpu_bytes} bytes.", flush=True)


def main() -> None:
    parser = build_arg_parser("Task 2 three-stage VLM curriculum training (resumable).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    cfg.setdefault("run_id", run_id_from_config(args.config, cfg))
    train(cfg, resume=args.resume, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
