"""Per-stage training-time + peak-GPU measurement for the 3-stage curriculum. (handout p9 evidence)

    python -m task2_vlm.time_stages --config configs/task2/vlm_stages.yaml

The committed training log only records the wall time / peak memory of the FINAL resumed process
(updates 3251-4500), which under-counts the per-stage cost. This script measures each stage directly
and independently of the long run: it rebuilds the model at that stage's STARTING weights, runs a few
warm-up updates, then times a fixed number of real updates with that stage's mixture, LRs, optimizer,
precision, batch, and gradient-checkpointing settings, and extrapolates to the stage's update count.

Starting weights per stage:
  * Stage 1: a fresh seeded connector plus the recorded calibration scale (from the training log).
  * Stage 2: the saved weights after Stage 1 (checkpoint at the Stage-1 boundary update).
  * Stage 3: the saved weights after Stage 2 (checkpoint at the Stage-2 boundary update).

Nothing is saved over existing checkpoints or results; the only output is
results/task2/stage_timing.json. The per-stage estimates EXCLUDE dev-evaluation and checkpoint-save
time (neither runs here). A stage whose starting checkpoint is missing is skipped with a note.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch

from common import SEED
from common.checkpoint import list_checkpoints, load_checkpoint
from common.config import build_arg_parser, load_yaml, run_id_from_config
from common.logging_utils import read_jsonl
from common.paths import REPO_ROOT, ensure_dir, task2_dir
from common.precision import PrecisionConfig, make_grad_scaler, optimizer_step
from common.resources import Timer
from common.seeding import seed_all

from .data import CurriculumSampler
from .features import load_cached_features
from .model import SigLIPQwenVLM
from .train import _build_optimizer, _build_scheduler, _load_weights, _stage_bounds


def _train_log_rows(run_id: str) -> list[dict]:
    """Read the committed training log (repo copy first, then the DRIVE_ROOT working copy)."""
    repo = REPO_ROOT / "results" / "task2" / "train" / f"{run_id}.jsonl"
    if repo.exists():
        return read_jsonl(os.fspath(repo))
    drive = os.environ.get("DRIVE_ROOT")
    if drive:
        alt = os.path.join(drive, "task2", "results", run_id, "train.jsonl")
        if os.path.exists(alt):
            return read_jsonl(alt)
    return []


def _calibration_scale(rows: list[dict]) -> Optional[float]:
    """Recorded connector calibration scale from the training log (last such row), or None."""
    scale = None
    for r in rows:
        if r.get("type") == "calibration" and "scale" in r:
            scale = float(r["scale"])
    return scale


def _logged_final_wall_seconds(rows: list[dict]) -> Optional[float]:
    """Wall time of the final resumed training process, from the log's resources row."""
    for r in rows:
        if r.get("type") == "resources" and "wall_seconds" in r:
            return float(r["wall_seconds"])
    return None


def _features_for(model: SigLIPQwenVLM, cfg: dict[str, Any], examples, cache: dict) -> torch.Tensor:
    feats = []
    for ex in examples:
        t = cache.get(ex.image_id)
        if t is None:
            t = load_cached_features(cfg, ex.image_id, device=model.device)
            cache[ex.image_id] = t
        feats.append(t)
    return torch.stack(feats)


def _time_stage(model: SigLIPQwenVLM, cfg: dict[str, Any], stage_cfg: dict[str, Any],
                stage_idx: int, warmup: int, timed: int) -> dict[str, Any]:
    """Run `warmup` untimed + `timed` timed real updates for one stage; return timing stats."""
    device = model.device
    amp = device.type == "cuda"
    pcfg = PrecisionConfig(compute_dtype=cfg["precision"].get("compute_dtype", "float16"),
                           fp32_master=False, grad_clip_norm=float(cfg["train"]["grad_clip_norm"]))
    scaler = make_grad_scaler(enabled=amp, device_type=device.type)

    micro = int(cfg["train"]["microbatch_size"])
    accum = int(cfg["train"]["grad_accum_steps"])
    draws_per_micro = max(1, micro // 2)

    model.set_adapter_trainable(bool(stage_cfg.get("lora_enabled", False)))
    model.train()
    sampler = CurriculumSampler(cfg, stage_cfg, model.tokenizer,
                               seed=int(SEED + 1000 * (stage_idx + 1)))
    optimizer = _build_optimizer(model, stage_cfg, cfg)
    scheduler = _build_scheduler(optimizer, int(stage_cfg["updates"]), cfg)
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    draw_iter = sampler.draws((warmup + timed) * accum * draws_per_micro)
    cache: dict[int, torch.Tensor] = {}

    def run_updates(n: int, base: int) -> None:
        for k in range(n):
            optimizer.zero_grad(set_to_none=True)
            for _m in range(accum):
                examples = []
                for _d in range(draws_per_micro):
                    examples.extend(next(draw_iter))
                if not examples:
                    continue
                feats = _features_for(model, cfg, examples, cache)
                out = model(examples, feats, pcfg=pcfg, amp_enabled=amp, with_image=True)
                scaler.scale(out["loss"] / accum).backward()
            optimizer_step(optimizer, scaler, trainable, pcfg, scheduler=scheduler, step=base + k + 1)

    run_updates(warmup, 0)                                   # untimed warm-up
    with Timer() as timer:                                   # Timer resets CUDA peak stats on entry
        run_updates(timed, warmup)

    sec_per_update = timer.wall_seconds / max(1, timed)
    stage_updates = int(stage_cfg["updates"])
    return {
        "seconds_per_update": sec_per_update,
        "peak_gpu_bytes": int(timer.peak_gpu_bytes),
        "timed_updates": int(timed),
        "stage_updates": stage_updates,
        "estimated_stage_seconds": sec_per_update * stage_updates,
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = cfg.get("run_id") or "task2_run"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = cfg.get("stage_timing", {})
    warmup = int(st.get("warmup_updates", 5))
    timed = int(st.get("timed_updates", 50))

    stages = cfg["stages"]
    bounds = _stage_bounds(stages)                          # [(0,g1),(g1,g2),(g2,g3)]
    ckpt_dir = task2_dir("checkpoints") / run_id
    ckpts = list_checkpoints(os.fspath(ckpt_dir))

    rows = _train_log_rows(run_id)
    calib_scale = _calibration_scale(rows)

    stage_results: list[dict[str, Any]] = []
    for stage_idx, stage_cfg in enumerate(stages):
        seed_all(SEED)
        model = SigLIPQwenVLM(cfg)
        model.to(device)

        start_step = bounds[stage_idx][0]
        if stage_idx == 0:
            if calib_scale is not None:
                model.connector.set_scale(calib_scale)
                start_desc = f"fresh_connector+calibration_scale({calib_scale:.6g})"
            else:
                start_desc = "fresh_connector (no calibration row found; scale=1.0)"
        else:
            path = ckpts.get(start_step)
            if path is not None:
                _load_weights(model, load_checkpoint(path).weights)
                start_desc = f"checkpoint@update{start_step}"
            elif ckpts:
                # Stage-start checkpoint missing: fall back to the latest available weights. Seconds
                # per update and peak memory depend on shapes, trainable params, and data, not on the
                # weight values, so this still yields valid per-stage timing (with this stage's own
                # config: LoRA/mixture/LRs applied below in _time_stage).
                fb_step = max(ckpts)
                _load_weights(model, load_checkpoint(ckpts[fb_step]).weights)
                start_desc = (f"fallback: checkpoint@update{fb_step} (stage-start checkpoint "
                              "missing; timing is weight-independent)")
            else:
                print(f"[task2.time_stages] no checkpoint at all in {ckpt_dir}; cannot time stage "
                      f"{stage_idx} ({stage_cfg['name']}). Skipping.", flush=True)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        print(f"[task2.time_stages] timing stage {stage_idx} ({stage_cfg['name']}) from "
              f"{start_desc}: {warmup} warm-up + {timed} timed updates...", flush=True)
        timing = _time_stage(model, cfg, stage_cfg, stage_idx, warmup, timed)
        timing.update({"stage_idx": stage_idx, "stage": stage_cfg["name"],
                       "start_weights": start_desc})
        stage_results.append(timing)
        print(f"[task2.time_stages]   {timing['seconds_per_update']:.3f}s/update, peak GPU "
              f"{timing['peak_gpu_bytes']} bytes, est. stage {timing['estimated_stage_seconds']:.1f}s",
              flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {
        "run_id": run_id,
        "note": (f"Per-stage seconds/update measured over {timed} timed updates after {warmup} "
                 "warm-up updates, from each stage's starting weights. estimated_stage_seconds "
                 "EXCLUDES dev-evaluation and checkpoint-save time (neither runs here)."),
        "warmup_updates": warmup,
        "timed_updates": timed,
        "final_process_wall_seconds_logged": _logged_final_wall_seconds(rows),
        "final_process_note": ("Logged training wall time covers only the final resumed process "
                               "(updates 3251-4500), not whole-run per-stage time; shown for "
                               "reference."),
        "stages": stage_results,
    }

    out_dir = ensure_dir(REPO_ROOT / "results" / "task2")
    with open(os.fspath(out_dir / "stage_timing.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[task2.time_stages] wrote results/task2/stage_timing.json "
          f"({len(stage_results)} stages timed).", flush=True)
    return out


def main() -> None:
    parser = build_arg_parser("Task 2 per-stage training-time + peak-GPU measurement.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    cfg.setdefault("run_id", run_id_from_config(args.config, cfg))
    run(cfg)


if __name__ == "__main__":
    main()
