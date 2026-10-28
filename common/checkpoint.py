"""Resumable checkpoint save/load — one primitive for both tasks.

Colab's local disk is wiped on disconnect, so every training run must be resumable from
DRIVE_ROOT. A checkpoint bundles everything needed to continue bit-for-bit:
  * weights: a full model state_dict OR just adapter/connector tensors (model_or_adapter).
  * optimizer, scheduler, GradScaler states.
  * RNG state (python/numpy/torch/cuda) via common.seeding.
  * data position: dataset cursor (Task 1) or sampler position + stage progress (Task 2).
  * step/update counters.

Handout: Task 1 p3 "Preserve optimizer, scheduler, gradient-scaler, RNG, and data-position
state for resuming." Task 2 p8 "Include optimizer, scheduler, scaler, sampler position, stage
progress, and RNG state." Recovery checkpoints are saved on the schedules each task defines.

Durability contract
--------------------
`save_checkpoint` is crash-safe against a Colab disconnect mid-write:

  1. It serializes to a temp file in the *same directory* as the target, flushes it, and
     `fsync`s the file descriptor so the bytes reach disk (best-effort: fsync is skipped where
     the platform/filesystem — e.g. a FUSE-mounted Drive — does not support it).
  2. It then `os.replace`s the temp file onto the final path. `os.replace` is atomic on both
     POSIX and Windows, so a reader either sees the complete old checkpoint or the complete new
     one — never a half-written file. The previous good checkpoint stays intact until the
     rename succeeds.
  3. On any failure the temp file is removed, leaving the previous checkpoint untouched.

Because each recovery checkpoint is written to its own step-stamped filename, distinct
checkpoints never overwrite each other; `latest_checkpoint` picks the most advanced one on
resume. Callers may prune older files *after* a successful save if they only want to keep the
last good one.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

CHECKPOINT_PREFIX = "ckpt_step"
CHECKPOINT_SUFFIX = ".pt"
_STEP_RE = re.compile(rf"^{re.escape(CHECKPOINT_PREFIX)}(\d+){re.escape(CHECKPOINT_SUFFIX)}$")

FORMAT_VERSION = 1


@dataclass
class CheckpointState:
    """Everything a run needs to resume. `weights` is a full model or just adapters/connector."""

    weights: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    rng: dict[str, Any]
    data_position: dict[str, Any]          # dataset cursor / sampler position + stage progress
    step: int
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. stage index, best-dev metric


def checkpoint_path(dir_path: str, step: int) -> str:
    """Canonical path for the recovery checkpoint at `step` inside `dir_path`."""
    return os.path.join(dir_path, f"{CHECKPOINT_PREFIX}{int(step)}{CHECKPOINT_SUFFIX}")


def save_checkpoint(
    path: str, state: CheckpointState, *, atomic: bool = True, verify: bool = True
) -> None:
    """Serialize a CheckpointState to `path` on DRIVE_ROOT.

    Args:
        atomic: write to a temp file (flush + fsync) then os.replace onto `path`, so a
            disconnect mid-write cannot corrupt the previous good checkpoint. See the module
            docstring. Set False only for throwaway/local tests.
        verify: after the atomic replace, read the file back (size > 0 + torch.load succeeds +
            required keys present) and raise RuntimeError if it is unreadable. Drive's FUSE mount
            has silently corrupted a file on us before, so a caller must not prune the previous
            good checkpoint until this reload confirms the new one is intact. Only meaningful with
            atomic=True.
    """
    # Build the payload explicitly (not dataclasses.asdict, which deep-copies tensors).
    payload = {
        "format_version": FORMAT_VERSION,
        "weights": state.weights,
        "optimizer": state.optimizer,
        "scheduler": state.scheduler,
        "scaler": state.scaler,
        "rng": state.rng,
        "data_position": state.data_position,
        "step": state.step,
        "extra": state.extra,
    }

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    if not atomic:
        torch.save(payload, path)
        return

    # Temp file in the same directory so os.replace stays on one filesystem (atomic rename).
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{CHECKPOINT_PREFIX}.", suffix=f"{CHECKPOINT_SUFFIX}.tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            _fsync_fd(f.fileno())
        os.replace(tmp_path, path)  # atomic; previous checkpoint survives until this succeeds
        _fsync_dir(directory)       # persist the rename itself (best-effort)
    except BaseException:
        # Never leave a partial temp file behind, and never touch the prior good checkpoint.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    if verify:
        _verify_checkpoint(path)


def _verify_checkpoint(path: str) -> None:
    """Read a just-written checkpoint back to confirm it is intact; raise if not.

    Catches a truncated/garbled write (e.g. Drive FUSE flaking) before the caller trusts it or
    prunes the previous good checkpoint. Does a full torch.load so a partially-flushed pickle is
    caught, not just a zero-byte file.
    """
    try:
        if os.path.getsize(path) <= 0:
            raise RuntimeError("file is empty")
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or "weights" not in obj or "step" not in obj:
            raise RuntimeError("missing expected keys")
    except Exception as exc:  # noqa: BLE001 — re-raised with context below
        raise RuntimeError(
            f"checkpoint {path} failed read-back verification ({exc}); the file may be "
            f"corrupt. The previous checkpoint was left untouched."
        ) from exc


def load_checkpoint(path: str, map_location: str = "cpu") -> CheckpointState:
    """Load a CheckpointState from `path`.

    Loads with `weights_only=False`: a checkpoint intentionally contains non-tensor objects
    (python/numpy RNG tuples, scaler/scheduler dicts), which torch>=2.6's default weights-only
    unpickler would reject. Checkpoints are our own trusted artifacts on DRIVE_ROOT.
    """
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch without the weights_only kwarg.
        obj = torch.load(path, map_location=map_location)

    return CheckpointState(
        weights=obj["weights"],
        optimizer=obj["optimizer"],
        scheduler=obj["scheduler"],
        scaler=obj["scaler"],
        rng=obj["rng"],
        data_position=obj["data_position"],
        step=obj["step"],
        extra=obj.get("extra", {}),
    )


def latest_checkpoint(dir_path: str) -> Optional[str]:
    """Return the most advanced (highest-step) checkpoint in `dir_path`, or None.

    Used by --resume to find where to continue without the caller tracking step numbers.
    """
    if not os.path.isdir(dir_path):
        return None
    best_step = -1
    best_path: Optional[str] = None
    for name in os.listdir(dir_path):
        m = _STEP_RE.match(name)
        if m is None:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = os.path.join(dir_path, name)
    return best_path


def list_checkpoints(dir_path: str) -> dict[int, str]:
    """Map step -> path for every recovery checkpoint in `dir_path` (empty if none)."""
    out: dict[int, str] = {}
    if not os.path.isdir(dir_path):
        return out
    for name in os.listdir(dir_path):
        m = _STEP_RE.match(name)
        if m is not None:
            out[int(m.group(1))] = os.path.join(dir_path, name)
    return out


def prune_checkpoints(dir_path: str, keep_steps: "set[int]") -> list[str]:
    """Delete every checkpoint in `dir_path` whose step is not in `keep_steps`.

    Call this only *after* a new checkpoint has been saved and read-back verified, so a
    superseded recovery checkpoint is removed only once its successor is known good. Checkpoints
    at steps you still need (e.g. the eval-checkpoint milestones) must be included in keep_steps.
    Returns the paths removed.
    """
    removed: list[str] = []
    for step, path in list_checkpoints(dir_path).items():
        if step in keep_steps:
            continue
        try:
            os.remove(path)
            removed.append(path)
        except OSError:
            pass
    return removed


def _fsync_fd(fd: int) -> None:
    """fsync a file descriptor, tolerating platforms/filesystems that can't (e.g. Drive FUSE)."""
    try:
        os.fsync(fd)
    except (OSError, AttributeError):
        pass


def _fsync_dir(directory: str) -> None:
    """fsync a directory so the rename is durable. No-op on Windows (can't open a dir fd)."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(dir_fd)
    except (OSError, AttributeError):
        pass
    finally:
        os.close(dir_fd)
