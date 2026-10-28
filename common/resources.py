"""Resource accounting: wall time, peak GPU memory, parameter counts, checkpoint sizes.

Every task requires a resource comparison (Task 1 Step 1 "training time, peak GPU memory,
trainable parameter count, and checkpoint sizes"; Task 2 Required Evidence "training and
generation time, peak GPU memory, and trainable parameters by stage, including feature-caching
costs"; Task 3 live run records runtime, I/O tokens, peak GPU memory, p12). These helpers
produce the numbers that land in results/ as machine-readable rows.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ResourceReport:
    wall_seconds: float
    peak_gpu_bytes: int
    trainable_params: int
    total_params: int
    extra: dict[str, Any]


class Timer:
    """Context manager measuring wall time; resets CUDA peak-memory stats on entry.

    Usage:
        with Timer() as t: ...   # then t.wall_seconds, t.peak_gpu_bytes
    """

    def __enter__(self) -> "Timer":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.wall_seconds = 0.0
        self.peak_gpu_bytes = 0
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # make the perf_counter/mem reads reflect finished kernels
        self.wall_seconds = time.perf_counter() - self._t0
        self.peak_gpu_bytes = peak_gpu_bytes()


def count_params(module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a module."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def file_size_bytes(path: str) -> int:
    """Return the on-disk size of a checkpoint file (for the resource table)."""
    return os.path.getsize(os.fspath(path))


def peak_gpu_bytes() -> int:
    """Return torch.cuda.max_memory_allocated() (0 if no CUDA)."""
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0
