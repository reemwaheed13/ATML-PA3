"""Resource accounting: wall time, peak GPU memory, parameter counts, checkpoint sizes.

Every task requires a resource comparison (Task 1 Step 1 "training time, peak GPU memory,
trainable parameter count, and checkpoint sizes"; Task 2 Required Evidence "training and
generation time, peak GPU memory, and trainable parameters by stage, including feature-caching
costs"; Task 3 live run records runtime, I/O tokens, peak GPU memory, p12). These helpers
produce the numbers that land in results/ as machine-readable rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


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

    TODO: on __enter__ record perf_counter and torch.cuda.reset_peak_memory_stats();
          on __exit__ compute elapsed and torch.cuda.max_memory_allocated().
    """

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        raise NotImplementedError


def count_params(module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a module.

    TODO: sum p.numel() over parameters, split by requires_grad.
    """
    raise NotImplementedError


def file_size_bytes(path: str) -> int:
    """Return the on-disk size of a checkpoint file (for the resource table).

    TODO: os.path.getsize.
    """
    raise NotImplementedError


def peak_gpu_bytes() -> int:
    """Return torch.cuda.max_memory_allocated() (0 if no CUDA).

    TODO: implement.
    """
    raise NotImplementedError
