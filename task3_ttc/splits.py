"""Task 3 dev/eval splits and the live-50 selection. (handout p11)

Within each dataset: sort the 1,000 question IDs lexicographically, permute with NumPy
default_rng(6304); first 200 are development, the remaining 800 are held-out evaluation. Keep all
rollouts of a question together. The first 50 MATH evaluation questions in this permutation are
the live-generation subset. Split IDs are saved under splits/task3 and reused.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from common.paths import SPLITS_DIR, ensure_dir


def make_splits(question_ids: list[str], *, seed: int = 6304, dev_size: int = 200) -> dict[str, list[str]]:
    """Return {"dev": [...200], "eval": [...rest]} via lexicographic sort + default_rng(seed) permute.

    The question IDs are first sorted lexicographically (as strings) to give a canonical, input-
    order-independent starting point, then permuted with a single `default_rng(seed)` draw; the
    first `dev_size` are development, the remainder held-out evaluation. We permute an index array
    and gather, so the mapping is explicit and reproducible regardless of NumPy's internal shuffle
    of object arrays.
    """
    ids = sorted(str(q) for q in question_ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    ordered = [ids[int(i)] for i in perm]
    return {"dev": ordered[:dev_size], "eval": ordered[dev_size:]}


def live_math_50(eval_ids: list[str]) -> list[str]:
    """Return the first 50 MATH evaluation question IDs (in the permuted order) for the live run.

    `eval_ids` must be the MATH dataset's held-out split from `make_splits` (already in permuted
    order), so slicing the first 50 preserves the frozen order (handout p11).
    """
    return list(eval_ids[:50])


def save_splits(dataset: str, splits: dict[str, Any]) -> str:
    """Persist split IDs to splits/task3/<dataset>_splits.json (committed, write-once). Returns path."""
    out_dir = ensure_dir(SPLITS_DIR / "task3")
    path = out_dir / f"{dataset}_splits.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    return str(path)
