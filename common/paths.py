"""Path resolution — everything derives from a single DRIVE_ROOT.

Colab wipes local disk on disconnect, so every expensive artifact is addressed relative to
DRIVE_ROOT (a Google Drive folder). Scripts never hardcode absolute paths; they ask this
module. The layout mirrors the approved Drive structure:

    DRIVE_ROOT = MyDrive/ATML-PA3
    ├── data/                         # raw datasets + supplied files (parquet, first_plot_questions.yaml)
    ├── task1/{checkpoints,eval,generations,results}
    ├── task2/{features,checkpoints,eval}
    └── task3/{live,cache}

The HF model/tokenizer cache (HF_HOME) deliberately lives on Colab's LOCAL disk at
/content/hf_cache, NOT on Drive: the hub cache is large and churny and the 0.6B model
re-downloads quickly, so keeping it local avoids thrashing Drive (and its rate limits) every
run. See hf_cache_dir().

Small, committed artifacts (split IDs, aggregate results) live in the repo under splits/ and
results/ instead — resolved here too so callers have one source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of this file's package.
REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "splits"     # committed: write-once selection IDs
RESULTS_DIR = REPO_ROOT / "results"   # committed: small machine-readable outputs

DRIVE_ROOT_ENV = "DRIVE_ROOT"         # set by colab/setup.ipynb
HF_HOME_ENV = "HF_HOME"               # set by colab/setup.ipynb (local, not Drive)
LOCAL_HF_CACHE = Path("/content/hf_cache")  # Colab-local HF cache default


def drive_root() -> Path:
    """Return DRIVE_ROOT from the environment (set in the Colab bootstrap).

    Fails fast with a clear error if unset, so a run never silently writes expensive artifacts to
    Colab's ephemeral local disk (wiped on disconnect).
    """
    value = os.environ.get(DRIVE_ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"{DRIVE_ROOT_ENV} is not set. Run colab/setup.ipynb (which mounts Drive and sets it) "
            f"or pass --drive-root."
        )
    return Path(value)


def data_dir() -> Path:
    """DRIVE_ROOT/data — raw datasets and supplied files (gitignored).

    NOTE: first_plot_questions.yaml and the ATML-PA3-Data parquet files are expected here; the
    exact filenames are config fields (not hardcoded) pending confirmation they are supplied.
    """
    return drive_root() / "data"


def hf_cache_dir() -> Path:
    """Local HF cache (HF_HOME), on Colab's LOCAL disk — NOT on Drive.

    The bootstrap sets HF_HOME=/content/hf_cache: the hub cache is large and churny and the 0.6B
    model re-downloads quickly, so keeping it local avoids thrashing Drive every run. Returns
    $HF_HOME if set, else the /content/hf_cache default.
    """
    value = os.environ.get(HF_HOME_ENV)
    return Path(value) if value else LOCAL_HF_CACHE


def task1_dir(sub: str = "") -> Path:
    """DRIVE_ROOT/task1[/sub] where sub in {checkpoints, eval, generations}."""
    return drive_root() / "task1" / sub if sub else drive_root() / "task1"


def task2_dir(sub: str = "") -> Path:
    """DRIVE_ROOT/task2[/sub] where sub in {features, checkpoints, eval}."""
    return drive_root() / "task2" / sub if sub else drive_root() / "task2"


def task3_dir(sub: str = "") -> Path:
    """DRIVE_ROOT/task3[/sub] where sub in {live, cache}."""
    return drive_root() / "task3" / sub if sub else drive_root() / "task3"


def ensure_dir(path: os.PathLike | str) -> Path:
    """Create a directory (and parents) if missing; return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
