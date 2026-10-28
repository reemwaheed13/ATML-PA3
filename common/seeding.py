"""Deterministic seeding.

Handout requires seed 6304 "everywhere" and asks to fix random seeds where practical
(Instructions p1; Task 1 p3; Task 2 p8; Task 3 p11). This module centralizes the seeding of
python / numpy / torch / cuda so every script seeds identically, and provides capture/restore
of RNG state for the resumable-checkpoint contract (see common.checkpoint).

Determinism tradeoff
--------------------
`seed_all` always seeds every generator, but *forcing* deterministic kernels is behind the
`deterministic` flag, which defaults to **off**. The reasons:

  * Reproducibility of a run across machines is already covered by seeding + the resumable
    checkpoint (same seed, same data order, restored RNG). Deterministic kernels only matter
    for bit-exact *GPU* reductions, which the handout does not require for the training runs.
  * `torch.use_deterministic_algorithms(True)` makes some ops slower or unavailable entirely
    (it raises for ops with no deterministic implementation), and cuDNN determinism disables
    the autotuner (`cudnn.benchmark`), costing throughput on the L4 we train on.
  * It also needs `CUBLAS_WORKSPACE_CONFIG` set *before* CUDA initializes to avoid a runtime
    error, which is brittle inside a long-lived Colab kernel.

So the default is: seed everything (reproducible given the same hardware), let the autotuner
run (fast). Flip `deterministic=True` for the test harness or when debugging a divergence,
accepting the speed hit. Note that even with the flag off, re-running the *same* computation on
the *same* device is reproducible in practice; the flag guards against cross-device reduction
order differences.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

from . import SEED


def seed_all(seed: int = SEED, *, deterministic: bool = False) -> None:
    """Seed python, numpy, torch (cpu+cuda) and optionally request deterministic algorithms.

    Args:
        seed: base seed (default 6304).
        deterministic: if True, request deterministic cuDNN/algorithms (slower, and raises on
            ops with no deterministic kernel). Default False — see the module docstring for the
            tradeoff. With the flag off, cuDNN's autotuner (`benchmark`) is left on for speed.
    """
    # Affects hash-based ordering only in *child* processes (it is read at interpreter start),
    # but we set it so any spawned worker inherits the same seed.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic cuBLAS GEMMs; must be set before CUDA context creation to
        # take effect, so we only set it (not override) here and document the limitation.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only=True: downgrade "no deterministic impl" errors to warnings so a stray op
        # does not crash a run that merely asked for best-effort determinism.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def capture_rng_state() -> dict[str, Any]:
    """Return a dict of python/numpy/torch(cpu)/cuda RNG states for checkpointing.

    The returned object is pickle/torch.save-friendly. CUDA state is included only when CUDA is
    available; on restore it is applied best-effort so a checkpoint taken on GPU can still be
    loaded on CPU (and vice versa).
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states previously captured by capture_rng_state (used on --resume).

    Inverse of capture_rng_state. The CPU generators (python/numpy/torch) are always restored;
    CUDA state is restored only if present and CUDA is available, tolerating a mismatch in the
    number of visible devices between save and load.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    torch_state = state["torch"]
    # torch.set_rng_state wants a CPU ByteTensor; torch.load(map_location="cpu") yields that,
    # but coerce defensively in case the state was moved to another device.
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
    torch.set_rng_state(torch_state.cpu().to(torch.uint8))

    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except (RuntimeError, ValueError):
            # Device count differs between the saving and loading environments; restore what we
            # can, device by device.
            for i, s in enumerate(cuda_states):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(s, i)


def spawn_generator(seed: int = SEED, *tags: int) -> np.random.Generator:
    """Return an independent NumPy Generator derived deterministically from (seed, *tags).

    Used for seeded data selection and the Task 3 permutation orders, so that independent
    streams (e.g. the ten rollout permutations) are reproducible and statistically independent
    without sharing or perturbing the global RNG state.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), *(int(t) for t in tags)]))
