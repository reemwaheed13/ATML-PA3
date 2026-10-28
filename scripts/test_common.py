"""Self-contained check that common/ gives bit-exact, resumable training. Run on Colab.

    python scripts/test_common.py        # from the repo root
    # or in a Colab cell:  !python scripts/test_common.py

CPU/GPU agnostic: it uses CUDA + FP16 AMP when a GPU is present, and plain FP32 on CPU, driving
exactly the same loop (common.precision makes the scaler/autocast no-ops when disabled).

What it proves:
  1. Resume reproducibility. Train a tiny model from scratch, checkpoint at step N, then keep
     going to N+5. Separately, reload the step-N checkpoint into fresh objects and re-run to
     N+5. The per-step losses AND the python/numpy/torch(+cuda) RNG draws must match — i.e. the
     checkpoint restored model + optimizer + scheduler + scaler + RNG + the exact seeded data
     position, so the resumed run continued mid-"epoch" in the identical order.
  2. Skipped-step accounting. common.precision.optimizer_step reports a skip via the scaler's
     scale decrease and does NOT advance the LR scheduler on a skip (but does on a good step).

Exits non-zero (via assert) on any mismatch so it is usable as a smoke gate before long runs.
"""

from __future__ import annotations

import pathlib
import random
import sys

# Make `common` importable whether run as `python scripts/test_common.py` (script dir on path)
# or `python -m scripts.test_common` (repo root on path).
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from common import SEED
from common.checkpoint import (
    CheckpointState,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from common.precision import PrecisionConfig, autocast_context, make_grad_scaler, optimizer_step
from common.seeding import capture_rng_state, restore_rng_state, seed_all, spawn_generator

# --- tiny fixed problem -------------------------------------------------------------------

DATA_N = 64
IN_DIM = 8
OUT_DIM = 4
BATCH = 4
N = 3            # checkpoint after this many steps
EXTRA = 5        # then run this many more (the window we compare)


class TinyModel(torch.nn.Module):
    """A couple of linears with dropout, so the forward consumes the torch RNG stream."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(IN_DIM, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(16, OUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


class ResumableSampler:
    """Seeded, position-tracked index sampler over a fixed dataset.

    The visiting order is a deterministic permutation derived from `seed` via
    common.seeding.spawn_generator — identical on every construction, independent of the global
    RNG. `pos` is the data cursor we checkpoint; restoring it continues the same seeded order
    exactly where we left off (mid-"epoch").
    """

    def __init__(self, n: int, batch: int, seed: int) -> None:
        self.n = n
        self.batch = batch
        self.seed = seed
        self.order = spawn_generator(seed).permutation(n)
        self.pos = 0

    def next_indices(self) -> list[int]:
        idx = [int(self.order[(self.pos + i) % self.n]) for i in range(self.batch)]
        self.pos += self.batch
        return idx

    def state(self) -> dict:
        return {"pos": self.pos, "seed": self.seed, "n": self.n, "batch": self.batch}

    def load_state(self, s: dict) -> None:
        # Rebuild the order from the saved seed, then jump to the saved cursor.
        self.order = spawn_generator(s["seed"]).permutation(s["n"])
        self.pos = s["pos"]


def make_training_objects(device: torch.device, amp_enabled: bool, cfg: PrecisionConfig):
    """Fresh model/optimizer/scheduler/scaler/sampler, deterministically initialized."""
    # Seed just before model construction so both the from-scratch run and the reload build the
    # identical initial model (the reload then overwrites it from the checkpoint anyway).
    # deterministic=False (the project default): re-running the *same* ops on the *same* device
    # is already reproducible, and forcing deterministic cuBLAS here would need
    # CUBLAS_WORKSPACE_CONFIG set before CUDA init (not possible mid-kernel on Colab).
    seed_all(SEED, deterministic=False)
    model = TinyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N + EXTRA)
    scaler = make_grad_scaler(enabled=amp_enabled, device_type=device.type)
    sampler = ResumableSampler(DATA_N, BATCH, seed=SEED)
    return model, optimizer, scheduler, scaler, sampler


def run_steps(model, optimizer, scheduler, scaler, sampler, data, n_steps, *,
              device, amp_enabled, cfg, start_step):
    """Run `n_steps` training steps; return (losses, rng_draws) recorded per step."""
    X, Y = data
    model.train()
    losses, draws = [], []
    for local in range(n_steps):
        step = start_step + local + 1

        # Draw from every generator to prove capture/restore of each: python, numpy, torch CPU,
        # and (on GPU) the torch.cuda generator. The cuda draw is None on CPU so the comparison
        # is still exact and device-agnostic.
        cuda_draw = (
            float(torch.rand((), device=device).item()) if device.type == "cuda" else None
        )
        draws.append((random.random(), float(np.random.rand()),
                      float(torch.rand(()).item()), cuda_draw))

        idx = sampler.next_indices()
        xb = X[idx].to(device)
        yb = Y[idx].to(device)
        noise = torch.randn_like(xb) * 0.01  # more torch-RNG consumption

        with autocast_context(cfg, device_type=device.type, enabled=amp_enabled):
            pred = model(xb + noise)
            loss = torch.nn.functional.mse_loss(pred, yb)

        scaler.scale(loss).backward()
        optimizer_step(optimizer, scaler, model.parameters(), cfg,
                       scheduler=scheduler, step=step)  # zeroes grads internally

        losses.append(float(loss.detach().cpu()))
    return losses, draws


def test_resume_reproducibility() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    cfg = PrecisionConfig(compute_dtype="float16", grad_clip_norm=1.0)
    print(f"[resume] device={device.type} amp={amp_enabled}")

    # Fixed dataset, built once and shared by both runs.
    gen = torch.Generator().manual_seed(SEED)
    X = torch.randn(DATA_N, IN_DIM, generator=gen)
    Y = torch.randn(DATA_N, OUT_DIM, generator=gen)
    data = (X, Y)

    ckpt_dir = pathlib.Path(tempfile_dir()) / "test_common_ckpts"

    # --- Run A: scratch -> N, checkpoint, then continue N -> N+EXTRA ---------------------
    model, opt, sched, scaler, sampler = make_training_objects(device, amp_enabled, cfg)
    run_steps(model, opt, sched, scaler, sampler, data, N,
              device=device, amp_enabled=amp_enabled, cfg=cfg, start_step=0)

    # Capture RNG *now* (after step N, before step N+1) — this is the resume point.
    state = CheckpointState(
        weights=model.state_dict(),
        optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),
        scaler=scaler.state_dict(),
        rng=capture_rng_state(),
        data_position={"sampler": sampler.state()},
        step=N,
    )
    path = checkpoint_path(str(ckpt_dir), N)
    save_checkpoint(path, state, atomic=True)

    # The continuation we want the reload to reproduce.
    losses_cont, draws_cont = run_steps(
        model, opt, sched, scaler, sampler, data, EXTRA,
        device=device, amp_enabled=amp_enabled, cfg=cfg, start_step=N)

    # --- Run B: fresh objects, reload from N, re-run N -> N+EXTRA ------------------------
    assert latest_checkpoint(str(ckpt_dir)) == path, "latest_checkpoint did not find our ckpt"
    # Load to CPU: load_state_dict copies weights onto the (possibly CUDA) model, the optimizer
    # casts its state onto the param device, and the RNG tensors stay on CPU where the
    # set_rng_state APIs want them. Mapping straight to CUDA would move the RNG ByteTensors too.
    loaded = load_checkpoint(path)
    assert loaded.step == N

    model2, opt2, sched2, scaler2, sampler2 = make_training_objects(device, amp_enabled, cfg)
    model2.load_state_dict(loaded.weights)
    opt2.load_state_dict(loaded.optimizer)
    sched2.load_state_dict(loaded.scheduler)
    scaler2.load_state_dict(loaded.scaler)
    sampler2.load_state(loaded.data_position["sampler"])
    restore_rng_state(loaded.rng)  # must come after the model build above, which uses the RNG

    losses_reload, draws_reload = run_steps(
        model2, opt2, sched2, scaler2, sampler2, data, EXTRA,
        device=device, amp_enabled=amp_enabled, cfg=cfg, start_step=N)

    # --- Assertions ----------------------------------------------------------------------
    assert draws_cont == draws_reload, (
        "RNG draws diverged after resume — RNG state was not restored exactly:\n"
        f"  continue: {draws_cont}\n  reload:   {draws_reload}")

    assert len(losses_cont) == EXTRA
    for i, (a, b) in enumerate(zip(losses_cont, losses_reload)):
        # Resume is meant to be bit-exact; allow a hair of tolerance only for GPU reduction
        # non-determinism across the two passes.
        assert abs(a - b) <= 1e-6, f"loss mismatch at resumed step {N + i + 1}: {a} vs {b}"

    assert sampler.pos == sampler2.pos, "data cursor diverged after resume"
    cuda_checked = device.type == "cuda"
    print(f"[resume] OK — losses & RNG draws (python/numpy/torch"
          f"{'+cuda' if cuda_checked else ''}) match across {EXTRA} resumed steps "
          f"(final loss {losses_cont[-1]:.6f}, cursor {sampler2.pos})")


class _StubScaler:
    """Minimal GradScaler stand-in with a scripted scale, to test skip detection on any device.

    Forcing a real GradScaler to skip requires inducing inf/nan grads under an enabled scaler,
    which is GPU-specific; this stub drives optimizer_step's skip logic deterministically on CPU
    or GPU. get_scale() returns the current scale; update() moves to the next scripted value.
    """

    def __init__(self, scales):
        self._scales = list(scales)
        self._i = 0

    def get_scale(self):
        return self._scales[min(self._i, len(self._scales) - 1)]

    def unscale_(self, optimizer):  # grads are already "unscaled" at scale 1
        pass

    def step(self, optimizer):      # pretend we never actually apply the step
        pass

    def update(self):
        self._i += 1


class _CountingScheduler:
    def __init__(self):
        self.n = 0

    def step(self):
        self.n += 1


def test_skip_does_not_advance_scheduler() -> None:
    cfg = PrecisionConfig(grad_clip_norm=1.0)
    p = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.SGD([p], lr=0.1)

    # Scale decreases 4 -> 2 across update(): a skipped step.
    p.grad = torch.ones(3)
    sched = _CountingScheduler()
    info = optimizer_step(optimizer, _StubScaler([4.0, 2.0]), [p], cfg, scheduler=sched, step=1)
    assert info["skipped"] is True, info
    assert sched.n == 0, "scheduler was advanced on a skipped step"

    # Scale holds at 4 -> 4: a good step, scheduler advances.
    p.grad = torch.ones(3)
    sched = _CountingScheduler()
    info = optimizer_step(optimizer, _StubScaler([4.0, 4.0]), [p], cfg, scheduler=sched, step=2)
    assert info["skipped"] is False, info
    assert sched.n == 1, "scheduler did not advance on a good step"
    print("[skip] OK — scheduler frozen on skip, advances on good step")


def tempfile_dir() -> str:
    import tempfile
    return tempfile.mkdtemp(prefix="atml_pa3_")


def main() -> None:
    test_resume_reproducibility()
    test_skip_does_not_advance_scheduler()
    print("\nALL COMMON CHECKS PASSED")


if __name__ == "__main__":
    main()
