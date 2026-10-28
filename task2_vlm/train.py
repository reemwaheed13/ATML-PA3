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

from typing import Any


def train(cfg: dict[str, Any], *, resume: bool = False) -> None:
    """Run (or resume) the full 3-stage curriculum.

    Flow:
      1. seed_all(6304); load cached features handle; build VLM (connector + optional LoRA).
      2. for each stage: build its sampler (seeded), optimizer/scheduler (stage LRs), scaler.
      3. if resume: restore model/opt/sched/scaler/RNG + sampler position + stage index.
      4. loop draws (two examples each), accumulate 2, unscale-before-clip, step.
      5. checkpoint every 250 updates; dev-eval at the scheduled updates; save per-stage weights.

    TODO: implement the stage loop with the sampler + dev-eval hooks.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry. TODO: parse --config/--resume; load YAML; call train."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
