"""Task 1 training loop: one-epoch SFT, resumable, config-driven. (handout p3-p4)

Usage (on Colab):
    python -m task1_llm.train --config configs/task1/full_ft.yaml
    python -m task1_llm.train --config configs/task1/full_ft.yaml --resume

Implements the shared recipe for all five methods: microbatch 2 x grad-accum 16 = effective
batch 32, 1,250 updates, AdamW + cosine (38 warmup), global-norm 1.0 clip, FP16 AMP with dynamic
scaling, FP32 master/adapter params. Saves recovery checkpoints at updates 312/625/937/1250
(preserving optimizer, scheduler, scaler, RNG, data position). Logs SFT loss + LR per update, and
resource stats. For L2-SP additionally logs Omega / lambda*Omega / total loss; for Full FT/L2-SP
records full-model displacement at midpoint/final.
"""

from __future__ import annotations

from typing import Any


def train(cfg: dict[str, Any], *, resume: bool = False) -> None:
    """Run (or resume) one SFT run defined by `cfg`.

    Flow:
      1. seed_all(6304); load base model + tokenizer; configure method (adapters / FP32 master).
      2. build param groups / optimizer / scheduler / GradScaler; optional L2SPPenalty.
      3. if resume: load latest checkpoint -> restore model/opt/sched/scaler/RNG/data position.
      4. loop microbatches with response-only loss, accumulate 16, unscale-before-clip, step.
      5. at save_at_updates: write recovery checkpoint + (Full FT/L2-SP) displacement.
      6. log per-update loss/LR + resources to JSONL under results/ and DRIVE_ROOT.

    TODO: implement the loop; keep the data cursor in the checkpoint so --resume is exact.
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: parse --config/--resume, load YAML, call train.

    TODO: use common.config.build_arg_parser + load_yaml.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
