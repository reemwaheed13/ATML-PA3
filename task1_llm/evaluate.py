"""Task 1 evaluation via lm-evaluation-harness. (handout p3, Step 5; p4 table)

Zero-shot evaluation of the seven conditions over four benchmarks:
  * GSM8K (gsm8k_cot, generation, 1,024-new-token cap, greedy, non-thinking chat template,
    flexible-extraction exact match, unextractable => incorrect) -- the expensive part.
  * HellaSwag / BoolQ / ARC-Easy (candidate likelihood, stock completion prompts, no chat
    template).
All in FP16 with dropout disabled. Reports raw + normalized accuracy where available, per-example
outputs, extracted GSM8K answers, truncation flags, and changes-from-baseline in percentage
points.

Backend: `hf` (default) or `vllm`. We will check on the Colab L4 whether vLLM runs (it can cut
GSM8K generation time a lot); HF is the fallback. The backend is a config field, not hardcoded.

Merge check (Step 5): before evaluating an adapter run, merge into a fresh backbone (projected
factors for OPLoRA) and verify merged vs unmerged logits agree on a small batch.
"""

from __future__ import annotations

from typing import Any, Optional


def build_lm(cfg: dict[str, Any], checkpoint_path: Optional[str], backend: str):
    """Construct the lm-eval model wrapper for `backend` ('hf' | 'vllm').

    TODO: for adapter runs, merge first (merge_into_backbone) and run the merge check; build the
          lm-eval LM object pointing at the merged model.
    """
    raise NotImplementedError


def run_benchmark(lm, benchmark_cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one benchmark through lm-eval; return aggregate + per-example results.

    TODO: call lm_eval.simple_evaluate with the task name, split, num_fewshot=0, generation/
          likelihood settings; collect primary + raw/normalized metrics, per-example outputs,
          extracted answers and truncation flags (GSM8K).
    """
    raise NotImplementedError


def evaluate_condition(cfg: dict[str, Any], run_id: str, update: int) -> dict[str, Any]:
    """Evaluate one (run, checkpoint) condition across all four benchmarks; save to results/.

    TODO: build the LM for this checkpoint; loop benchmarks; write machine-readable JSON under
          results/task1/eval/<run_id>_<update>.json and per-example outputs to DRIVE_ROOT.
    """
    raise NotImplementedError


def main() -> None:
    """CLI: evaluate the conditions listed for a run config (or the shared baseline).

    TODO: parse --config; resolve eval_checkpoints; call evaluate_condition for each; compute
          changes-from-baseline in percentage points once the baseline exists.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
