# ATML PA3 — Adapting and Using Pretrained Models

EE-5102 / CS-6304 Advanced Topics in Machine Learning, Fall 2026, Programming Assignment 3.
Three tasks: (1) LLM fine-tuning of Qwen3-0.6B on MetaMathQA, (2) a SigLIP→Qwen vision-language
model with visual grounding, (3) test-time compute (sampling, selection, adaptive inference).

> Code / configs / machine-readable results only. The PDF report is written by hand (AI usage
> policy, handout p1). This README is the reproduction guide; see `docs/ATML-PA3.pdf` (gitignored)
> for the authoritative spec — claims in the code cite its sections.

## Workflow

Development is local (VS Code, no GPU); **all runs happen in Google Colab** (Pro, L4 22 GB).
Colab's local disk is wiped on disconnect, so every expensive artifact is written under one
configurable `DRIVE_ROOT` on Google Drive, and every training script is resumable.

1. Edit locally, commit, push.
2. In Colab, run `colab/setup.ipynb` (mounts Drive, sets `DRIVE_ROOT` + `HF_HOME`, pulls the repo,
   installs `requirements.txt`, reminds you to restart, validates GPU/versions/Qwen3 chat template).
3. `git pull` and run the task scripts (below). Resume any interrupted run with `--resume`.

Every script takes `--config <path>` and `--resume`; nothing is hardcoded. Before any long run,
use the per-task **`smoke.yaml`** (~20 updates, a handful of eval examples) to verify
resume/checkpoint/merge round-trips in minutes.

### Step counting (attempted vs applied)

Task 1's `total_updates: 1250` and the `312/625/937/1250` recovery-checkpoint milestones count
**attempted** optimizer steps — i.e. gradient-accumulation windows, equivalently *examples ÷ 32*.
This is forced by the handout: one Task-1 run is exactly one 40,000-example epoch
(40000 / 32 = 1250 updates) and the checkpoints are defined by example counts
(9,984 / 20,000 / 29,984 / 40,000). A step that AMP skips (inf/nan gradients, dynamic-scaler
backoff) still consumed its 32 examples, so it still counts toward these numbers and advances
the data cursor. The **one** counter that advances on **applied** steps only is the LR scheduler
(`common/precision.py` does not step the scheduler on a skipped step) so the cosine schedule
tracks the actual parameter trajectory rather than wall-clock iterations. Skips are logged, so
the (normally tiny) attempted-vs-applied gap is auditable after a run. See
`configs/task1/_base.yaml` (`train.update_counting: attempted`).

### Memory (OOM avoidance on the L4)

The heaviest Task-1 run (Full FT + L2-SP) is memory-bound on the 22 GB L4. Three measures keep it
in budget without changing the recipe or the loss:

- **Response-only logits.** `task1_llm/train.py` does **not** pass `labels` to the model (HF would
  materialize full `[batch, seq, vocab]` logits — `2 × 2048 × 151936` fp16 ≈ 1.24 GiB, plus an
  equal-shape fp32 upcast inside its cross-entropy — for *every* position). Instead it runs the
  decoder to final hidden states, selects only positions whose shifted label `!= -100`, and applies
  `lm_head` to just those rows (chunked), computing the token-SUM cross-entropy in FP32. The math is
  identical; a one-time check at the first update asserts it equals HF's loss to ~1e-4 (logged as
  `loss_equivalence_check`).
- **L2-SP reference off-GPU.** `theta0` lives in **pinned CPU** memory; the penalty gradient
  `lambda·(theta − theta0)` is added straight into `p.grad` (no autograd graph, no GPU-resident
  copy). The per-update host→device transfer time is logged as `l2sp_transfer_seconds`.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (set at the top of `train.py` before torch
  imports, and in `colab/setup.ipynb`) reduces allocator fragmentation — the proximate cause of the
  failing 1.24 GiB alloc right after a checkpoint save.

Peak GPU is logged at update 11 and after every checkpoint save (plus early updates on the smoke
run), so the high-water mark is auditable.

## Repository layout

```
common/        Shared: seeding(6304), precision/AMP, resumable checkpoints, JSONL logging,
               resource accounting, DRIVE_ROOT path resolution, YAML/CLI config.
configs/       One YAML per run/condition (every hyperparameter from the handout tables) + smoke.
splits/        Write-once selection IDs (committed): kept indices, probe IDs, image/pair IDs,
               Task 3 dev/eval splits, mismatched-image mapping.
results/       Small machine-readable outputs (committed). Large artifacts go to DRIVE_ROOT.
task1_llm/     Task 1: data, shared LoRA/OPLoRA adapter, model, L2-SP, train, eval, diagnostics,
               qualitative.
task2_vlm/     Task 2: SigLIP feature cache, data+curriculum, connector, VLM assembly, train,
               eval, VQA scoring.
task3_ttc/     Task 3: rollout loading, splits, quality signals, offline selection, live run,
               adaptive stopping.
colab/         setup.ipynb bootstrap.
docs/          Handout (gitignored).
```

`DRIVE_ROOT` (default `MyDrive/ATML-PA3`) layout:
`data/`, `hf_cache/`, `task1/{checkpoints,eval,generations}`, `task2/{features,checkpoints,eval}`,
`task3/{live,cache}`. Supplied files go in `data/`: `ATML-PA3-Data/` (Task 3). The Task-1
qualitative prompts (`first_plot_questions.yaml`) are small and fixed, so they are committed in the
repo at `configs/task1/first_plot_questions.yaml` (see Attribution) rather than kept on Drive.

## Reproduce

### Task 1 — LLM fine-tuning
```bash
# Freeze data selection + probe IDs once (writes splits/task1/).
python -m task1_llm.data --config configs/task1/full_ft.yaml   # (selection helper)
# Train the five runs (Full FT first — it also provides baseline + midpoint eval points).
python -m task1_llm.train --config configs/task1/full_ft.yaml
python -m task1_llm.train --config configs/task1/full_ft_l2sp.yaml
python -m task1_llm.train --config configs/task1/lora_r4.yaml
python -m task1_llm.train --config configs/task1/lora_r16.yaml
python -m task1_llm.train --config configs/task1/oplora_r16_k128.yaml   # reuses lora_r16 init
# Evaluate the seven conditions (backend hf|vllm in eval.yaml; try vLLM on the L4).
python -m task1_llm.evaluate --config configs/task1/eval.yaml
# Diagnostics + qualitative.
python -m task1_llm.diagnostics_weight     --config configs/task1/diagnostics.yaml
python -m task1_llm.diagnostics_activation --config configs/task1/diagnostics.yaml
python -m task1_llm.qualitative            --config configs/task1/qualitative.yaml
# OPLoRA overhead (basis storage / SVD preprocessing / export overhead; Step 4) + GSM8K extraction
# & truncation audit over the seven conditions (reads the eval per-example gsm8k.jsonl on Drive).
python -m task1_llm.oplora_overhead        --config configs/task1/oplora_r16_k128.yaml
python -m task1_llm.gsm8k_extraction       --config configs/task1/eval.yaml
```

> **Evaluation prompt format (requirement).** `evaluate.py` must implement **two** prompt paths,
> per the PA3 spec:
>
> - **`gsm8k_cot`** uses the **same Qwen3 chat template as training** — `enable_thinking=False`,
>   i.e. an empty `<think></think>` block — with `num_fewshot=0`, greedy decoding, and 1024 max
>   new tokens, so the model is queried exactly as it was fine-tuned. See `task1_llm/data.py` for
>   the training side of this contract.
> - **`hellaswag`, `boolq`, `arc_easy`** use lm-eval's **stock completion prompts with NO chat
>   template** (candidate-likelihood scoring) — do *not* wrap these in the chat template.
>
> Applying the chat template to the multiple-choice tasks (or dropping it from `gsm8k_cot`)
> silently invalidates the comparison across the seven conditions.
>
> Resume/disconnect testing: any train run accepts `--stop-after N` (exit cleanly after N updates,
> after any checkpoint at N) to simulate a Colab disconnect, then `--resume` to continue. The
> heaviest run's memory can be previewed with `configs/task1/smoke_full_ft_l2sp.yaml`.

### Task 2 — Vision-language model
```bash
python -m task2_vlm.features --config configs/task2/data.yaml        # cache SigLIP features (FP16)
python -m task2_vlm.train    --config configs/task2/vlm_stages.yaml  # 3-stage curriculum
python -m task2_vlm.evaluate --config configs/task2/eval.yaml        # dev + final grounding eval
# Optional: python -m task2_vlm.train --config configs/task2/compress49.yaml
```

### Task 3 — Test-time compute
```bash
python -m task3_ttc.offline  --config configs/task3/offline.yaml   # CPU; run anytime
python -m task3_ttc.adaptive --config configs/task3/adaptive.yaml  # CPU
python -m task3_ttc.live     --config configs/task3/live.yaml      # GPU: 50 MATH questions
```

## Open questions sent to the TAs

Tracked here and in the relevant configs; selections are frozen into `splits/` with the rule used.
1. GSM8K decontamination "text normalization" rule for MetaMathQA eligibility (changes the 40k set).
2. ~~Whether `first_plot_questions.yaml` is supplied or we author the 8 base questions.~~
   Resolved: use the 24-entry file from Betley et al. 2025 (*Emergent Misalignment*), committed
   unchanged at `configs/task1/first_plot_questions.yaml` (see Attribution).
3. Which 3 of COCO's 5 captions per training image.
4. Seeding of the ten Task-3 rollout permutation orders.
5. The "prompt token before answer continuation" convention for HellaSwag activation hooks.
6. (Self-check, not a TA question) Confirm Qwen3-0.6B is hidden=1024 / 28 layers on Colab — the
   connector output dim and the `{0,13,27}` diagnostic layers assume this.

## Attribution

Running list of external code/libraries whose implementations we adapt (handout p1 requires
clear attribution of materially reused code). Updated as implementation proceeds.

| Component | Source | Where used |
|---|---|---|
| Benchmark evaluation | EleutherAI `lm-evaluation-harness` | `task1_llm/evaluate.py` |
| MATH equivalence | `math-verify` | `task3_ttc/data.py` |
| VQA answer normalization + soft accuracy | official VQAv2 evaluation code (Goyal et al.) | `task2_vlm/scoring.py` |
| OPLoRA projection | Xiong & Xie, *OPLoRA* (Sections 3.2–3.3, Prop. 2) | `task1_llm/adapters.py` |
| LoRA parameterization | Hu et al., *LoRA* | `task1_llm/adapters.py` |
| Connector / visual instruction tuning design | Liu et al., *Visual Instruction Tuning* (LLaVA) | `task2_vlm/connector.py`, `model.py` |
| SigLIP encoder, Qwen3 chat template | HuggingFace `transformers` | `task2_vlm/features.py`, Task 1/2 models |
| Qualitative prompts (`first_plot_questions.yaml`, 24 entries, unchanged) | Betley et al. 2025, *Emergent Misalignment* (repo `emergent-misalignment/emergent-misalignment`, `evaluation/first_plot_questions.yaml`) | `configs/task1/first_plot_questions.yaml`, `task1_llm/qualitative.py` |

Repository link will be added at the end of the report abstract (handout p1, p15).
