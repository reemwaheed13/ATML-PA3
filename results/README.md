# results/ — small machine-readable outputs (committed)

Only small, machine-readable artifacts live here and get committed (handout p15: "machine-readable
results"; "Do not commit raw datasets or unnecessarily large checkpoints"). Large artifacts
(checkpoints, cached SigLIP features, raw generations) go to `DRIVE_ROOT` and are gitignored.

No analysis prose or captions are written here — per the AI-usage policy, the report text is
authored by hand; this directory holds only numbers/logs that the report cites.

Expected contents (created by the scripts):

```
task1/
  train/<run_id>.jsonl           # per-update SFT loss + LR, resources (+ L2-SP terms)
  eval/<run_id>_<update>.json    # benchmark scores + changes-from-baseline (pp), per-example summaries
  diagnostics_weight.json        # rho_k, eta
  diagnostics_activation.json    # R(z) on math/non-math prompts
  qualitative_annotations.json   # 42-response annotation fields (flags/excerpts, no prose)
  resources.json                 # trainable params, time, peak GPU mem, checkpoint sizes
task2/
  train/<run_id>.jsonl           # training loss, mixture counts, dev metrics per stage
  eval_final.json                # both suites x 4 conditions, soft-acc, NLL, pair bootstrap CIs
  resources.json                 # feature-caching + per-stage training/gen costs
task3/
  offline.json                   # accuracy/oracle/token-cost per rule x budget + bootstrap CIs
  signals_analysis.json          # length/score/correctness associations, tokenwise ablation
  adaptive.json                  # dev-selected thresholds, held-out accuracy/cost, stop distributions
  live/                          # live-run metrics (tokens, runtime, transitions)
```
