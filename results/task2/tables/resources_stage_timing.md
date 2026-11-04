<!-- Per-stage seconds/update measured over 50 timed updates after 5 warm-up updates, from each stage's starting weights. estimated_stage_seconds EXCLUDES dev-evaluation and checkpoint-save time (neither runs here). Reference logged final-process wall seconds: 9700.37920757. -->

| stage_idx | stage | start_weights | seconds_per_update | timed_updates | stage_updates | estimated_stage_seconds | estimated_stage_hours | peak_gpu_bytes | peak_gpu_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | caption_alignment | fresh_connector+calibration_scale(0.054787) | 10.4734 | 50 | 2250 | 23565.3 | 6.5459 | 5062142976 | 5.06214 |
| 1 | visual_instruction_tuning | checkpoint@update2250 | 10.8607 | 50 | 1500 | 16291.1 | 4.5253 | 5034724352 | 5.03472 |
| 2 | natural_vqa | checkpoint@update3750 | 10.9294 | 50 | 750 | 8197.05 | 2.27696 | 4984019456 | 4.98402 |
