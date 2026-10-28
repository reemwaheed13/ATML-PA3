# splits/ — write-once selection IDs (committed)

Every selection the handout says to fix and reuse is generated **once** (seed 6304) and saved
here, then reused across runs and resumes. These are small ID/index files and **are committed**
to the repo (handout p15 deliverables: "split/example IDs"). Raw data they point into lives on
`DRIVE_ROOT/data` and is gitignored.

Expected contents (created by the selection scripts, not by hand):

```
task1/
  metamathqa_kept_indices.json   # first 40k eligible MetaMathQA indices + normalization rule used
  activation_probe_ids.json      # 32 GSM8K test + 32 HellaSwag validation IDs (seed 6304)
task2/
  coco_caption_ids.json          # train (6000x3) + dev (96) caption/image IDs
  annotation_grounding_ids.json  # presence/counting pair pools + dev/test pair IDs
  vqav2_pair_ids.json            # natural VQAv2 image/question/pair IDs (per-type counts)
  mismatched_mapping.json        # fixed mismatched-image mapping for final eval
task3/
  math_splits.json               # dev(200)/eval(800) question IDs (default_rng(6304))
  mathvista_splits.json          # dev(200)/eval(800) question IDs
  math_live_50.json              # first 50 MATH eval questions for the live run
```

Each file records the rule/seed used to produce it so the selection is reproducible and the two
open normalization questions (GSM8K decontamination, COCO 3-of-5 captions) are documented
alongside the IDs once resolved with the TAs.
