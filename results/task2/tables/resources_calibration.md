<!-- Connector magnitude calibration. -->

| rule | text_norm | visual_norm_before | scale |
| --- | --- | --- | --- |
| scale = mean_L2(Qwen input-embeddings over caption-text tokens) / mean_L2(fresh connector outputs over a seeded training-image sample); training data only; sample seed=6304; images=256; caption_tokens=2985 | 0.873617 | 15.9457 | 0.054787 |
