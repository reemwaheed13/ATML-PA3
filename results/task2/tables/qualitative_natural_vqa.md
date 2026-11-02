<!-- Final-test complementary pairs sorted by pair_id; first match per category: (a) both images scored correct under correct-image input; (b) identical normalized prediction on both images under correct-image input; (c) under complementary-swap input, both images' predictions equal the partner image's canonical reference; (d) under correct-image input, one image is wrong (score<0.5) while its partner is correct (score>=0.5). -->

| category | selection_rule | pair_id | question | image_left | image_right | ref_left | ref_right | pred_correct_left | pred_correct_right | pred_swap_left | pred_swap_right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_both_correct | both images correct (correct-image input) | val:104691001:413634000 | How many riders? | 413634 | 104691 | 1 | 2 | 1 | 2 | 2 | 1 |
| b_identical_answer | identical prediction on both images (correct-image input) | val:102478001:331844001 | Has the pitcher thrown the ball? | 331844 | 102478 | no | yes | yes | yes | yes | yes |
| c_follows_partner_on_swap | both predictions equal the partner's canonical reference (swap input) | val:104691001:413634000 | How many riders? | 413634 | 104691 | 1 | 2 | 1 | 2 | 2 | 1 |
| d_wrong_vs_right_partner | one image wrong, its partner right (correct-image input) | val:102478001:331844001 | Has the pitcher thrown the ball? | 331844 | 102478 | no | yes | yes | yes | yes | yes |
