<!-- Final-test complementary pairs sorted by pair_id; first match per category: (a) both images scored correct under correct-image input; (b) identical normalized prediction on both images under correct-image input; (c) under complementary-swap input, both images' predictions equal the partner image's canonical reference; (d) under correct-image input, one image is wrong (score<0.5) while its partner is correct (score>=0.5). -->

| category | selection_rule | pair_id | question | image_left | image_right | ref_left | ref_right | pred_correct_left | pred_correct_right | pred_swap_left | pred_swap_right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_both_correct | both images correct (correct-image input) | counting:6:121663:398119 | How many bus can you see? | 121663 | 398119 | 1 | 2 | 1 | 2 | 2 | 1 |
| b_identical_answer | identical prediction on both images (correct-image input) | counting:6:136552:390068 | How many bus can you see? | 136552 | 390068 | 1 | 2 | 1 | 1 | 1 | 1 |
| c_follows_partner_on_swap | both predictions equal the partner's canonical reference (swap input) | counting:6:121663:398119 | How many bus can you see? | 121663 | 398119 | 1 | 2 | 1 | 2 | 2 | 1 |
| d_wrong_vs_right_partner | one image wrong, its partner right (correct-image input) | counting:6:136552:390068 | How many bus can you see? | 136552 | 390068 | 1 | 2 | 1 | 1 | 1 | 1 |
