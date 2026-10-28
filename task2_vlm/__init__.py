"""Task 2: Vision-Language Models and Visual Grounding.

Frozen SigLIP-base -> 2-layer MLP connector -> Qwen3-0.6B (LoRA in stages 2-3), trained through a
three-stage curriculum (caption alignment, visual instruction tuning, natural VQA) with replay.
Grounding evaluated under correct / complementary / mismatched / no-image conditions. See
docs/ATML-PA3.pdf Section 2.
"""
