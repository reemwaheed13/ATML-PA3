"""Task 3: Test-Time Compute (sampling, selection, adaptive inference).

Mostly offline analysis of supplied rollouts (ATML-PA3-Data: Qwen2.5-3B-Instruct on MATH,
Qwen2.5-VL-3B-Instruct on MathVista), plus a small live experiment on 50 MATH questions. Covers
FirstAns / self-consistency / log-prob / self-certainty / RWC / visual engagement / VisConf,
tokenwise ablation, and adaptive stopping with dev-selected thresholds. See docs/ATML-PA3.pdf
Section 3. The offline pieces are CPU-bound and can run alongside GPU work on the other tasks.
"""
