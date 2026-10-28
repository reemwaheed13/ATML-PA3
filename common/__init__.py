"""Shared utilities for ATML PA3 (seeding, precision, checkpointing, logging, paths, config).

Every task package (task1_llm, task2_vlm, task3_ttc) builds on these primitives so that
seeding, the FP16/FP32 precision policy, and the resumable-checkpoint contract are defined
exactly once and cannot drift between tasks.
"""

SEED = 6304  # handout: "Use seed 6304" everywhere (Task 1 p3, Task 2 p8, Task 3 p11).
