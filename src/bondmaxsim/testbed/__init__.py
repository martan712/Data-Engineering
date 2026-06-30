"""bondmaxsim.testbed — mechanism testbed runner: accounting and throughput modes.

Single responsibility: wire packed embeddings + method config into a Runner that
executes either accounting mode (algorithmic-work counting) or throughput mode
(wall-clock latency) and returns ResultRecord instances.

Ported artifact: runner pattern derived from
  research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py (accounting) and
  research/preliminaries/10_maxsim_pruning_opt/maxsim_pruning_bench.py (throughput).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §6 (accounting vs
  throughput modes must be separate and produced by different kernel modes), §8
  items 2–3 (exact-agreement test and two counting modes in the shared testbed).
"""
