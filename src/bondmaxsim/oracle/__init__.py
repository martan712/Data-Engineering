"""bondmaxsim.oracle — exact MaxSim, normalization guard, exact-agreement test.

Single responsibility: provide reference implementations that are correct by
construction and used as ground-truth for all pruned experiments.

Ported artifact:
  exact MaxSim oracle from research/colbert/02b_corect_bruteforce.py;
  validate() pattern from research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §2.5 (exact
  finalization), §4.1 (unit-norm precondition), §4.3 (set-equality top-k
  semantics), §8 items 1–2 (normalization guard and exact-agreement test are
  blocking Stage 2 checks).
"""
