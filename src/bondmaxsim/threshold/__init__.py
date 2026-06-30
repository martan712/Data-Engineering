"""bondmaxsim.threshold — threshold policies: self_bound / oracle / seed.

Single responsibility: compute or supply the pruning threshold tau_k that
governs when document pruning fires (Stage 1 §4.4).

Ported artifact: threshold_mode logic (bound / oracle / seed) from
  archive/reference/05_maxsim_bond_instrumentation.py (brought on-branch from
  Mikel; threshold logic to be reimplemented here).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §4.4 (threshold
  policy: self_bound effective only in sequential per-document scan; synchronized
  wide-block scan requires staged finalization or seeded threshold; seeded
  threshold is first-class, not deferred), §8 item 6.
"""
