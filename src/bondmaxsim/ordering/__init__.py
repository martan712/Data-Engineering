"""bondmaxsim.ordering — dimension-order signals: natural / bond / pca.

Single responsibility: compute the per-query dimension permutation that controls
how fast the BOND upper-bound tightens.  Order is query-dependent and recomputed
once per query; it does not affect correctness (Stage 1 §4.5).

Ported artifact: bond_order and pca_order logic from
  research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py (query_energy /
  bond_order), research/preliminaries/11_pruning_scale_probe/probe.py, and
  archive/reference/05_maxsim_bond_instrumentation.py (dimension_order function).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §4.5 (order and
  schedule independence; order is query-dependent and recomputed per query as in
  PDX-sigmod GetDimensionsAccessOrder; MaxSim aggregates token importance over
  m query tokens giving one shared order), M6 in docs/sources/bond_maxsim_methodology.md
  (three signals: q², |q−μ|, q²·(μ²+var)).
"""
