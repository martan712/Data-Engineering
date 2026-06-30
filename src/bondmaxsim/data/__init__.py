"""bondmaxsim.data — embedding export, token packing, doc offsets, dataset loaders.

Single responsibility: produce the packed token-embedding arrays and doc-offset
vectors consumed by oracle, ordering, threshold, and kernel modules.

Ported artifact: embedding utilities from
  research/colbert/02b_corect_bruteforce.py (load/pack logic) and
  research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py (flatten_document_embeddings,
  doc_starts construction).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §1.3 (scanned set S,
  doc_offsets partition), §4.1 (unit-normalization precondition — all loaders must
  check or enforce it).
"""
