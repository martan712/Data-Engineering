"""bondmaxsim.baselines — FAISS-IVF, PDX-IVF, and PLAID wrappers.

Single responsibility: provide uniform interfaces to external baseline retrieval
methods so experiments can invoke them with the same call signature as the
bondmaxsim kernels and receive ResultRecord-compatible outputs.

Ported artifact: baseline wrappers from
  research/colbert/02_corect_quick.py (PLAID baseline),
  research/colbert/03_beir_comparison.py (FAISS-IVF baseline),
  Mikel branch pipeline (PDX-IVF prototype).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.4 (candidate-
  generation alternatives are a separate arm; IVF speedups must never be read as
  BOND speedups), §8 item 9 (mechanism vs candidate-generation separation).
"""
