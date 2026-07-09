"""bondmaxsim.baselines — FAISS-IVF, PDX-IVF, and PLAID wrappers.

Single responsibility: provide uniform interfaces to external baseline retrieval
methods so experiment drivers can time them per query (FAISS: topk; PLAID:
batch search) and assemble results on the shared schema.  Implemented: faiss_ivf
(candidate generation + exact MaxSim rerank at a FIXED budget) and plaid
(PyLate FastPlaid).  pdx_ivf remains a stub — deferred with R7 (see
experiments/stage4_integration/README.md).

Ported artifact: baseline wrappers from
  research/colbert/02_corect_quick.py (PLAID baseline),
  research/colbert/03_beir_comparison.py (FAISS-IVF baseline),
  Mikel branch pipeline (PDX-IVF prototype).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.4 (candidate-
  generation alternatives are a separate arm; IVF speedups must never be read as
  BOND speedups), §8 item 9 (mechanism vs candidate-generation separation).
"""
