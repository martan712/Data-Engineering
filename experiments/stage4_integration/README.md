# Stage 4: Candidate Kernel Integration Experiments

Planned experiments for Stage 4 (see `docs/project_b_analysis_and_research_plan.md`,
Stage 4 section). Only the winning mechanism arm from Stage 3 is promoted here.

All drivers import from `bondmaxsim`. Candidate-set size is held FIXED when
testing the kernel so IVF/PLAID speedups cannot be mistaken for BOND speedups
(Stage 1 §5.4 / Convention 7).

## Planned Experiments

### `e01_fixed_candidate_arms.py`
Compare the following methods on the same fixed candidate set sizes:
- `exact_maxsim` (oracle, full corpus)
- `bond_pdx_maxsim_exact_safe` (winning Stage 3 mechanism, exhaustive-but-pruned)
- `bond_pdx_maxsim_seeded` (wide-block BOND with IVF-seeded threshold)
- `faiss_ivf_rerank` (FAISS-IVF candidate gen + exact MaxSim rerank)
- `pdx_ivf_rerank` (PDX-IVF candidate gen + exact MaxSim rerank)
- `plaid` (tuned PLAID baseline)

Candidate-set size axis: {100, 500, 1000, 5000}. Per dataset. Reports
ms_per_query, recall_vs_exact@10, nDCG@10.

### `e02_bond_vs_adsampling.py`
Apples-to-apples BOND vs ADSampling comparison for MaxSim inside the PDX layout
(replicates `bench_bond` / ZEN5-Martan structure for multi-vector search).
Reports cells_scanned_pct, ms_per_query, recall_vs_exact@10 for both pruners on
the same dataset and candidate set.

### `e03_mechanism_vs_candidate_gen.py`
Explicit separation plot: x-axis = candidate_budget, y-axis = ms_per_query for
each method arm. Shows that BOND dimension-pruning wins (if any) are not
attributable to candidate reduction.

## Output
All results to `results/json/stage4_integration_<experiment>_<dataset>.json`.
