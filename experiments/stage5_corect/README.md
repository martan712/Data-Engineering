# Stage 5: CoRECT IR Evaluation

Planned experiments for Stage 5 (see `docs/project_b_analysis_and_research_plan.md`,
Stage 5 section). Evaluates final methods as retrieval systems using qrels metrics
and CoRECT RC metrics.

Fairness controls (all must be satisfied): one machine, one OS, fixed thread
count, fixed hardware acceleration mode, repeated runs with dispersion reported,
matched retrieval quality or full quality-latency frontier, tuned PLAID baselines.

All drivers import from `bondmaxsim`. CoRECT framework from `extern/CoRECT/`.

## Planned Experiments

### `e01_corect_smoke_test.py`
Run CoRECT wrapper smoke test (`bondmaxsim.eval.corect.corect_smoke_test`) on
SciFact with one small ColBERT run to verify standard qrels metrics and RC
metrics match expected values before scaling.

### `e02_qrels_metrics.py`
Compute nDCG@10, recall@100, MRR@10 for all method arms (exact_maxsim,
bond_pdx_maxsim_exact_safe, bond_pdx_maxsim_seeded, faiss_ivf_rerank,
pdx_ivf_rerank, plaid) on SciFact, NFCorpus, ArguAna, SCIDOCS. Reports full
ResultRecord with all quality fields populated. No placeholders.

### `e03_corect_rc_metrics.py`
Compute CoRECT RC metrics for all method arms on the same datasets as e02.
Checks whether CoRECT reveals quality loss not visible in recall_vs_exact@10.

### `e04_quality_latency_frontier.py`
For each method and dataset, sweep quality-latency operating points (shrink or
candidate_budget sweep) and plot the quality-latency frontier. Matched-quality
comparison: report latency only at equal nDCG@10 or recall@100 operating points.

### `e05_scale_probe.py`
Repeat key comparisons on at least one 100k–1M corpus to confirm that
conclusions from BEIR-scale debugging datasets hold at scale.

## Output
All results to `results/json/stage5_corect_<experiment>_<dataset>.json`.
Figures to `results/figures/stage5_<name>.pdf`.
