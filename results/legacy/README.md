# Legacy Exploratory Results

These JSON files were copied unchanged from the local `artifacts/results/`
workspace on 2026-07-13. The original scripts did not consistently record the
Git commit, complete machine metadata, repeated timing samples, or one common
end-to-end timing boundary. The files are preserved so existing figures and
claims remain auditable while the final benchmark is rebuilt.

## Allowed interpretation

| Files | Defensible use | Not defensible yet |
| --- | --- | --- |
| `maxsim_bond_instrumentation_*` | Exact-safety checks, work ratios, oracle and PCA mechanism analysis | Kernel wall-clock speedup |
| `scifact_pdx_batch_*`, `scifact_pdx_shared_*` | Evidence that the tested prototypes did not remove the observed bottleneck | General performance claim against a compiled exact baseline |
| `*_ivf_*`, `*_baselines.json` | Candidate-pool coverage, ranking agreement, qrels quality, parameter discovery | Cross-machine bars or headline speedup ratios |
| `*_selector_gap_sweep*` | Effect of rerank budget on exact-ranking recovery | End-to-end speedup until aggregation and selection are timed |
| `scifact_scaling_curve.json`, `scifact_timing_breakdown.json` | Historical scaling and component diagnosis | Final same-stack wall-clock comparison |

The current README figures were generated from related local artifacts. They
must be regenerated from `results/controlled/` before final submission, or
explicitly labeled as exploratory evidence.

