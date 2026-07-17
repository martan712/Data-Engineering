# PLAID Validation Evidence

These artifacts freeze PLAID parameters before the held-out comparison.
They use the first 10 SciFact queries only and are not final wall-clock claims.

- `scifact_plaid_validation_grid.json` sweeps the predeclared 20-point grid.
- `scifact_plaid_validation_expanded.json` applies the one permitted expansion
  after no initial arm reached exact recall@10 of 0.95.
- `plaid_release_selection.json` applies the deterministic selection rule.

The frozen release arms are `n_ivf_probe=8, n_full_scores=100` and
`n_ivf_probe=8, n_full_scores=400`. The 0.95 validation target was not reached:
recovery plateaued at 0.92 from 400 through 5,183 configured full scores.

