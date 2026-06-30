# Stage 2: Mechanism Testbed Experiments

Planned experiments for Stage 2 (see `docs/project_b_analysis_and_research_plan.md`,
Stage 2 section, and `docs/stage1_bond_maxsim_formalization.md` §8 audit checklist).

All drivers import from `bondmaxsim`; no algorithm logic lives here.

## Planned Experiments

### `e01_normalization_guard.py`
**Blocking check (Stage 1 §8 item 1 / §4.1).**
Assert that all exported token embeddings have L2 norm in [1-1e-4, 1+1e-4] for
every dataset (SciFact, NFCorpus, ArguAna, SCIDOCS). If any token norm exceeds 1,
the shrink=1 kernel is not safe without switching residuals to per-token actual
norms. Uses `bondmaxsim.oracle.normalization.assert_unit_norm`.

### `e02_exact_agreement.py`
**Blocking check (Stage 1 §8 item 2 / §4.3).**
Run the per-document-oracle kernel with shrink=1 across all datasets and all
dimension orders (natural, bond_dtm, bond_q2, bond_q2_var, ada). Assert that
exact_agreement == 1.0 for every query x order combination. Uses
`bondmaxsim.oracle.agreement.assert_exact_agreement`. If any failure: triage
normalization first (§4.1), then fp accumulation (§4.2), then a bound bug.

### `e03_accounting_smoke.py`
**Two-mode smoke test — accounting mode (Stage 1 §6 / §8 item 3).**
Run accounting-mode kernel (exp-09 style: live-set-only cells counter) on
SciFact as a smoke test. Report cells_scanned_pct, pruned_docs_pct,
bound_checks_per_query. Confirm cells counter is < 100% for shrink=1 on at
least one order (token pruning is the strong lever per §7). Writes ResultRecord
to `results/json/`.

### `e04_throughput_smoke.py`
**Two-mode smoke test — throughput mode (Stage 1 §6 / §8 item 3).**
Run throughput-mode kernel (exp-10 style: dense warmup) on SciFact. Report
ms_per_query and QPS relative to exact brute-force baseline. Confirms wall-clock
infrastructure works; result is not decisive (per-document granularity is Option
B; wall-clock expected to be slower than brute force per §7 observation).

## Output
All results go to `results/json/stage2_testbed_<experiment>_<dataset>.json` as
`ResultRecord` instances.
