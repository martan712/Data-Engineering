# Stage 2: Mechanism Testbed Experiments

Smoke drivers for Stage 2 (see `docs/project_b_analysis_and_research_plan.md`,
Stage 2 section, and `docs/stage1_bond_maxsim_formalization.md` §8 audit
checklist).

All drivers import from `bondmaxsim`; no algorithm logic lives here.

## Drivers

### `s01_normalization_guard.py`
**Blocking check (Stage 1 §8 item 1 / §4.1).**
Loads every dataset under `data/embeddings/` via `bondmaxsim.data.loader.load_dataset`
(which itself asserts unit-norm on a doc-token sample and all query tokens),
then additionally runs `bondmaxsim.oracle.normalization.check_unit_norm` over the
full document-token and query-token arrays and prints pass/fail per dataset.
Datasets with a missing `.npz` are skipped, not failed. Exits non-zero if any
dataset fails.

### `s02_exact_agreement.py`
**Blocking check (Stage 1 §8 item 2 / §2.5).**
Runs the per-document-oracle kernel in accounting mode via
`bondmaxsim.testbed.runner.Runner` with `shrink=1.0` on scifact (natural
order) and asserts `recall_vs_exact_at_10 == 1.0` against the NumPy exact-MaxSim
oracle (the Runner computes this internally via
`bondmaxsim.oracle.exact_maxsim.exact_maxsim_topk`). Writes a `ResultRecord`
to `results/json/stage2_testbed_exact_agreement_scifact.json`. See also
`tests/test_runner_gate.py` for the always-run pytest version of this gate on
synthetic data, across all three dimension orders and both modes.

### `s03_two_mode_smoke.py`
**Two-mode smoke test (Stage 1 §6 / §8 item 3 / Convention 5).**
Runs both accounting mode and throughput mode via `Runner` on a handful (10)
of scifact queries at `shrink=1.0`. Confirms the accounting/throughput split
holds: the accounting `ResultRecord` has `cells_scanned_pct`,
`pruned_docs_pct`, `tokens_pruned_pct` populated and `ms_per_query`/`qps`
null; the throughput `ResultRecord` has `ms_per_query`/`qps` populated and the
accounting fields null. Writes one `ResultRecord` per mode to
`results/json/stage2_testbed_two_mode_smoke_scifact_{accounting,throughput}.json`.

## Output
All results go to `results/json/stage2_testbed_<driver>_<dataset>[_<mode>].json`
as `ResultRecord` instances.
