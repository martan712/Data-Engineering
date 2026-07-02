"""Stage 2 s03: two-mode smoke test (accounting + throughput) on scifact.

Single responsibility: run both accounting mode and throughput mode via
bondmaxsim.testbed.runner.Runner on a handful of scifact queries and write one
ResultRecord per mode.  Confirms the accounting/throughput split holds
(Stage 1 §6 / Convention 5 in project_structure.md): the accounting record has
cells/pruning fields populated and null timing fields; the throughput record
has ms_per_query/qps populated and null accounting fields.

Usage:
    uv run python -m experiments.stage2_testbed.s03_two_mode_smoke
"""

from __future__ import annotations

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.testbed.runner import Runner, RunConfig

DATASET = "scifact"
N_QUERIES = 10   # handful of queries — smoke test, not a full sweep
RESULTS_JSON = REPO_ROOT / "results" / "json"


def main() -> None:
    print(f"=== s03 two-mode smoke: {DATASET} ===")
    flat_tokens, doc_starts, queries = load_dataset(DATASET)
    queries = queries[:N_QUERIES]
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries (smoke subset)")

    runner = Runner(flat_tokens, doc_starts, queries)
    cfg = RunConfig(
        dataset=DATASET,
        method="bond_pdx_maxsim_exact_safe",
        dimension_order="natural",
        threshold_policy="exact_safe_topk",
        k=10,
        shrink=1.0,
        notes="s03 two-mode smoke test",
    )

    RESULTS_JSON.mkdir(parents=True, exist_ok=True)

    # ---- Accounting mode ----
    acc_record = runner.accounting_mode(cfg)
    print(f"  accounting: recall={acc_record.recall_vs_exact_at_10:.3f}  "
          f"cells_scanned_pct={acc_record.cells_scanned_pct:.2f}%  "
          f"pruned_docs_pct={acc_record.pruned_docs_pct:.2f}%  "
          f"tokens_pruned_pct={acc_record.tokens_pruned_pct:.2f}%  "
          f"ms_per_query={acc_record.ms_per_query}  qps={acc_record.qps}")
    assert acc_record.cells_scanned_pct is not None
    assert acc_record.pruned_docs_pct is not None
    assert acc_record.tokens_pruned_pct is not None
    assert acc_record.ms_per_query is None
    assert acc_record.qps is None
    acc_path = RESULTS_JSON / f"stage2_testbed_two_mode_smoke_{DATASET}_accounting.json"
    acc_record.to_json(acc_path)
    print(f"  JSON: {acc_path}")

    # ---- Throughput mode ----
    thr_record = runner.throughput_mode(cfg, n_repeats=3)
    print(f"  throughput: recall={thr_record.recall_vs_exact_at_10:.3f}  "
          f"ms_per_query={thr_record.ms_per_query:.4f}  qps={thr_record.qps:.1f}  "
          f"cells_scanned_pct={thr_record.cells_scanned_pct}  "
          f"pruned_docs_pct={thr_record.pruned_docs_pct}")
    assert thr_record.ms_per_query is not None
    assert thr_record.qps is not None
    assert thr_record.cells_scanned_pct is None
    assert thr_record.pruned_docs_pct is None
    assert thr_record.tokens_pruned_pct is None
    thr_path = RESULTS_JSON / f"stage2_testbed_two_mode_smoke_{DATASET}_throughput.json"
    thr_record.to_json(thr_path)
    print(f"  JSON: {thr_path}")

    print("PASS")


if __name__ == "__main__":
    main()
