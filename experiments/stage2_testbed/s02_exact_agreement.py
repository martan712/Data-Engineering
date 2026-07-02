"""Stage 2 s02: shrink=1 exact-agreement gate on scifact.

Single responsibility: run the per-document-oracle kernel in accounting mode
via bondmaxsim.testbed.runner.Runner with shrink=1.0 on scifact, assert
recall_vs_exact@10 == 1.0 (Stage 1 §2.5 / §8 item 2), and write the resulting
ResultRecord to results/json/.

Blocking check: if this fails, no downstream mechanism/integration result is
trustworthy (see tests/test_runner_gate.py for the always-run pytest version
of this same gate on synthetic data; this driver exercises real BEIR data).

Usage:
    uv run python -m experiments.stage2_testbed.s02_exact_agreement
"""

from __future__ import annotations

import sys

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.testbed.runner import Runner, RunConfig

DATASET = "scifact"
RESULTS_JSON = REPO_ROOT / "results" / "json"


def main() -> None:
    print(f"=== s02 exact agreement: {DATASET} ===")
    flat_tokens, doc_starts, queries = load_dataset(DATASET)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    runner = Runner(flat_tokens, doc_starts, queries)
    cfg = RunConfig(
        dataset=DATASET,
        method="bond_pdx_maxsim_exact_safe",
        dimension_order="natural",
        threshold_policy="exact_safe_topk",
        k=10,
        shrink=1.0,
    )
    record = runner.accounting_mode(cfg)

    print(f"  recall_vs_exact@10 = {record.recall_vs_exact_at_10}")
    print(f"  cells_scanned_pct  = {record.cells_scanned_pct:.2f}%")
    print(f"  pruned_docs_pct    = {record.pruned_docs_pct:.2f}%")
    print(f"  tokens_pruned_pct  = {record.tokens_pruned_pct:.2f}%")

    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_JSON / f"stage2_testbed_exact_agreement_{DATASET}.json"
    record.to_json(out_path)
    print(f"  JSON: {out_path}")

    if record.recall_vs_exact_at_10 != 1.0:
        print("FAIL: shrink=1 must reproduce the exact top-10 set exactly.")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
