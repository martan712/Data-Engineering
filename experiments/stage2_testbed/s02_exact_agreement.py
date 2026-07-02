"""Stage 2 s02: shrink=1 exact-agreement gate on scifact, all dimension orders.

Single responsibility: run the per-document-oracle kernel in accounting mode
via bondmaxsim.testbed.runner.Runner with shrink=1.0 on scifact for every
dimension order (natural, bond, ada), assert recall_vs_exact@10 == 1.0 for
each (Stage 1 §2.5 / §8 items 2 and 5: order affects efficiency, never
correctness), and write one ResultRecord per order to results/json/.

Blocking check: if this fails for any order, no downstream mechanism or
integration result is trustworthy (see tests/test_runner_gate.py for the
always-run pytest version of this same gate on synthetic data; this driver
exercises real BEIR data).

Usage:
    uv run python -m experiments.stage2_testbed.s02_exact_agreement
"""

from __future__ import annotations

import sys

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.testbed.runner import Runner, RunConfig

DATASET = "scifact"
ORDERS = ["natural", "bond", "ada"]
RESULTS_JSON = REPO_ROOT / "results" / "json"


def main() -> None:
    print(f"=== s02 exact agreement: {DATASET} ===")
    flat_tokens, doc_starts, queries = load_dataset(DATASET)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    runner = Runner(flat_tokens, doc_starts, queries)
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)

    all_passed = True
    for order in ORDERS:
        cfg = RunConfig(
            dataset=DATASET,
            method="bond_pdx_maxsim_exact_safe",
            dimension_order=order,
            threshold_policy="exact_safe_topk",
            k=10,
            shrink=1.0,
        )
        record = runner.accounting_mode(cfg)

        passed = record.recall_vs_exact_at_10 == 1.0
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  order={order}: {status}  "
              f"recall_vs_exact@10={record.recall_vs_exact_at_10}  "
              f"cells_scanned_pct={record.cells_scanned_pct:.2f}%  "
              f"pruned_docs_pct={record.pruned_docs_pct:.2f}%  "
              f"tokens_pruned_pct={record.tokens_pruned_pct:.2f}%")

        out_path = RESULTS_JSON / f"stage2_testbed_exact_agreement_{DATASET}_{order}.json"
        record.to_json(out_path)
        print(f"    JSON: {out_path}")

    if not all_passed:
        print("FAIL: shrink=1 must reproduce the exact top-10 set for every order.")
        sys.exit(1)
    print("PASS (all orders)")


if __name__ == "__main__":
    main()
