"""Stage 2 s02: shrink=1 exact-agreement gate, oracle + wide-block kernels.

Single responsibility: run the per-document-oracle kernel (unchanged) and the
wide-block kernel (self_bound threshold policy) in accounting mode via
bondmaxsim.testbed.runner.Runner with shrink=1.0, for every dimension order
(natural, bond, pca), assert recall_vs_exact@10 == 1.0 for each (Stage 1 §2.5
/ §8 items 2 and 5: order affects efficiency, never correctness), and write
one ResultRecord per order/kernel to results/json/.

Blocking check: if this fails for any order/kernel, no downstream mechanism or
integration result is trustworthy (see tests/test_runner_gate.py and
tests/test_wide_block_gate.py for the always-run pytest versions of this same
gate on synthetic data; this driver exercises real BEIR data).

Output naming:
  results/json/stage2_testbed_exact_agreement_<dataset>_<order>.json          (oracle kernel)
  results/json/stage2_testbed_exact_agreement_<dataset>_wide_<order>.json     (wide kernel)

The oracle-kernel runs are only meaningful/needed on scifact (already
committed there in Stage 2); for other datasets this driver runs the
wide-kernel gate only, to avoid redundant oracle-kernel wall time.

Usage:
    uv run python -m experiments.stage2_testbed.s02_exact_agreement [dataset ...]
"""

from __future__ import annotations

import sys

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.testbed.runner import Runner, RunConfig

DATASETS = ["scifact"]
ORDERS = ["natural", "bond", "pca"]
RESULTS_JSON = REPO_ROOT / "results" / "json"


def run_oracle(runner: Runner, dataset: str) -> bool:
    """Per-document-oracle kernel gate (unchanged file naming)."""
    all_passed = True
    for order in ORDERS:
        cfg = RunConfig(
            dataset=dataset,
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
        print(f"  [oracle] order={order}: {status}  "
              f"recall_vs_exact@10={record.recall_vs_exact_at_10}  "
              f"cells_scanned_pct={record.cells_scanned_pct:.2f}%  "
              f"pruned_docs_pct={record.pruned_docs_pct:.2f}%  "
              f"tokens_pruned_pct={record.tokens_pruned_pct:.2f}%")

        out_path = RESULTS_JSON / f"stage2_testbed_exact_agreement_{dataset}_{order}.json"
        record.to_json(out_path)
        print(f"    JSON: {out_path}")
    return all_passed


def run_wide(runner: Runner, dataset: str) -> bool:
    """Wide-block kernel gate (self_bound threshold policy)."""
    all_passed = True
    for order in ORDERS:
        cfg = RunConfig(
            dataset=dataset,
            method="wide_block_maxsim_bond",
            dimension_order=order,
            threshold_policy="self_bound",
            k=10,
            shrink=1.0,
        )
        record = runner.accounting_mode(cfg)

        passed = record.recall_vs_exact_at_10 == 1.0
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [wide]   order={order}: {status}  "
              f"recall_vs_exact@10={record.recall_vs_exact_at_10}  "
              f"cells_scanned_pct={record.cells_scanned_pct:.2f}%  "
              f"pruned_docs_pct={record.pruned_docs_pct:.2f}%  "
              f"tokens_pruned_pct={record.tokens_pruned_pct:.2f}%")

        out_path = RESULTS_JSON / f"stage2_testbed_exact_agreement_{dataset}_wide_{order}.json"
        record.to_json(out_path)
        print(f"    JSON: {out_path}")
    return all_passed


def run_dataset(dataset: str) -> bool:
    print(f"=== s02 exact agreement: {dataset} ===")
    flat_tokens, doc_starts, queries = load_dataset(dataset)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    runner = Runner(flat_tokens, doc_starts, queries)
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)

    all_passed = True
    if dataset == "scifact":
        all_passed = run_oracle(runner, dataset) and all_passed
    all_passed = run_wide(runner, dataset) and all_passed
    return all_passed


def main() -> None:
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS

    all_passed = True
    for dataset in datasets:
        all_passed = run_dataset(dataset) and all_passed

    if not all_passed:
        print("FAIL: shrink=1 must reproduce the exact top-10 set for every order/kernel.")
        sys.exit(1)
    print("PASS (all datasets/orders/kernels)")


if __name__ == "__main__":
    main()
