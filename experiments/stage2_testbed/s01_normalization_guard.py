"""Stage 2 s01: Normalization guard across all exported datasets.

Single responsibility: for every dataset under data/embeddings/, load it via
bondmaxsim.data.loader.load_dataset (which itself runs a unit-norm sample
check) and additionally run bondmaxsim.oracle.normalization.check_unit_norm
over the full document-token and query-token arrays, printing pass/fail.

Blocking check (Stage 1 §8 item 1 / §4.1): if any token norm deviates from 1,
the shrink=1 kernel is not guaranteed exact-safe.

Usage:
    uv run python -m experiments.stage2_testbed.s01_normalization_guard
"""

from __future__ import annotations

import sys

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.oracle.normalization import check_unit_norm

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
DATA_DIR = REPO_ROOT / "data" / "embeddings"


def check_dataset(name: str) -> bool:
    npz_path = DATA_DIR / f"{name}.npz"
    if not npz_path.exists():
        print(f"  {name}: SKIP (missing {npz_path})")
        return True

    # load_dataset(verify_norm=True) already raises AssertionError on failure
    # for a sample of doc tokens + all query tokens; we additionally report
    # full-corpus stats here for visibility.
    flat_tokens, doc_starts, queries = load_dataset(name, verify_norm=True)

    doc_stats = check_unit_norm(flat_tokens)
    all_query_tokens = np.concatenate(queries, axis=0) if queries else flat_tokens[:0]
    query_stats = check_unit_norm(all_query_tokens)

    passed = doc_stats["n_violating"] == 0 and query_stats["n_violating"] == 0
    status = "PASS" if passed else "FAIL"
    print(
        f"  {name}: {status}  "
        f"docs[min={doc_stats['min_norm']:.6f}, max={doc_stats['max_norm']:.6f}, "
        f"n_violating={doc_stats['n_violating']}]  "
        f"queries[min={query_stats['min_norm']:.6f}, max={query_stats['max_norm']:.6f}, "
        f"n_violating={query_stats['n_violating']}]  "
        f"(docs={len(doc_starts)}, tokens={flat_tokens.shape[0]}, queries={len(queries)})"
    )
    return passed


def main() -> None:
    print("=== s01 normalization guard ===")
    all_passed = True
    for name in DATASETS:
        try:
            ok = check_dataset(name)
        except AssertionError as exc:
            print(f"  {name}: FAIL ({exc})")
            ok = False
        all_passed = all_passed and ok

    print("Result:", "ALL PASS" if all_passed else "FAILURES PRESENT")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
