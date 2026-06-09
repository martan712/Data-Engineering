"""Tiny PDX-BOND correctness smoke test.

This script is intended for Linux/WSL2 after the PDX Python extension has been
built. It compares PDX-BOND exact search against NumPy brute-force squared L2 on
a tiny random dataset. It is a correctness check, not a benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDX_PYTHON_PATH = PROJECT_ROOT / "external" / "PDX" / "python"


def import_pdx_bond():
    try:
        from pdxearch.index_factory import IndexPDXBONDFlat
        return IndexPDXBONDFlat
    except ModuleNotFoundError as first_exc:
        if PDX_PYTHON_PATH.exists():
            sys.path.insert(0, str(PDX_PYTHON_PATH))
            try:
                from pdxearch.index_factory import IndexPDXBONDFlat
                return IndexPDXBONDFlat
            except ModuleNotFoundError as second_exc:
                raise SystemExit(
                    "Could not import PDX-BOND Python API. Build/install PDX first in "
                    "Linux/WSL2 with `python -m pip install .` from `external/PDX`. "
                    f"Original error: {second_exc}"
                ) from second_exc

        raise SystemExit(
            "Could not import PDX-BOND Python API. Build/install PDX first in "
            "Linux/WSL2 with `python -m pip install .` from `external/PDX`. "
            f"Original error: {first_exc}"
        ) from first_exc


def brute_force_l2(data: np.ndarray, queries: np.ndarray, k: int) -> tuple[list[np.ndarray], float]:
    start = perf_counter()
    topk = []
    for query in queries:
        distances = np.sum((data - query) ** 2, axis=1)
        topk.append(np.argsort(distances, kind="stable")[:k])
    return topk, perf_counter() - start


def pdx_bond_l2(data: np.ndarray, queries: np.ndarray, k: int) -> tuple[list[np.ndarray], float, float]:
    IndexPDXBONDFlat = import_pdx_bond()

    index = IndexPDXBONDFlat(ndim=data.shape[1])

    start = perf_counter()
    index.add_load(np.ascontiguousarray(data))
    index_time = perf_counter() - start

    start = perf_counter()
    topk = []
    for query in queries:
        hits = index.search(np.ascontiguousarray(query), k)
        topk.append(np.array([hit.index for hit in hits], dtype=np.int64))
    query_time = perf_counter() - start

    return topk, index_time, query_time


def recall_at_k(expected: list[np.ndarray], observed: list[np.ndarray], k: int) -> float:
    recalls = []
    for exact, candidate in zip(expected, observed):
        recalls.append(len(set(exact[:k]).intersection(candidate[:k])) / k)
    return float(np.mean(recalls))


def exact_match_rate(expected: list[np.ndarray], observed: list[np.ndarray]) -> float:
    matches = [
        np.array_equal(exact, candidate)
        for exact, candidate in zip(expected, observed)
    ]
    return float(np.mean(matches))


def main() -> None:
    rng = np.random.default_rng(seed=42)
    num_vectors = 1_000
    num_dimensions = 64
    num_queries = 5
    k = 10

    data = rng.normal(size=(num_vectors, num_dimensions)).astype(np.float32)
    queries = rng.normal(size=(num_queries, num_dimensions)).astype(np.float32)

    exact_topk, brute_time = brute_force_l2(data, queries, k)
    pdx_topk, index_time, pdx_query_time = pdx_bond_l2(data, queries, k)

    recall = recall_at_k(exact_topk, pdx_topk, k)
    match_rate = exact_match_rate(exact_topk, pdx_topk)

    print("PDX-BOND smoke test")
    print(f"vectors={num_vectors}, dimensions={num_dimensions}, queries={num_queries}, k={k}")
    print(f"NumPy brute-force query time: {brute_time:.6f}s")
    print(f"PDX-BOND index/load time: {index_time:.6f}s")
    print(f"PDX-BOND query time: {pdx_query_time:.6f}s")
    print(f"recall@{k}: {recall:.4f}")
    print(f"exact top-{k} match rate: {match_rate:.4f}")
    print(f"first query NumPy top-{k}: {exact_topk[0].tolist()}")
    print(f"first query PDX top-{k}:   {pdx_topk[0].tolist()}")


if __name__ == "__main__":
    main()
