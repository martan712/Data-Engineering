"""Benchmark a shared-scan PDX batch prototype on SciFact token vectors.

Experiment 25 showed that moving the per-token BOND loop from Python into C++
does not materially improve runtime. This experiment tests a different idea:
scan each PDX vectorgroup once for all query tokens and keep one heap per query
token.

The prototype method is `IndexPDXBONDFlat.search_batch_shared_scan`.

Important scope:
- This is a full exact scan over the PDX layout, not BOND pruning.
- It is intended as a kernel-direction test: can multi-token search benefit
  from sharing document-token memory access?
- It should return the same top-L token vectors as exact BOND on the flat index.
- No end-to-end speedup is claimed.

Run from the WSL PDX environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from utils_colbert import (
    flatten_document_embeddings,
    l2_normalize,
    load_packed_embeddings,
    save_json,
    topk_overlap,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "scifact_benchmark_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
EXACT_RESULTS_PATH = RESULTS_DIR / "scifact_colbert_maxsim_numpy.json"
OUTPUT_PATH = RESULTS_DIR / "scifact_pdx_shared_scan_benchmark.json"

DEFAULT_L_VALUES = [10, 20, 50, 100]
REPORT_K_VALUES = [1, 3, 5, 10]
UINT32_SENTINEL = np.iinfo(np.uint32).max


def import_pdx_api():
    """Import PDX from the installed package, or from the local source tree."""
    try:
        from pdxearch.constants import PDXConstants
        from pdxearch.index_factory import IndexPDXBONDFlat

        return IndexPDXBONDFlat, PDXConstants.SUPPORTED_METRICS
    except ModuleNotFoundError as first_exc:
        source_path = PROJECT_ROOT / "external" / "PDX" / "python"
        if source_path.exists():
            sys.path.insert(0, str(source_path))
            try:
                from pdxearch.constants import PDXConstants
                from pdxearch.index_factory import IndexPDXBONDFlat

                return IndexPDXBONDFlat, PDXConstants.SUPPORTED_METRICS
            except ModuleNotFoundError as second_exc:
                raise SystemExit(
                    "Could not import PDX. Build/install patched PDX in WSL/Linux first. "
                    f"Original error: {second_exc}"
                ) from second_exc
        raise SystemExit(
            "Could not import PDX. Build/install patched PDX in WSL/Linux first. "
            f"Original error: {first_exc}"
        ) from first_exc


def load_inputs() -> dict[str, Any]:
    """Load packed SciFact embeddings and exact MaxSim rankings."""
    document_path = EMBEDDINGS_DIR / "documents_packed.npz"
    query_path = EMBEDDINGS_DIR / "queries_packed.npz"
    if not document_path.exists() or not query_path.exists():
        raise SystemExit("Missing SciFact embeddings. Run experiment 20 first.")
    if not EXACT_RESULTS_PATH.exists():
        raise SystemExit("Missing exact SciFact results. Run experiment 21 first.")

    document_pack = load_packed_embeddings(document_path)
    query_pack = load_packed_embeddings(query_path)
    with EXACT_RESULTS_PATH.open("r", encoding="utf-8") as file:
        exact_results = json.load(file)
    exact_rankings = {
        item["query_id"]: item["ranking_doc_ids"]
        for item in exact_results["query_results"]
    }

    return {
        "document_pack": document_pack,
        "query_pack": query_pack,
        "documents": unpack_embeddings(document_pack["values"], document_pack["offsets"]),
        "queries": unpack_embeddings(query_pack["values"], query_pack["offsets"]),
        "exact_rankings": exact_rankings,
    }


def aggregate_arrays(
    indices: np.ndarray,
    distances: np.ndarray,
    token_to_doc_index: np.ndarray,
    num_documents: int,
) -> dict[str, Any]:
    """Aggregate token-level hits into a document candidate pool."""
    candidate_doc_indices: set[int] = set()
    retrieved_token_vectors = 0
    for query_token_index in range(indices.shape[0]):
        for hit_index in range(indices.shape[1]):
            token_index = int(indices[query_token_index, hit_index])
            if token_index == UINT32_SENTINEL:
                continue
            candidate_doc_indices.add(int(token_to_doc_index[token_index]))
            retrieved_token_vectors += 1

    return {
        "candidate_doc_indices": candidate_doc_indices,
        "retrieved_token_vectors": retrieved_token_vectors,
    }


def summarize_pool(
    generated: list[dict[str, Any]],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
) -> dict[str, Any]:
    """Summarize candidate-pool coverage against exact MaxSim rankings."""
    pool_sizes = []
    coverage = {k: [] for k in REPORT_K_VALUES}
    all_recovered = {k: 0 for k in REPORT_K_VALUES}

    for item in generated:
        query_id = item["query_id"]
        pool_doc_ids = {
            doc_ids[doc_index]
            for doc_index in item["features"]["candidate_doc_indices"]
        }
        reference = exact_rankings[query_id]
        oracle_ranking = [doc_id for doc_id in reference if doc_id in pool_doc_ids]
        pool_sizes.append(len(pool_doc_ids))

        for k in REPORT_K_VALUES:
            coverage[k].append(topk_overlap(reference, oracle_ranking, k)["ratio"])
            if set(reference[:k]).issubset(pool_doc_ids):
                all_recovered[k] += 1

    return {
        "mean_candidate_pool_size": float(np.mean(pool_sizes)),
        "mean_exact_recall_at_k": {
            str(k): float(np.mean(coverage[k]))
            for k in REPORT_K_VALUES
        },
        "exact_topk_all_recovered_count": {
            str(k): int(all_recovered[k])
            for k in REPORT_K_VALUES
        },
    }


def validate_results(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Validate shared-scan hits against the BOND batch reference."""
    reference_indices = reference["indices"]
    candidate_indices = candidate["indices"]
    reference_distances = reference["distances"]
    candidate_distances = candidate["distances"]

    indices_equal = bool(np.array_equal(reference_indices, candidate_indices))
    distances_allclose = bool(
        np.allclose(reference_distances, candidate_distances, rtol=1e-5, atol=1e-5)
    )
    mismatch_rows = 0
    if not indices_equal or not distances_allclose:
        row_equal = (
            np.all(reference_indices == candidate_indices, axis=1)
            & np.all(
                np.isclose(reference_distances, candidate_distances, rtol=1e-5, atol=1e-5),
                axis=1,
            )
        )
        mismatch_rows = int(np.size(row_equal) - np.count_nonzero(row_equal))

    row_sets_equal = [
        set(reference_indices[row].tolist()) == set(candidate_indices[row].tolist())
        for row in range(reference_indices.shape[0])
    ]
    set_mismatch_rows = int(len(row_sets_equal) - sum(row_sets_equal))

    return {
        "indices_equal": indices_equal,
        "distances_allclose": distances_allclose,
        "mismatch_query_token_rows": mismatch_rows,
        "top_l_sets_equal": set_mismatch_rows == 0,
        "top_l_set_mismatch_query_token_rows": set_mismatch_rows,
    }


def search_with_method(index, method_name: str, query_matrix: np.ndarray, top_l: int) -> dict[str, Any]:
    """Run one PDX batch method and return NumPy arrays plus elapsed time."""
    query_matrix = np.ascontiguousarray(query_matrix, dtype=np.float32)
    method = getattr(index, method_name)
    start = perf_counter()
    indices, distances = method(query_matrix, top_l)
    elapsed = perf_counter() - start
    return {
        "indices": np.asarray(indices, dtype=np.uint32),
        "distances": np.asarray(distances, dtype=np.float32),
        "seconds": elapsed,
    }


def benchmark_top_l(
    *,
    index,
    normalized_queries: list[np.ndarray],
    query_ids: list[str],
    doc_ids: list[str],
    token_to_doc_index: np.ndarray,
    exact_rankings: dict[str, list[str]],
    top_l: int,
) -> dict[str, Any]:
    """Compare BOND batch search against shared full scan for one L."""
    bond_generated = []
    shared_generated = []
    bond_search_seconds = 0.0
    shared_search_seconds = 0.0
    bond_aggregation_seconds = 0.0
    shared_aggregation_seconds = 0.0
    validation_failures = []
    set_validation_failures = []
    pools_equal = True

    wall_start = perf_counter()
    for query_index, query_matrix in enumerate(normalized_queries):
        bond = search_with_method(index, "search_batch", query_matrix, top_l)
        shared = search_with_method(index, "search_batch_shared_scan", query_matrix, top_l)
        bond_search_seconds += bond["seconds"]
        shared_search_seconds += shared["seconds"]

        validation = validate_results(bond, shared)
        if not validation["indices_equal"] or not validation["distances_allclose"]:
            validation_failures.append(
                {
                    "query_id": query_ids[query_index],
                    **validation,
                }
            )
        if not validation["top_l_sets_equal"]:
            set_validation_failures.append(
                {
                    "query_id": query_ids[query_index],
                    **validation,
                }
            )

        start = perf_counter()
        bond_features = aggregate_arrays(
            bond["indices"],
            bond["distances"],
            token_to_doc_index,
            len(doc_ids),
        )
        bond_aggregation_seconds += perf_counter() - start

        start = perf_counter()
        shared_features = aggregate_arrays(
            shared["indices"],
            shared["distances"],
            token_to_doc_index,
            len(doc_ids),
        )
        shared_aggregation_seconds += perf_counter() - start

        if bond_features["candidate_doc_indices"] != shared_features["candidate_doc_indices"]:
            pools_equal = False

        bond_generated.append(
            {
                "query_id": query_ids[query_index],
                "features": bond_features,
            }
        )
        shared_generated.append(
            {
                "query_id": query_ids[query_index],
                "features": shared_features,
            }
        )

    wall_seconds = perf_counter() - wall_start

    bond_total = bond_search_seconds + bond_aggregation_seconds
    shared_total = shared_search_seconds + shared_aggregation_seconds
    return {
        "top_l": top_l,
        "wall_seconds": wall_seconds,
        "bond_batch": {
            "search_seconds": bond_search_seconds,
            "aggregation_seconds": bond_aggregation_seconds,
            "measured_component_seconds": bond_total,
            "pool_summary": summarize_pool(bond_generated, exact_rankings, doc_ids),
        },
        "shared_scan": {
            "search_seconds": shared_search_seconds,
            "aggregation_seconds": shared_aggregation_seconds,
            "measured_component_seconds": shared_total,
            "pool_summary": summarize_pool(shared_generated, exact_rankings, doc_ids),
        },
        "speedups_bond_over_shared": {
            "search_seconds": bond_search_seconds / shared_search_seconds if shared_search_seconds else None,
            "measured_component_seconds": bond_total / shared_total if shared_total else None,
        },
        "validation": {
            "query_count": len(query_ids),
            "query_token_count": int(sum(len(query) for query in normalized_queries)),
            "all_indices_equal": not validation_failures,
            "all_top_l_sets_equal": not set_validation_failures,
            "all_candidate_pools_equal": pools_equal,
            "failure_count": len(validation_failures),
            "top_l_set_failure_count": len(set_validation_failures),
            "failures": validation_failures[:5],
            "top_l_set_failures": set_validation_failures[:5],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l-values", nargs="+", type=int, default=DEFAULT_L_VALUES)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    IndexPDXBONDFlat, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")

    loaded = load_inputs()
    documents = loaded["documents"]
    queries = loaded["queries"]
    document_pack = loaded["document_pack"]
    query_pack = loaded["query_pack"]
    doc_ids = document_pack["ids"]
    query_ids = query_pack["ids"]
    exact_rankings = loaded["exact_rankings"]

    start = perf_counter()
    normalized_documents = [l2_normalize(document) for document in documents]
    normalized_queries = [l2_normalize(query) for query in queries]
    normalization_seconds = perf_counter() - start

    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(normalized_documents)

    np.random.seed(0)
    index = IndexPDXBONDFlat(ndim=flat_tokens.shape[1])
    if not hasattr(index, "search_batch_shared_scan"):
        raise SystemExit(
            "This PDX build does not expose search_batch_shared_scan. "
            "Rebuild the patched WSL PDX source first."
        )

    start = perf_counter()
    index.add_load(flat_tokens)
    index_build_seconds = perf_counter() - start

    runs = []
    for top_l in args.l_values:
        if top_l > len(flat_tokens):
            continue
        run = benchmark_top_l(
            index=index,
            normalized_queries=normalized_queries,
            query_ids=query_ids,
            doc_ids=doc_ids,
            token_to_doc_index=token_to_doc_index,
            exact_rankings=exact_rankings,
            top_l=top_l,
        )
        runs.append(run)
        speedup = run["speedups_bond_over_shared"]["search_seconds"]
        print(
            f"L={top_l:<3} "
            f"bond_batch={run['bond_batch']['search_seconds']:.6f}s "
            f"shared_scan={run['shared_scan']['search_seconds']:.6f}s "
            f"bond_over_shared={speedup:.2f}x "
            f"ordered_hits_equal={run['validation']['all_indices_equal']} "
            f"hit_sets_equal={run['validation']['all_top_l_sets_equal']} "
            f"pools_equal={run['validation']['all_candidate_pools_equal']}"
        )

    output = {
        "experiment": "scifact_pdx_shared_scan_benchmark",
        "note": (
            "search_batch_shared_scan is an exact full scan over PDX vectorgroups. "
            "It shares the document-token block scan across query tokens but does not "
            "use BOND pruning."
        ),
        "metric": "l2sq_on_l2_normalized_colbert_token_vectors",
        "dataset": {
            "documents": len(documents),
            "queries": len(queries),
            "document_token_vectors": int(flat_tokens.shape[0]),
            "query_token_vectors": int(sum(len(query) for query in queries)),
            "embedding_dim": int(flat_tokens.shape[1]),
            "full_exact_document_comparisons": len(documents) * len(queries),
        },
        "timings": {
            "normalization_seconds": normalization_seconds,
            "pdx_index_build_seconds": index_build_seconds,
        },
        "l_values": [run["top_l"] for run in runs],
        "runs": runs,
    }
    save_json(args.output, output)

    best = max(
        runs,
        key=lambda item: item["speedups_bond_over_shared"]["search_seconds"] or 0.0,
    )
    print("\nSummary")
    print(f"Index build: {index_build_seconds:.6f}s")
    print(
        "Best BOND/shared search ratio: "
        f"L={best['top_l']} "
        f"{best['speedups_bond_over_shared']['search_seconds']:.2f}x"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
