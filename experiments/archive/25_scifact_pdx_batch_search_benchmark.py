"""Benchmark the prototype PDX batch-search API on SciFact token vectors.

The current ColBERT/PDX pipeline calls `index.search(q_token, L)` once per
query token. This script compares that existing Python loop against the new
prototype `index.search_batch(query_tokens, L)` binding.

Important scope:
- `search_batch` still loops over the existing PDX `Search` implementation in
  C++; it is not a new multi-token PDX algorithm.
- The measurement is meant to isolate Python/PyBind call and hit-object parsing
  overhead. It is not an end-to-end speedup claim.
- Candidate generation is compared against the already saved exact SciFact
  MaxSim rankings.

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
OUTPUT_PATH = RESULTS_DIR / "scifact_pdx_batch_search_benchmark.json"

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
                    "Could not import PDX. Build/install PDX in WSL/Linux first. "
                    f"Original error: {second_exc}"
                ) from second_exc
        raise SystemExit(
            "Could not import PDX. Build/install PDX in WSL/Linux first. "
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
        "exact_results": exact_results,
        "exact_rankings": exact_rankings,
    }


def search_query_tokens_loop(index, query_matrix: np.ndarray, top_l: int) -> dict[str, Any]:
    """Search query tokens one at a time with the existing Python API."""
    query_matrix = np.ascontiguousarray(query_matrix, dtype=np.float32)
    indices = np.full((len(query_matrix), top_l), UINT32_SENTINEL, dtype=np.uint32)
    distances = np.full((len(query_matrix), top_l), np.inf, dtype=np.float32)
    search_seconds = 0.0
    parse_seconds = 0.0

    for query_token_index, query_token in enumerate(query_matrix):
        start = perf_counter()
        hits = index.search(query_token, top_l)
        search_seconds += perf_counter() - start

        start = perf_counter()
        for hit_index, hit in enumerate(hits[:top_l]):
            indices[query_token_index, hit_index] = int(hit.index)
            distances[query_token_index, hit_index] = float(hit.distance)
        parse_seconds += perf_counter() - start

    return {
        "indices": indices,
        "distances": distances,
        "search_seconds": search_seconds,
        "parse_seconds": parse_seconds,
    }


def search_query_tokens_batch(index, query_matrix: np.ndarray, top_l: int) -> dict[str, Any]:
    """Search all query tokens with the prototype C++ batch wrapper."""
    query_matrix = np.ascontiguousarray(query_matrix, dtype=np.float32)
    start = perf_counter()
    indices, distances = index.search_batch(query_matrix, top_l)
    search_seconds = perf_counter() - start

    start = perf_counter()
    indices = np.asarray(indices, dtype=np.uint32)
    distances = np.asarray(distances, dtype=np.float32)
    parse_seconds = perf_counter() - start

    return {
        "indices": indices,
        "distances": distances,
        "search_seconds": search_seconds,
        "parse_seconds": parse_seconds,
    }


def aggregate_arrays(
    indices: np.ndarray,
    distances: np.ndarray,
    token_to_doc_index: np.ndarray,
    num_documents: int,
) -> dict[str, Any]:
    """Aggregate token-level PDX hits into document-level candidate features."""
    per_token_doc_best = np.full(
        (indices.shape[0], num_documents),
        -np.inf,
        dtype=np.float32,
    )
    retrieved_token_count = np.zeros(num_documents, dtype=np.int32)
    candidate_doc_indices: set[int] = set()
    retrieved_token_vectors = 0

    for query_token_index in range(indices.shape[0]):
        for hit_index in range(indices.shape[1]):
            token_index = int(indices[query_token_index, hit_index])
            if token_index == UINT32_SENTINEL:
                continue

            similarity = 1.0 - (float(distances[query_token_index, hit_index]) / 2.0)
            doc_index = int(token_to_doc_index[token_index])
            candidate_doc_indices.add(doc_index)
            retrieved_token_count[doc_index] += 1
            retrieved_token_vectors += 1
            if similarity > per_token_doc_best[query_token_index, doc_index]:
                per_token_doc_best[query_token_index, doc_index] = similarity

    matched_mask = np.isfinite(per_token_doc_best)
    approx_scores = np.where(matched_mask, per_token_doc_best, 0.0).sum(axis=0)
    matched_query_token_count = matched_mask.sum(axis=0).astype(np.int32)

    return {
        "candidate_doc_indices": candidate_doc_indices,
        "approx_scores": approx_scores.astype(np.float32),
        "retrieved_token_count": retrieved_token_count,
        "matched_query_token_count": matched_query_token_count,
        "retrieved_token_vectors": retrieved_token_vectors,
    }


def summarize_pool(
    generated: list[dict[str, Any]],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
) -> dict[str, Any]:
    """Summarize full candidate-pool coverage against exact MaxSim rankings."""
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
        "min_candidate_pool_size": int(np.min(pool_sizes)),
        "max_candidate_pool_size": int(np.max(pool_sizes)),
        "mean_exact_recall_at_k": {
            str(k): float(np.mean(coverage[k]))
            for k in REPORT_K_VALUES
        },
        "exact_topk_all_recovered_count": {
            str(k): int(all_recovered[k])
            for k in REPORT_K_VALUES
        },
    }


def validate_loop_vs_batch(loop_result: dict[str, Any], batch_result: dict[str, Any]) -> dict[str, Any]:
    """Check that batch search returns the same token hits as the loop API."""
    loop_indices = loop_result["indices"]
    batch_indices = batch_result["indices"]
    loop_distances = loop_result["distances"]
    batch_distances = batch_result["distances"]

    index_equal = bool(np.array_equal(loop_indices, batch_indices))
    distance_equal = bool(np.allclose(loop_distances, batch_distances, rtol=1e-6, atol=1e-6))
    mismatch_rows = 0
    if not index_equal or not distance_equal:
        row_equal = (
            np.all(loop_indices == batch_indices, axis=1)
            & np.all(np.isclose(loop_distances, batch_distances, rtol=1e-6, atol=1e-6), axis=1)
        )
        mismatch_rows = int(np.size(row_equal) - np.count_nonzero(row_equal))

    return {
        "indices_equal": index_equal,
        "distances_allclose": distance_equal,
        "mismatch_query_token_rows": mismatch_rows,
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
    """Benchmark loop and batch search for one L value."""
    loop_generated = []
    batch_generated = []
    loop_search_seconds = 0.0
    loop_parse_seconds = 0.0
    loop_aggregation_seconds = 0.0
    batch_search_seconds = 0.0
    batch_parse_seconds = 0.0
    batch_aggregation_seconds = 0.0
    validation_failures = []
    loop_wall_start = perf_counter()

    loop_results = []
    for query_matrix in normalized_queries:
        retrieved = search_query_tokens_loop(index, query_matrix, top_l)
        loop_results.append(retrieved)
        loop_search_seconds += retrieved["search_seconds"]
        loop_parse_seconds += retrieved["parse_seconds"]

        start = perf_counter()
        features = aggregate_arrays(
            retrieved["indices"],
            retrieved["distances"],
            token_to_doc_index,
            len(doc_ids),
        )
        loop_aggregation_seconds += perf_counter() - start
        loop_generated.append(features)

    loop_wall_seconds = perf_counter() - loop_wall_start

    batch_wall_start = perf_counter()
    batch_results = []
    for query_matrix in normalized_queries:
        retrieved = search_query_tokens_batch(index, query_matrix, top_l)
        batch_results.append(retrieved)
        batch_search_seconds += retrieved["search_seconds"]
        batch_parse_seconds += retrieved["parse_seconds"]

        start = perf_counter()
        features = aggregate_arrays(
            retrieved["indices"],
            retrieved["distances"],
            token_to_doc_index,
            len(doc_ids),
        )
        batch_aggregation_seconds += perf_counter() - start
        batch_generated.append(features)

    batch_wall_seconds = perf_counter() - batch_wall_start

    pool_sets_equal = True
    for query_index, (loop_result, batch_result) in enumerate(zip(loop_results, batch_results)):
        validation = validate_loop_vs_batch(loop_result, batch_result)
        if not validation["indices_equal"] or not validation["distances_allclose"]:
            validation_failures.append(
                {
                    "query_id": query_ids[query_index],
                    **validation,
                }
            )
        if loop_generated[query_index]["candidate_doc_indices"] != batch_generated[query_index]["candidate_doc_indices"]:
            pool_sets_equal = False

    loop_summary_inputs = [
        {
            "query_id": query_id,
            "features": features,
        }
        for query_id, features in zip(query_ids, loop_generated)
    ]
    batch_summary_inputs = [
        {
            "query_id": query_id,
            "features": features,
        }
        for query_id, features in zip(query_ids, batch_generated)
    ]
    loop_pool_summary = summarize_pool(loop_summary_inputs, exact_rankings, doc_ids)
    batch_pool_summary = summarize_pool(batch_summary_inputs, exact_rankings, doc_ids)

    loop_total = loop_search_seconds + loop_parse_seconds + loop_aggregation_seconds
    batch_total = batch_search_seconds + batch_parse_seconds + batch_aggregation_seconds

    return {
        "top_l": top_l,
        "loop": {
            "wall_seconds": loop_wall_seconds,
            "search_seconds": loop_search_seconds,
            "parse_seconds": loop_parse_seconds,
            "aggregation_seconds": loop_aggregation_seconds,
            "measured_component_seconds": loop_total,
            "pool_summary": loop_pool_summary,
        },
        "batch": {
            "wall_seconds": batch_wall_seconds,
            "search_seconds": batch_search_seconds,
            "parse_seconds": batch_parse_seconds,
            "aggregation_seconds": batch_aggregation_seconds,
            "measured_component_seconds": batch_total,
            "pool_summary": batch_pool_summary,
        },
        "speedups_loop_over_batch": {
            "search_seconds": loop_search_seconds / batch_search_seconds if batch_search_seconds else None,
            "wall_seconds": loop_wall_seconds / batch_wall_seconds if batch_wall_seconds else None,
            "measured_component_seconds": loop_total / batch_total if batch_total else None,
        },
        "validation": {
            "query_count": len(query_ids),
            "query_token_count": int(sum(len(query) for query in normalized_queries)),
            "all_indices_equal": not validation_failures,
            "all_candidate_pools_equal": pool_sets_equal,
            "failure_count": len(validation_failures),
            "failures": validation_failures[:5],
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
    document_pack = loaded["document_pack"]
    query_pack = loaded["query_pack"]
    documents = loaded["documents"]
    queries = loaded["queries"]
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
    if not hasattr(index, "search_batch"):
        raise SystemExit(
            "This PDX build does not expose IndexPDXBONDFlat.search_batch. "
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

        speedup = run["speedups_loop_over_batch"]["search_seconds"]
        print(
            f"L={top_l:<3} "
            f"loop_search={run['loop']['search_seconds']:.6f}s "
            f"batch_search={run['batch']['search_seconds']:.6f}s "
            f"search_speedup={speedup:.2f}x "
            f"loop_wall={run['loop']['wall_seconds']:.6f}s "
            f"batch_wall={run['batch']['wall_seconds']:.6f}s "
            f"pools_equal={run['validation']['all_candidate_pools_equal']}"
        )

    output = {
        "experiment": "scifact_pdx_batch_search_benchmark",
        "note": (
            "The prototype search_batch binding loops over the existing PDX Search "
            "implementation in C++. It reduces Python/PyBind call and object parsing "
            "overhead, but it is not a synchronized multi-token search algorithm."
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

    best_search = max(
        runs,
        key=lambda item: item["speedups_loop_over_batch"]["search_seconds"] or 0.0,
    )
    best_wall = max(
        runs,
        key=lambda item: item["speedups_loop_over_batch"]["wall_seconds"] or 0.0,
    )
    print("\nSummary")
    print(f"Index build: {index_build_seconds:.6f}s")
    print(
        "Best search-component speedup: "
        f"L={best_search['top_l']} "
        f"{best_search['speedups_loop_over_batch']['search_seconds']:.2f}x"
    )
    print(
        "Best wall-clock candidate-generation speedup: "
        f"L={best_wall['top_l']} "
        f"{best_wall['speedups_loop_over_batch']['wall_seconds']:.2f}x"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
