"""Scaling curve for SciFact exact MaxSim vs PDX candidate generation.

This script uses the already exported SciFact embeddings and evaluates several
document-prefix subsets. It is meant to answer a narrow question:

How do full exact MaxSim time and current PDX-token candidate generation time
change as the number of indexed document token vectors grows?

Run from the WSL PDX environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from utils_colbert import (
    compute_qrels_metrics,
    flatten_document_embeddings,
    l2_normalize,
    load_packed_embeddings,
    maxsim_score,
    maxsim_scores_for_candidate_docs,
    rank_documents,
    save_json,
    topk_overlap,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "scifact_benchmark"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "scifact_benchmark_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
OUTPUT_PATH = RESULTS_DIR / "scifact_scaling_curve.json"

DEFAULT_DOC_LIMITS = [250, 500, 1000]
DEFAULT_L_VALUES = [10, 20, 50, 100]
DEFAULT_C_VALUES = [20, 50, 100]
POLICIES = ["approx_score", "retrieved_token_count", "union_approx_and_count"]
REPORT_K_VALUES = [1, 3, 5, 10]


def import_pdx_api():
    """Import the PDX Python API from the installed package or local source."""
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
    """Load exported SciFact embeddings and qrels."""
    with (CORPUS_DIR / "qrels.json").open("r", encoding="utf-8") as file:
        qrels = json.load(file)
    document_pack = load_packed_embeddings(EMBEDDINGS_DIR / "documents_packed.npz")
    query_pack = load_packed_embeddings(EMBEDDINGS_DIR / "queries_packed.npz")
    return {
        "qrels": qrels,
        "document_pack": document_pack,
        "query_pack": query_pack,
        "documents": unpack_embeddings(document_pack["values"], document_pack["offsets"]),
        "queries": unpack_embeddings(query_pack["values"], query_pack["offsets"]),
    }


def filter_qrels_to_docs(qrels: dict[str, list[str]], doc_ids: list[str]) -> dict[str, list[str]]:
    """Keep only qrels whose relevant documents exist in the current subset."""
    doc_id_set = set(doc_ids)
    filtered = {}
    for query_id, relevant_ids in qrels.items():
        kept = [doc_id for doc_id in relevant_ids if doc_id in doc_id_set]
        if kept:
            filtered[query_id] = kept
    return filtered


def compute_exact_rankings(
    queries: list[np.ndarray],
    documents: list[np.ndarray],
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, list[str]],
) -> dict[str, Any]:
    """Compute full exact MaxSim rankings for one document subset."""
    start = perf_counter()
    rankings_by_query = {}
    for query_id, query_matrix in zip(query_ids, queries):
        scores = np.empty(len(documents), dtype=np.float32)
        for doc_index, document_matrix in enumerate(documents):
            scores[doc_index] = maxsim_score(query_matrix, document_matrix, normalize=False)
        ranking = rank_documents(scores)
        rankings_by_query[query_id] = [doc_ids[index] for index in ranking]
    seconds = perf_counter() - start
    return {
        "seconds": seconds,
        "comparisons": len(queries) * len(documents),
        "rankings_by_query": rankings_by_query,
        "qrels_metrics": compute_qrels_metrics(
            rankings_by_query,
            qrels,
            k_values=(1, 3, 5, 10),
            mrr_k=10,
        )["macro"],
    }


def retrieve_query_token_hits(index, query_matrix: np.ndarray, top_l: int) -> dict[str, Any]:
    """Retrieve PDX hits for every query token."""
    query_token_hits = []
    search_seconds = 0.0
    parse_seconds = 0.0
    for query_token in query_matrix:
        contiguous_query_token = np.ascontiguousarray(query_token, dtype=np.float32)
        start = perf_counter()
        hits = index.search(contiguous_query_token, top_l)
        search_seconds += perf_counter() - start

        start = perf_counter()
        query_token_hits.append(
            [
                (int(hit.index), 1.0 - (float(hit.distance) / 2.0))
                for hit in hits
            ]
        )
        parse_seconds += perf_counter() - start

    return {
        "query_token_hits": query_token_hits,
        "search_seconds": search_seconds,
        "parse_seconds": parse_seconds,
    }


def build_candidate_features(
    query_token_hits: list[list[tuple[int, float]]],
    token_to_doc_index: np.ndarray,
    num_documents: int,
) -> dict[str, Any]:
    """Aggregate token-level hits into document-level candidate features."""
    per_token_doc_best = np.full(
        (len(query_token_hits), num_documents),
        -np.inf,
        dtype=np.float32,
    )
    retrieved_token_count = np.zeros(num_documents, dtype=np.int32)
    candidate_doc_indices: set[int] = set()

    for query_token_index, hits in enumerate(query_token_hits):
        for token_index, similarity in hits:
            doc_index = int(token_to_doc_index[int(token_index)])
            candidate_doc_indices.add(doc_index)
            retrieved_token_count[doc_index] += 1
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
        "retrieved_token_vectors": sum(len(hits) for hits in query_token_hits),
    }


def sort_by_approx(features: dict[str, Any]) -> list[int]:
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (-float(features["approx_scores"][doc]), doc),
    )


def sort_by_count(features: dict[str, Any]) -> list[int]:
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (
            -int(features["retrieved_token_count"][doc]),
            -float(features["approx_scores"][doc]),
            doc,
        ),
    )


def select_candidates(features: dict[str, Any], top_c: int, policy: str) -> list[int]:
    """Select top-C candidate docs using existing deterministic policies."""
    approx_order = sort_by_approx(features)
    count_order = sort_by_count(features)
    if policy == "approx_score":
        return approx_order[:top_c]
    if policy == "retrieved_token_count":
        return count_order[:top_c]
    if policy == "union_approx_and_count":
        first_take = math.ceil(top_c / 2)
        approx_head = set(approx_order[:first_take])
        count_head = set(count_order[: top_c - first_take])
        selected = approx_head.union(count_head)
        for doc_index in approx_order:
            if len(selected) >= top_c:
                break
            selected.add(doc_index)
        return sorted(
            selected,
            key=lambda doc: (
                -(doc in approx_head and doc in count_head),
                -int(features["matched_query_token_count"][doc]),
                -float(features["approx_scores"][doc]),
                doc,
            ),
        )[:top_c]
    raise ValueError(f"Unknown policy: {policy}")


def compare_to_exact(reference: list[str], reranked: list[str], pool: set[str]) -> dict[str, Any]:
    """Compare a ranking or candidate pool against exact MaxSim."""
    exact_top5 = set(reference[:5])
    exact_top10 = set(reference[:10])
    return {
        "exact_recall_at_k": {
            str(k): topk_overlap(reference, reranked, k)["ratio"]
            for k in REPORT_K_VALUES
        },
        "exact_top1_match": reference[:1] == reranked[:1],
        "exact_top3_all_recovered": set(reference[:3]).issubset(set(reranked[:3])),
        "exact_top5_all_recovered": exact_top5.issubset(set(reranked[:5])),
        "exact_top10_all_recovered": exact_top10.issubset(set(reranked[:10])),
        "pool_contains_top5": exact_top5.issubset(pool),
        "pool_contains_top10": exact_top10.issubset(pool),
    }


def summarize_pool(
    generated: list[dict[str, Any]],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
    full_exact_comparisons: int,
) -> dict[str, Any]:
    """Summarize full unique candidate pool coverage."""
    comparisons = []
    pool_sizes = []
    for item in generated:
        query_id = item["query_id"]
        pool_doc_ids = {doc_ids[index] for index in item["features"]["candidate_doc_indices"]}
        oracle_pool_ranking = [
            doc_id for doc_id in exact_rankings[query_id] if doc_id in pool_doc_ids
        ]
        comparisons.append(compare_to_exact(exact_rankings[query_id], oracle_pool_ranking, pool_doc_ids))
        pool_sizes.append(len(pool_doc_ids))

    pool_comparisons = int(sum(pool_sizes))
    return {
        "mean_candidate_pool_size": float(np.mean(pool_sizes)),
        "pool_comparisons_if_reranked": pool_comparisons,
        "pool_comparison_ratio_if_reranked": pool_comparisons / full_exact_comparisons,
        "mean_exact_recall_at_k": {
            str(k): float(np.mean([item["exact_recall_at_k"][str(k)] for item in comparisons]))
            for k in REPORT_K_VALUES
        },
        "top5_all_recovered_count": int(sum(item["pool_contains_top5"] for item in comparisons)),
        "top10_all_recovered_count": int(sum(item["pool_contains_top10"] for item in comparisons)),
    }


def summarize_policy(
    *,
    policy: str,
    top_c: int,
    generated: list[dict[str, Any]],
    queries: list[np.ndarray],
    documents: list[np.ndarray],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
    qrels: dict[str, list[str]],
    full_exact_comparisons: int,
) -> dict[str, Any]:
    """Select, rerank, and summarize one top-C policy."""
    selection_seconds = 0.0
    rerank_seconds = 0.0
    comparisons = []
    rankings_by_query = {}
    selected_counts = []

    for item in generated:
        query_id = item["query_id"]
        query_index = item["query_index"]
        features = item["features"]
        start = perf_counter()
        selected = select_candidates(features, top_c, policy)
        selection_seconds += perf_counter() - start

        start = perf_counter()
        scores = maxsim_scores_for_candidate_docs(
            queries[query_index],
            documents,
            selected,
            normalize=False,
        )
        rerank_seconds += perf_counter() - start

        ordered_candidates = list(scores)
        candidate_scores = np.array([scores[index] for index in ordered_candidates], dtype=np.float32)
        reranked_indices = [ordered_candidates[index] for index in rank_documents(candidate_scores)]
        reranked_doc_ids = [doc_ids[index] for index in reranked_indices]
        rankings_by_query[query_id] = reranked_doc_ids

        pool_doc_ids = {doc_ids[index] for index in features["candidate_doc_indices"]}
        comparisons.append(compare_to_exact(exact_rankings[query_id], reranked_doc_ids, pool_doc_ids))
        selected_counts.append(len(selected))

    reranked_comparisons = int(sum(selected_counts))
    return {
        "policy": policy,
        "top_c": top_c,
        "selection_seconds": selection_seconds,
        "rerank_seconds": rerank_seconds,
        "selection_plus_rerank_seconds": selection_seconds + rerank_seconds,
        "reranked_comparisons": reranked_comparisons,
        "reranked_comparison_ratio": reranked_comparisons / full_exact_comparisons,
        "mean_selected_candidate_count": float(np.mean(selected_counts)),
        "mean_exact_recall_at_k": {
            str(k): float(np.mean([item["exact_recall_at_k"][str(k)] for item in comparisons]))
            for k in REPORT_K_VALUES
        },
        "top1_match_count": int(sum(item["exact_top1_match"] for item in comparisons)),
        "top3_all_recovered_count": int(sum(item["exact_top3_all_recovered"] for item in comparisons)),
        "top5_all_recovered_count": int(sum(item["exact_top5_all_recovered"] for item in comparisons)),
        "top10_all_recovered_count": int(sum(item["exact_top10_all_recovered"] for item in comparisons)),
        "qrels_metrics": compute_qrels_metrics(
            rankings_by_query,
            qrels,
            k_values=(1, 3, 5, 10),
            mrr_k=10,
        )["macro"],
    }


def run_for_doc_limit(
    *,
    IndexPDXBONDFlat,
    doc_limit: int,
    normalized_documents_all: list[np.ndarray],
    normalized_queries: list[np.ndarray],
    doc_ids_all: list[str],
    query_ids: list[str],
    qrels_all: dict[str, list[str]],
    l_values: list[int],
    c_values: list[int],
) -> dict[str, Any]:
    """Run exact and PDX timing for one document limit."""
    documents = normalized_documents_all[:doc_limit]
    doc_ids = doc_ids_all[:doc_limit]
    qrels = filter_qrels_to_docs(qrels_all, doc_ids)
    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(documents)

    exact = compute_exact_rankings(
        normalized_queries,
        documents,
        query_ids,
        doc_ids,
        qrels,
    )

    np.random.seed(0)
    index = IndexPDXBONDFlat(ndim=flat_tokens.shape[1])
    start = perf_counter()
    index.add_load(flat_tokens)
    index_build_seconds = perf_counter() - start

    l_runs = []
    for top_l in l_values:
        generated = []
        search_seconds = 0.0
        parse_seconds = 0.0
        aggregation_seconds = 0.0
        retrieved_token_vectors = 0
        wall_start = perf_counter()

        for query_index, query_matrix in enumerate(normalized_queries):
            retrieved = retrieve_query_token_hits(index, query_matrix, top_l)
            search_seconds += retrieved["search_seconds"]
            parse_seconds += retrieved["parse_seconds"]

            start = perf_counter()
            features = build_candidate_features(
                retrieved["query_token_hits"],
                token_to_doc_index,
                len(doc_ids),
            )
            aggregation_seconds += perf_counter() - start
            retrieved_token_vectors += features["retrieved_token_vectors"]
            generated.append(
                {
                    "query_id": query_ids[query_index],
                    "query_index": query_index,
                    "features": features,
                }
            )

        wall_seconds = perf_counter() - wall_start
        pool = summarize_pool(
            generated,
            exact["rankings_by_query"],
            doc_ids,
            exact["comparisons"],
        )
        policy_results = []
        for policy in POLICIES:
            for top_c in c_values:
                policy_results.append(
                    summarize_policy(
                        policy=policy,
                        top_c=top_c,
                        generated=generated,
                        queries=normalized_queries,
                        documents=documents,
                        exact_rankings=exact["rankings_by_query"],
                        doc_ids=doc_ids,
                        qrels=qrels,
                        full_exact_comparisons=exact["comparisons"],
                    )
                )

        l_runs.append(
            {
                "top_l": top_l,
                "candidate_generation": {
                    "wall_seconds": wall_seconds,
                    "pdx_search_seconds": search_seconds,
                    "hit_parse_seconds": parse_seconds,
                    "aggregation_seconds": aggregation_seconds,
                    "search_fraction_of_wall": search_seconds / wall_seconds if wall_seconds else 0.0,
                    "total_retrieved_token_vectors": int(retrieved_token_vectors),
                    "search_seconds_per_query_token": search_seconds / sum(len(q) for q in normalized_queries),
                    "mean_retrieved_token_fraction_per_query": (
                        retrieved_token_vectors / len(normalized_queries) / flat_tokens.shape[0]
                    ),
                },
                "pool": pool,
                "policy_results": policy_results,
            }
        )

    all_policy_results = [result for run in l_runs for result in run["policy_results"]]
    best_c20_recall5 = max(
        (result | {"top_l": run["top_l"]} for run in l_runs for result in run["policy_results"] if result["top_c"] == 20),
        key=lambda result: (
            result["mean_exact_recall_at_k"]["5"],
            result["top5_all_recovered_count"],
            result["mean_exact_recall_at_k"]["10"],
        ),
    )
    best_c50_recall10 = max(
        (result | {"top_l": run["top_l"]} for run in l_runs for result in run["policy_results"] if result["top_c"] == 50),
        key=lambda result: (
            result["mean_exact_recall_at_k"]["10"],
            result["top10_all_recovered_count"],
            result["mean_exact_recall_at_k"]["5"],
        ),
    )

    return {
        "doc_limit": doc_limit,
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "qrels_queries_with_relevant_docs": len(qrels),
        "qrels_labels_in_subset": int(sum(len(ids) for ids in qrels.values())),
        "document_token_vectors": int(flat_tokens.shape[0]),
        "query_token_vectors": int(sum(len(q) for q in normalized_queries)),
        "exact": {
            "seconds": exact["seconds"],
            "comparisons": exact["comparisons"],
            "seconds_per_comparison": exact["seconds"] / exact["comparisons"],
            "qrels_metrics_filtered_to_subset": exact["qrels_metrics"],
        },
        "pdx_index": {
            "build_seconds": index_build_seconds,
            "flat_values_shape": list(flat_tokens.shape),
        },
        "best_results": {
            "c20_recall5": best_c20_recall5,
            "c50_recall10": best_c50_recall10,
        },
        "runs": l_runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-limits", nargs="+", type=int, default=DEFAULT_DOC_LIMITS)
    parser.add_argument("--l-values", nargs="+", type=int, default=DEFAULT_L_VALUES)
    parser.add_argument("--c-values", nargs="+", type=int, default=DEFAULT_C_VALUES)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    IndexPDXBONDFlat, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")

    loaded = load_inputs()
    doc_ids_all = loaded["document_pack"]["ids"]
    query_ids = loaded["query_pack"]["ids"]
    max_docs = len(doc_ids_all)
    doc_limits = [limit for limit in args.doc_limits if 0 < limit <= max_docs]
    if not doc_limits:
        raise SystemExit(f"No valid doc limits. Available document count: {max_docs}")

    start = perf_counter()
    normalized_documents_all = [l2_normalize(document) for document in loaded["documents"]]
    normalized_queries = [l2_normalize(query) for query in loaded["queries"]]
    normalization_seconds = perf_counter() - start

    subset_results = []
    for doc_limit in doc_limits:
        print(f"Running doc_limit={doc_limit}")
        subset_results.append(
            run_for_doc_limit(
                IndexPDXBONDFlat=IndexPDXBONDFlat,
                doc_limit=doc_limit,
                normalized_documents_all=normalized_documents_all,
                normalized_queries=normalized_queries,
                doc_ids_all=doc_ids_all,
                query_ids=query_ids,
                qrels_all=loaded["qrels"],
                l_values=args.l_values,
                c_values=args.c_values,
            )
        )

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "candidate_generation": "l2sq_on_l2_normalized_vectors",
            "rerank": "exact_normalized_inner_product_maxsim",
        },
        "api_observation": {
            "python_batch_search_available": False,
            "note": "Current Python bindings expose search(q, k) for one query vector at a time.",
        },
        "timing_note": (
            "This is a component scaling curve for the current Python/WSL implementation, "
            "not a controlled end-to-end speedup claim."
        ),
        "global_dataset": {
            "available_documents": max_docs,
            "queries": len(query_ids),
            "total_available_document_token_vectors": int(loaded["document_pack"]["values"].shape[0]),
            "total_query_token_vectors": int(loaded["query_pack"]["values"].shape[0]),
            "embedding_dimension": int(loaded["document_pack"]["values"].shape[1]),
        },
        "normalization_seconds": normalization_seconds,
        "doc_limits": doc_limits,
        "l_values": args.l_values,
        "c_values": args.c_values,
        "policies": POLICIES,
        "subsets": subset_results,
    }
    save_json(args.output, output)

    print()
    print("SciFact scaling curve")
    for subset in subset_results:
        print(
            f"docs={subset['documents']} tokens={subset['document_token_vectors']} "
            f"exact={subset['exact']['seconds']:.6f}s "
            f"index={subset['pdx_index']['build_seconds']:.6f}s"
        )
        for run in subset["runs"]:
            gen = run["candidate_generation"]
            pool = run["pool"]
            print(
                f"  L={run['top_l']}: gen={gen['wall_seconds']:.6f}s "
                f"search={gen['pdx_search_seconds']:.6f}s "
                f"pool={pool['mean_candidate_pool_size']:.2f} "
                f"pool_rec5={pool['mean_exact_recall_at_k']['5']:.4f} "
                f"pool_rec10={pool['mean_exact_recall_at_k']['10']:.4f}"
            )
        best20 = subset["best_results"]["c20_recall5"]
        best50 = subset["best_results"]["c50_recall10"]
        print(
            f"  best C=20: L={best20['top_l']} {best20['policy']} "
            f"rec5={best20['mean_exact_recall_at_k']['5']:.4f} "
            f"ratio={best20['reranked_comparison_ratio']:.4f}"
        )
        print(
            f"  best C=50: L={best50['top_l']} {best50['policy']} "
            f"rec10={best50['mean_exact_recall_at_k']['10']:.4f} "
            f"ratio={best50['reranked_comparison_ratio']:.4f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
