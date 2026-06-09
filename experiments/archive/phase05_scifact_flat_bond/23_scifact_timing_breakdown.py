"""Timing breakdown for the SciFact PDX candidate-generation pipeline.

This script is intentionally measurement-oriented. It reuses the same flat
ColBERT-token representation as the previous SciFact experiment, but separates
the main runtime components:

- full exact NumPy MaxSim reference
- PDX index build
- PDX search calls
- Python hit parsing
- Python candidate aggregation
- top-C selection
- exact MaxSim reranking over selected candidates

Run it from the WSL PDX environment, because it imports the compiled PDX
extension.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "scifact_benchmark"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "scifact_benchmark_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
OUTPUT_PATH = RESULTS_DIR / "scifact_timing_breakdown.json"

DEFAULT_L_VALUES = [10, 20, 50, 100]
DEFAULT_C_VALUES = [20, 50, 100]
POLICIES = ["approx_score", "retrieved_token_count", "union_approx_and_count"]
REPORT_K_VALUES = [1, 3, 5, 10]


def import_pdx_api():
    """Import the PDX Python API, falling back to the local source tree."""
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
    """Load SciFact embeddings and metadata."""
    if not (EMBEDDINGS_DIR / "documents_packed.npz").exists():
        raise SystemExit("Missing SciFact embeddings. Run experiment 20 first.")
    if not (CORPUS_DIR / "qrels.json").exists():
        raise SystemExit("Missing SciFact qrels. Run experiment 19 first.")

    with (CORPUS_DIR / "qrels.json").open("r", encoding="utf-8") as file:
        qrels = json.load(file)

    document_pack = load_packed_embeddings(EMBEDDINGS_DIR / "documents_packed.npz")
    query_pack = load_packed_embeddings(EMBEDDINGS_DIR / "queries_packed.npz")
    documents = unpack_embeddings(document_pack["values"], document_pack["offsets"])
    queries = unpack_embeddings(query_pack["values"], query_pack["offsets"])

    return {
        "qrels": qrels,
        "document_pack": document_pack,
        "query_pack": query_pack,
        "documents": documents,
        "queries": queries,
    }


def compute_full_exact_reference(
    queries: list[np.ndarray],
    documents: list[np.ndarray],
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, list[str]],
) -> dict[str, Any]:
    """Compute the full exact normalized MaxSim reference in this process."""
    start = perf_counter()
    rankings_by_query: dict[str, list[str]] = {}
    for query_id, query_matrix in zip(query_ids, queries):
        scores = np.empty(len(documents), dtype=np.float32)
        for doc_index, document_matrix in enumerate(documents):
            scores[doc_index] = maxsim_score(query_matrix, document_matrix, normalize=False)
        ranking_indices = rank_documents(scores)
        rankings_by_query[query_id] = [doc_ids[index] for index in ranking_indices]

    elapsed = perf_counter() - start
    return {
        "seconds": elapsed,
        "document_comparisons": len(queries) * len(documents),
        "rankings_by_query": rankings_by_query,
        "qrels_metrics": compute_qrels_metrics(
            rankings_by_query,
            qrels,
            k_values=(1, 3, 5, 10),
            mrr_k=10,
        ),
    }


def retrieve_query_token_hits(index, query_matrix: np.ndarray, top_l: int) -> dict[str, Any]:
    """Retrieve PDX hits for every query token and time search vs hit parsing."""
    query_token_hits: list[list[tuple[int, float]]] = []
    search_seconds = 0.0
    parse_seconds = 0.0
    for query_token in query_matrix:
        contiguous_query_token = np.ascontiguousarray(query_token, dtype=np.float32)
        start = perf_counter()
        hits = index.search(contiguous_query_token, top_l)
        search_seconds += perf_counter() - start

        start = perf_counter()
        parsed_hits = [
            (int(hit.index), 1.0 - (float(hit.distance) / 2.0))
            for hit in hits
        ]
        parse_seconds += perf_counter() - start
        query_token_hits.append(parsed_hits)

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
    """Aggregate token hits into document-level selector features."""
    query_token_count = len(query_token_hits)
    per_token_doc_best = np.full(
        (query_token_count, num_documents),
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
    """Sort candidate docs by approximate aggregate score."""
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (-float(features["approx_scores"][doc]), doc),
    )


def sort_by_count(features: dict[str, Any]) -> list[int]:
    """Sort candidate docs by retrieved token count, then approximate score."""
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (
            -int(features["retrieved_token_count"][doc]),
            -float(features["approx_scores"][doc]),
            doc,
        ),
    )


def select_candidates(features: dict[str, Any], top_c: int | None, policy: str) -> list[int]:
    """Select candidate docs with the same deterministic policies as experiment 22."""
    if top_c is None:
        return sort_by_approx(features)

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


def rerank_candidates(
    query_matrix: np.ndarray,
    documents: list[np.ndarray],
    selected_doc_indices: list[int],
) -> tuple[list[int], float]:
    """Exact MaxSim rerank for a selected candidate set."""
    start = perf_counter()
    candidate_scores = maxsim_scores_for_candidate_docs(
        query_matrix,
        documents,
        selected_doc_indices,
        normalize=False,
    )
    elapsed = perf_counter() - start
    if not candidate_scores:
        return [], elapsed

    ordered_candidates = list(candidate_scores)
    scores = np.array([candidate_scores[index] for index in ordered_candidates], dtype=np.float32)
    ranking = rank_documents(scores)
    return [ordered_candidates[index] for index in ranking], elapsed


def compare_to_exact(
    reference: list[str],
    reranked: list[str],
    selected: set[str],
    pool: set[str],
) -> dict[str, Any]:
    """Compare a candidate/reranked result against full exact MaxSim."""
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
        "selector_loss_at_5": (
            "none"
            if exact_top5.issubset(selected)
            else "pool_coverage_failure"
            if not exact_top5.issubset(pool)
            else "selector_failure"
        ),
        "selector_loss_at_10": (
            "none"
            if exact_top10.issubset(selected)
            else "pool_coverage_failure"
            if not exact_top10.issubset(pool)
            else "selector_failure"
        ),
    }


def summarize_policy(
    *,
    policy: str,
    top_l: int,
    top_c: int,
    generated: list[dict[str, Any]],
    normalized_queries: list[np.ndarray],
    normalized_documents: list[np.ndarray],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
    qrels: dict[str, list[str]],
    full_exact_comparisons: int,
) -> dict[str, Any]:
    """Profile one top-C selection policy and exact reranking."""
    selection_seconds = 0.0
    rerank_seconds = 0.0
    selected_counts = []
    rankings_by_query = {}
    comparisons = []

    for item in generated:
        query_id = item["query_id"]
        query_index = item["query_index"]
        features = item["features"]

        start = perf_counter()
        selected = select_candidates(features, top_c, policy)
        selection_seconds += perf_counter() - start

        reranked, elapsed = rerank_candidates(
            normalized_queries[query_index],
            normalized_documents,
            selected,
        )
        rerank_seconds += elapsed

        selected_doc_ids = {doc_ids[index] for index in selected}
        pool_doc_ids = {doc_ids[index] for index in features["candidate_doc_indices"]}
        reranked_doc_ids = [doc_ids[index] for index in reranked]
        rankings_by_query[query_id] = reranked_doc_ids
        selected_counts.append(len(selected))
        comparisons.append(
            compare_to_exact(
                exact_rankings[query_id],
                reranked_doc_ids,
                selected_doc_ids,
                pool_doc_ids,
            )
        )

    reranked_comparisons = int(sum(selected_counts))
    return {
        "policy": policy,
        "top_l": top_l,
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
        "exact_top1_match_count": int(sum(item["exact_top1_match"] for item in comparisons)),
        "exact_top3_all_recovered_count": int(sum(item["exact_top3_all_recovered"] for item in comparisons)),
        "exact_top5_all_recovered_count": int(sum(item["exact_top5_all_recovered"] for item in comparisons)),
        "exact_top10_all_recovered_count": int(sum(item["exact_top10_all_recovered"] for item in comparisons)),
        "selector_loss": {
            "5": {
                "pool_coverage_failure": int(sum(item["selector_loss_at_5"] == "pool_coverage_failure" for item in comparisons)),
                "selector_failure": int(sum(item["selector_loss_at_5"] == "selector_failure" for item in comparisons)),
                "none": int(sum(item["selector_loss_at_5"] == "none" for item in comparisons)),
            },
            "10": {
                "pool_coverage_failure": int(sum(item["selector_loss_at_10"] == "pool_coverage_failure" for item in comparisons)),
                "selector_failure": int(sum(item["selector_loss_at_10"] == "selector_failure" for item in comparisons)),
                "none": int(sum(item["selector_loss_at_10"] == "none" for item in comparisons)),
            },
        },
        "qrels_metrics": compute_qrels_metrics(
            rankings_by_query,
            qrels,
            k_values=(1, 3, 5, 10),
            mrr_k=10,
        )["macro"],
    }


def summarize_pool(
    generated: list[dict[str, Any]],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
    full_exact_comparisons: int,
) -> dict[str, Any]:
    """Summarize full candidate-pool coverage without reranking scores."""
    comparisons = []
    pool_sizes = []
    for item in generated:
        query_id = item["query_id"]
        pool_doc_ids = {doc_ids[index] for index in item["features"]["candidate_doc_indices"]}
        oracle_ranking = [doc_id for doc_id in exact_rankings[query_id] if doc_id in pool_doc_ids]
        comparisons.append(
            compare_to_exact(
                exact_rankings[query_id],
                oracle_ranking,
                pool_doc_ids,
                pool_doc_ids,
            )
        )
        pool_sizes.append(len(pool_doc_ids))

    reranked_comparisons = int(sum(pool_sizes))
    return {
        "mean_candidate_pool_size": float(np.mean(pool_sizes)),
        "pool_comparisons_if_reranked": reranked_comparisons,
        "pool_comparison_ratio_if_reranked": reranked_comparisons / full_exact_comparisons,
        "mean_exact_recall_at_k": {
            str(k): float(np.mean([item["exact_recall_at_k"][str(k)] for item in comparisons]))
            for k in REPORT_K_VALUES
        },
        "exact_top1_match_count": int(sum(item["exact_top1_match"] for item in comparisons)),
        "exact_top3_all_recovered_count": int(sum(item["exact_top3_all_recovered"] for item in comparisons)),
        "exact_top5_all_recovered_count": int(sum(item["exact_top5_all_recovered"] for item in comparisons)),
        "exact_top10_all_recovered_count": int(sum(item["exact_top10_all_recovered"] for item in comparisons)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    qrels = loaded["qrels"]
    document_pack = loaded["document_pack"]
    query_pack = loaded["query_pack"]
    documents = loaded["documents"]
    queries = loaded["queries"]
    doc_ids = document_pack["ids"]
    query_ids = query_pack["ids"]

    start = perf_counter()
    normalized_documents = [l2_normalize(document) for document in documents]
    normalized_queries = [l2_normalize(query) for query in queries]
    normalization_seconds = perf_counter() - start

    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(normalized_documents)
    full_exact = compute_full_exact_reference(
        normalized_queries,
        normalized_documents,
        query_ids,
        doc_ids,
        qrels,
    )
    exact_rankings = full_exact["rankings_by_query"]

    np.random.seed(0)
    index = IndexPDXBONDFlat(ndim=flat_tokens.shape[1])
    start = perf_counter()
    index.add_load(flat_tokens)
    index_build_seconds = perf_counter() - start

    full_exact_comparisons = len(normalized_queries) * len(normalized_documents)
    total_query_token_vectors = int(sum(len(query) for query in normalized_queries))
    runs = []

    for top_l in args.l_values:
        generated = []
        search_seconds = 0.0
        parse_seconds = 0.0
        aggregation_seconds = 0.0
        total_retrieved_token_vectors = 0
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
            total_retrieved_token_vectors += features["retrieved_token_vectors"]
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
            exact_rankings,
            doc_ids,
            full_exact_comparisons,
        )

        policy_results = []
        for policy in POLICIES:
            for top_c in args.c_values:
                policy_results.append(
                    summarize_policy(
                        policy=policy,
                        top_l=top_l,
                        top_c=top_c,
                        generated=generated,
                        normalized_queries=normalized_queries,
                        normalized_documents=normalized_documents,
                        exact_rankings=exact_rankings,
                        doc_ids=doc_ids,
                        qrels=qrels,
                        full_exact_comparisons=full_exact_comparisons,
                    )
                )

        runs.append(
            {
                "top_l": top_l,
                "candidate_generation": {
                    "wall_seconds": wall_seconds,
                    "pdx_search_seconds": search_seconds,
                    "hit_parse_seconds": parse_seconds,
                    "aggregation_seconds": aggregation_seconds,
                    "measured_component_seconds": search_seconds + parse_seconds + aggregation_seconds,
                    "total_retrieved_token_vectors": int(total_retrieved_token_vectors),
                    "search_seconds_per_query_token": search_seconds / total_query_token_vectors,
                    "retrieved_token_vectors_per_query_token": top_l,
                    "mean_retrieved_token_fraction_per_query": (
                        total_retrieved_token_vectors
                        / len(normalized_queries)
                        / flat_tokens.shape[0]
                    ),
                },
                "pool": pool,
                "policy_results": policy_results,
            }
        )

    all_policy_results = [result for run in runs for result in run["policy_results"]]
    best_c20_recall5 = max(
        (result for result in all_policy_results if result["top_c"] == 20),
        key=lambda result: (
            result["mean_exact_recall_at_k"]["5"],
            result["exact_top5_all_recovered_count"],
            result["mean_exact_recall_at_k"]["10"],
        ),
    )
    best_c50_recall10 = max(
        (result for result in all_policy_results if result["top_c"] == 50),
        key=lambda result: (
            result["mean_exact_recall_at_k"]["10"],
            result["exact_top10_all_recovered_count"],
            result["mean_exact_recall_at_k"]["5"],
        ),
    )

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "candidate_generation": "l2sq_on_l2_normalized_vectors",
            "rerank": "exact_normalized_inner_product_maxsim",
        },
        "timing_note": (
            "This is a component timing run, not a controlled end-to-end speedup claim. "
            "It measures the current Python/WSL implementation."
        ),
        "dataset": {
            "documents": len(doc_ids),
            "queries": len(query_ids),
            "qrels_labels": int(sum(len(labels) for labels in qrels.values())),
            "document_values_shape": list(document_pack["values"].shape),
            "query_values_shape": list(query_pack["values"].shape),
            "total_document_token_vectors": int(flat_tokens.shape[0]),
            "total_query_token_vectors": total_query_token_vectors,
            "embedding_dimension": int(flat_tokens.shape[1]),
        },
        "preprocessing": {
            "normalization_seconds": normalization_seconds,
        },
        "full_exact_reference": {
            "seconds": full_exact["seconds"],
            "document_comparisons": full_exact["document_comparisons"],
            "qrels_metrics": full_exact["qrels_metrics"]["macro"],
        },
        "pdx_index": {
            "build_seconds": index_build_seconds,
            "flat_values_shape": list(flat_tokens.shape),
        },
        "l_values": args.l_values,
        "c_values": args.c_values,
        "policies": POLICIES,
        "best_results": {
            "c20_recall5": best_c20_recall5,
            "c50_recall10": best_c50_recall10,
        },
        "runs": runs,
    }
    save_json(args.output, output)

    print("SciFact timing breakdown")
    print(f"documents: {len(doc_ids)}")
    print(f"queries: {len(query_ids)}")
    print(f"document token vectors: {flat_tokens.shape[0]}")
    print(f"query token vectors: {total_query_token_vectors}")
    print(f"full exact MaxSim: {full_exact['seconds']:.6f}s")
    print(f"PDX index build: {index_build_seconds:.6f}s")
    print()
    print("Candidate generation by L")
    for run in runs:
        gen = run["candidate_generation"]
        pool = run["pool"]
        print(
            f"L={run['top_l']}: wall={gen['wall_seconds']:.6f}s "
            f"search={gen['pdx_search_seconds']:.6f}s "
            f"parse={gen['hit_parse_seconds']:.6f}s "
            f"agg={gen['aggregation_seconds']:.6f}s "
            f"pool={pool['mean_candidate_pool_size']:.2f} "
            f"pool_rec5={pool['mean_exact_recall_at_k']['5']:.4f} "
            f"pool_rec10={pool['mean_exact_recall_at_k']['10']:.4f}"
        )

    print()
    print("Best C=20 by exact recall@5")
    print(
        f"L={best_c20_recall5['top_l']} policy={best_c20_recall5['policy']} "
        f"recall@5={best_c20_recall5['mean_exact_recall_at_k']['5']:.4f} "
        f"select={best_c20_recall5['selection_seconds']:.6f}s "
        f"rerank={best_c20_recall5['rerank_seconds']:.6f}s "
        f"ratio={best_c20_recall5['reranked_comparison_ratio']:.4f}"
    )
    print("Best C=50 by exact recall@10")
    print(
        f"L={best_c50_recall10['top_l']} policy={best_c50_recall10['policy']} "
        f"recall@10={best_c50_recall10['mean_exact_recall_at_k']['10']:.4f} "
        f"select={best_c50_recall10['selection_seconds']:.6f}s "
        f"rerank={best_c50_recall10['rerank_seconds']:.6f}s "
        f"ratio={best_c50_recall10['reranked_comparison_ratio']:.4f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
