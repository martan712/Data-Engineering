"""Study top-C candidate selection policies for realish ColBERT/PDX retrieval.

This script separates two failure modes:

1. Candidate pool coverage: did PDX token retrieval find the exact MaxSim top-k
   documents anywhere in the unique document pool?
2. Selector loss: were those documents in the pool but removed by the top-C
   selection policy before exact MaxSim reranking?

The experiment still uses PDX-BOND only as token-level candidate generation and
uses exact normalized ColBERT MaxSim for final reranking.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from utils_colbert import (
    aggregate_token_retrieval_scores,
    compute_qrels_metrics,
    flatten_document_embeddings,
    l2_normalize,
    load_packed_embeddings,
    maxsim_scores_for_candidate_docs,
    rank_documents,
    save_json,
    topk_overlap,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
EXACT_PATH = RESULTS_DIR / "realish_colbert_maxsim_numpy.json"
OUTPUT_PATH = RESULTS_DIR / "realish_candidate_selection_study.json"
L_VALUES = [5, 10, 20, 50, 100, 200]
C_VALUES = [10, 20, 50, 100]
REPORT_K_VALUES = [1, 3, 5, 10]
HYBRID_ALPHAS = [0.25, 0.5, 1.0, 2.0]


def import_pdx_api():
    """Import installed PDX API, falling back to the local source checkout."""
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


def retrieve_query_token_hits(index, query_matrix: np.ndarray, top_l: int):
    """Retrieve PDX token hits and convert squared L2 to cosine similarity."""
    query_token_hits = []
    start = perf_counter()
    for query_token in query_matrix:
        hits = index.search(np.ascontiguousarray(query_token, dtype=np.float32), top_l)
        query_token_hits.append(
            [
                (int(hit.index), 1.0 - (float(hit.distance) / 2.0))
                for hit in hits
            ]
        )
    return query_token_hits, perf_counter() - start


def candidate_features(
    query_token_hits: list[list[tuple[int, float]]],
    token_to_doc_index: np.ndarray,
    num_documents: int,
) -> dict[str, Any]:
    """Compute candidate document features from token-level hits."""
    approx_scores, candidate_doc_indices, retrieved_tokens = aggregate_token_retrieval_scores(
        query_token_hits,
        token_to_doc_index,
        num_documents=num_documents,
        missing_value=0.0,
    )
    matched_query_token_count = np.zeros(num_documents, dtype=np.int32)
    retrieved_token_count = np.zeros(num_documents, dtype=np.int32)

    for query_token_hits_for_one_token in query_token_hits:
        docs_for_query_token = set()
        for token_index, _similarity in query_token_hits_for_one_token:
            doc_index = int(token_to_doc_index[int(token_index)])
            docs_for_query_token.add(doc_index)
            retrieved_token_count[doc_index] += 1
        for doc_index in docs_for_query_token:
            matched_query_token_count[doc_index] += 1

    return {
        "approx_scores": approx_scores,
        "candidate_doc_indices": set(candidate_doc_indices),
        "matched_query_token_count": matched_query_token_count,
        "retrieved_token_count": retrieved_token_count,
        "retrieved_token_vectors": retrieved_tokens,
    }


def normalize_feature(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Min-max normalize a feature over candidate documents."""
    candidate_values = values[candidates].astype(np.float32)
    if len(candidate_values) == 0:
        return candidate_values
    min_value = float(np.min(candidate_values))
    max_value = float(np.max(candidate_values))
    if max_value <= min_value:
        return np.zeros_like(candidate_values, dtype=np.float32)
    return (candidate_values - min_value) / (max_value - min_value)


def select_candidates(
    features: dict[str, Any],
    top_c: int | None,
    policy: str,
    alpha: float | None = None,
) -> list[int]:
    """Select candidates by a deterministic top-C policy."""
    candidates = np.array(sorted(features["candidate_doc_indices"]), dtype=np.int64)
    if len(candidates) == 0:
        return []
    if top_c is None:
        return candidates.astype(int).tolist()

    approx = features["approx_scores"]
    matched = features["matched_query_token_count"]
    token_hits = features["retrieved_token_count"]

    if policy == "approx_score":
        primary = approx[candidates].astype(np.float32)
        order = np.lexsort((candidates, -primary))
    elif policy == "matched_query_token_count":
        primary = matched[candidates].astype(np.float32)
        secondary = approx[candidates].astype(np.float32)
        order = np.lexsort((candidates, -secondary, -primary))
    elif policy == "retrieved_token_count":
        primary = token_hits[candidates].astype(np.float32)
        secondary = approx[candidates].astype(np.float32)
        order = np.lexsort((candidates, -secondary, -primary))
    elif policy == "hybrid_score_coverage":
        if alpha is None:
            raise ValueError("hybrid_score_coverage requires alpha")
        normalized_approx = normalize_feature(approx, candidates)
        normalized_matched = normalize_feature(matched.astype(np.float32), candidates)
        hybrid = normalized_approx + alpha * normalized_matched
        order = np.lexsort((candidates, -approx[candidates], -hybrid))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return candidates[order[:top_c]].astype(int).tolist()


def rerank_candidates(
    query_matrix: np.ndarray,
    documents: list[np.ndarray],
    selected_doc_indices: list[int],
) -> tuple[list[int], float]:
    """Rerank selected candidates with exact normalized MaxSim."""
    start = perf_counter()
    candidate_scores = maxsim_scores_for_candidate_docs(
        query_matrix,
        documents,
        selected_doc_indices,
        normalize=False,
    )
    if candidate_scores:
        ordered_candidates = list(candidate_scores)
        scores = np.array(
            [candidate_scores[index] for index in ordered_candidates],
            dtype=np.float32,
        )
        local_ranking = rank_documents(scores)
        reranked = [ordered_candidates[index] for index in local_ranking]
    else:
        reranked = []
    return reranked, perf_counter() - start


def coverage_for_docs(reference_ranking: list[str], doc_ids: set[str]) -> dict[str, Any]:
    """Measure whether an unordered document set contains exact top-k docs."""
    return {
        "exact_top1_in_pool": bool(reference_ranking[:1] and reference_ranking[0] in doc_ids),
        "exact_top3_all_in_pool": set(reference_ranking[:3]).issubset(doc_ids),
        "exact_top5_all_in_pool": set(reference_ranking[:5]).issubset(doc_ids),
        "exact_top10_all_in_pool": set(reference_ranking[:10]).issubset(doc_ids),
        "exact_recall_at_k": {
            str(k): len(set(reference_ranking[:k]).intersection(doc_ids)) / k
            for k in REPORT_K_VALUES
        },
    }


def compare_reranked(reference_ranking: list[str], reranked_doc_ids: list[str]) -> dict[str, Any]:
    """Compare exact-reranked candidates to the full exact ranking."""
    return {
        "exact_top1_match": reference_ranking[:1] == reranked_doc_ids[:1],
        "exact_top3_all_recovered": set(reference_ranking[:3]).issubset(set(reranked_doc_ids[:3])),
        "exact_top5_all_recovered": set(reference_ranking[:5]).issubset(set(reranked_doc_ids[:5])),
        "exact_top10_all_recovered": set(reference_ranking[:10]).issubset(set(reranked_doc_ids[:10])),
        "exact_recall_at_k": {
            str(k): topk_overlap(reference_ranking, reranked_doc_ids, k)["ratio"]
            for k in REPORT_K_VALUES
        },
    }


def selector_loss(
    reference_ranking: list[str],
    pool_doc_ids: set[str],
    selected_doc_ids: set[str],
    k: int,
) -> str:
    """Classify top-k failure source for one query."""
    exact_top_k = set(reference_ranking[:k])
    if exact_top_k.issubset(selected_doc_ids):
        return "none"
    if not exact_top_k.issubset(pool_doc_ids):
        return "pool_coverage_failure"
    return "selector_failure"


def summarize_pool_coverage(pool_query_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate candidate-pool coverage over queries."""
    return {
        "mean_unique_candidate_documents": float(np.mean([
            result["unique_candidate_documents"] for result in pool_query_results
        ])),
        "exact_top1_in_pool": sum(
            1 for result in pool_query_results if result["coverage"]["exact_top1_in_pool"]
        ),
        "exact_top3_all_in_pool": sum(
            1 for result in pool_query_results if result["coverage"]["exact_top3_all_in_pool"]
        ),
        "exact_top5_all_in_pool": sum(
            1 for result in pool_query_results if result["coverage"]["exact_top5_all_in_pool"]
        ),
        "exact_top10_all_in_pool": sum(
            1 for result in pool_query_results if result["coverage"]["exact_top10_all_in_pool"]
        ),
        "mean_exact_recall_at_k": {
            str(k): float(np.mean([
                result["coverage"]["exact_recall_at_k"][str(k)]
                for result in pool_query_results
            ]))
            for k in REPORT_K_VALUES
        },
    }


def policy_name(policy: str, alpha: float | None = None) -> str:
    if policy == "hybrid_score_coverage":
        return f"{policy}_alpha_{alpha:g}"
    return policy


def main() -> None:
    IndexPDXBONDFlat, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")
    if not EXACT_PATH.exists():
        raise SystemExit("Run 13_realish_colbert_maxsim_numpy.py before this script.")

    with EXACT_PATH.open("r", encoding="utf-8") as file:
        exact = json.load(file)
    with (CORPUS_DIR / "qrels.json").open("r", encoding="utf-8") as file:
        qrels = json.load(file)

    document_pack = load_packed_embeddings(EMBEDDINGS_DIR / "documents_packed.npz")
    query_pack = load_packed_embeddings(EMBEDDINGS_DIR / "queries_packed.npz")
    documents = unpack_embeddings(document_pack["values"], document_pack["offsets"])
    queries = unpack_embeddings(query_pack["values"], query_pack["offsets"])
    normalized_documents = [l2_normalize(document) for document in documents]
    normalized_queries = [l2_normalize(query) for query in queries]
    flat_tokens, token_to_doc_index, token_to_token_position = flatten_document_embeddings(
        normalized_documents
    )

    doc_ids = document_pack["ids"]
    exact_rankings = {
        result["query_id"]: result["ranking_doc_ids"]
        for result in exact["query_results"]
    }

    np.random.seed(0)
    index = IndexPDXBONDFlat(ndim=flat_tokens.shape[1])
    start = perf_counter()
    index.add_load(flat_tokens)
    index_build_time = perf_counter() - start

    full_exact_comparisons = len(normalized_queries) * len(normalized_documents)
    total_query_token_vectors = int(sum(len(query) for query in normalized_queries))
    total_document_token_vectors = int(flat_tokens.shape[0])
    selectable_policies: list[tuple[str, float | None]] = [
        ("approx_score", None),
        ("matched_query_token_count", None),
        ("retrieved_token_count", None),
    ] + [("hybrid_score_coverage", alpha) for alpha in HYBRID_ALPHAS]

    runs = []
    for top_l in L_VALUES:
        if top_l > total_document_token_vectors:
            continue

        generated = []
        total_generation_time = 0.0
        total_retrieved_tokens = 0
        pool_query_results = []

        for query_index, query_matrix in enumerate(normalized_queries):
            query_id = query_pack["ids"][query_index]
            query_token_hits, retrieval_time = retrieve_query_token_hits(index, query_matrix, top_l)
            features = candidate_features(
                query_token_hits,
                token_to_doc_index,
                num_documents=len(doc_ids),
            )
            candidate_doc_indices = features["candidate_doc_indices"]
            candidate_doc_ids = {doc_ids[index] for index in candidate_doc_indices}
            coverage = coverage_for_docs(exact_rankings[query_id], candidate_doc_ids)

            generated.append(
                {
                    "query_id": query_id,
                    "query_index": query_index,
                    "features": features,
                    "generation_seconds": retrieval_time,
                    "coverage": coverage,
                }
            )
            total_generation_time += retrieval_time
            total_retrieved_tokens += features["retrieved_token_vectors"]
            pool_query_results.append(
                {
                    "query_id": query_id,
                    "unique_candidate_documents": len(candidate_doc_indices),
                    "coverage": coverage,
                }
            )

        pool_summary = summarize_pool_coverage(pool_query_results)
        policy_results = []

        for policy, alpha in selectable_policies:
            for top_c in C_VALUES:
                if top_c > len(doc_ids):
                    continue

                query_results = []
                rankings_by_query = {}
                total_rerank_time = 0.0
                reranked_comparisons = 0
                loss_counts = {
                    "5": {"pool_coverage_failure": 0, "selector_failure": 0, "none": 0},
                    "10": {"pool_coverage_failure": 0, "selector_failure": 0, "none": 0},
                }

                for item in generated:
                    query_id = item["query_id"]
                    query_index = item["query_index"]
                    features = item["features"]
                    selected = select_candidates(features, top_c, policy, alpha)
                    reranked, rerank_time = rerank_candidates(
                        normalized_queries[query_index],
                        normalized_documents,
                        selected,
                    )
                    total_rerank_time += rerank_time
                    reranked_comparisons += len(selected)

                    selected_doc_ids = {doc_ids[index] for index in selected}
                    pool_doc_ids = {
                        doc_ids[index] for index in features["candidate_doc_indices"]
                    }
                    reranked_doc_ids = [doc_ids[index] for index in reranked]
                    rankings_by_query[query_id] = reranked_doc_ids
                    for k in (5, 10):
                        loss_counts[str(k)][
                            selector_loss(
                                exact_rankings[query_id],
                                pool_doc_ids,
                                selected_doc_ids,
                                k,
                            )
                        ] += 1

                    query_results.append(
                        {
                            "query_id": query_id,
                            "selected_count": len(selected),
                            "rerank_seconds": rerank_time,
                            "comparison_to_exact": compare_reranked(
                                exact_rankings[query_id],
                                reranked_doc_ids,
                            ),
                        }
                    )

                qrels_metrics = compute_qrels_metrics(
                    rankings_by_query,
                    qrels,
                    k_values=(1, 3, 5, 10),
                    mrr_k=10,
                )
                policy_results.append(
                    {
                        "policy": policy_name(policy, alpha),
                        "base_policy": policy,
                        "alpha": alpha,
                        "top_c": top_c,
                        "total_rerank_seconds": total_rerank_time,
                        "full_exact_comparisons": full_exact_comparisons,
                        "reranked_comparisons": reranked_comparisons,
                        "reranked_comparison_ratio": reranked_comparisons / full_exact_comparisons,
                        "mean_candidate_count": reranked_comparisons / len(normalized_queries),
                        "mean_exact_recall_at_k": {
                            str(k): float(np.mean([
                                result["comparison_to_exact"]["exact_recall_at_k"][str(k)]
                                for result in query_results
                            ]))
                            for k in REPORT_K_VALUES
                        },
                        "exact_top1_match_count": sum(
                            1
                            for result in query_results
                            if result["comparison_to_exact"]["exact_top1_match"]
                        ),
                        "exact_top3_all_recovered_count": sum(
                            1
                            for result in query_results
                            if result["comparison_to_exact"]["exact_top3_all_recovered"]
                        ),
                        "exact_top5_all_recovered_count": sum(
                            1
                            for result in query_results
                            if result["comparison_to_exact"]["exact_top5_all_recovered"]
                        ),
                        "exact_top10_all_recovered_count": sum(
                            1
                            for result in query_results
                            if result["comparison_to_exact"]["exact_top10_all_recovered"]
                        ),
                        "selector_loss": loss_counts,
                        "qrels_metrics": qrels_metrics,
                    }
                )

        all_unique_query_results = []
        all_unique_rankings_by_query = {}
        all_unique_rerank_time = 0.0
        all_unique_comparisons = 0
        for item in generated:
            query_id = item["query_id"]
            query_index = item["query_index"]
            selected = sorted(item["features"]["candidate_doc_indices"])
            reranked, rerank_time = rerank_candidates(
                normalized_queries[query_index],
                normalized_documents,
                selected,
            )
            all_unique_rerank_time += rerank_time
            all_unique_comparisons += len(selected)
            reranked_doc_ids = [doc_ids[index] for index in reranked]
            all_unique_rankings_by_query[query_id] = reranked_doc_ids
            all_unique_query_results.append(
                {
                    "query_id": query_id,
                    "selected_count": len(selected),
                    "comparison_to_exact": compare_reranked(
                        exact_rankings[query_id],
                        reranked_doc_ids,
                    ),
                }
            )

        all_unique_result = {
            "policy": "all_unique_candidates",
            "total_rerank_seconds": all_unique_rerank_time,
            "full_exact_comparisons": full_exact_comparisons,
            "reranked_comparisons": all_unique_comparisons,
            "reranked_comparison_ratio": all_unique_comparisons / full_exact_comparisons,
            "mean_candidate_count": all_unique_comparisons / len(normalized_queries),
            "mean_exact_recall_at_k": {
                str(k): float(np.mean([
                    result["comparison_to_exact"]["exact_recall_at_k"][str(k)]
                    for result in all_unique_query_results
                ]))
                for k in REPORT_K_VALUES
            },
            "exact_top1_match_count": sum(
                1
                for result in all_unique_query_results
                if result["comparison_to_exact"]["exact_top1_match"]
            ),
            "exact_top3_all_recovered_count": sum(
                1
                for result in all_unique_query_results
                if result["comparison_to_exact"]["exact_top3_all_recovered"]
            ),
            "exact_top5_all_recovered_count": sum(
                1
                for result in all_unique_query_results
                if result["comparison_to_exact"]["exact_top5_all_recovered"]
            ),
            "exact_top10_all_recovered_count": sum(
                1
                for result in all_unique_query_results
                if result["comparison_to_exact"]["exact_top10_all_recovered"]
            ),
            "qrels_metrics": compute_qrels_metrics(
                all_unique_rankings_by_query,
                qrels,
                k_values=(1, 3, 5, 10),
                mrr_k=10,
            ),
        }

        runs.append(
            {
                "top_l": top_l,
                "candidate_generation_seconds": total_generation_time,
                "total_retrieved_token_vectors": total_retrieved_tokens,
                "token_retrieval_estimate": {
                    "total_query_token_vectors": total_query_token_vectors,
                    "total_document_token_vectors": total_document_token_vectors,
                    "mean_retrieved_token_vectors_per_query": (
                        total_retrieved_tokens / len(normalized_queries)
                    ),
                    "mean_retrieved_token_fraction_per_query": (
                        (total_retrieved_tokens / len(normalized_queries))
                        / total_document_token_vectors
                    ),
                },
                "candidate_pool_coverage": {
                    "summary": pool_summary,
                    "queries": pool_query_results,
                },
                "policy_results": policy_results,
                "all_unique_candidates": all_unique_result,
            }
        )

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "used": "l2sq_on_l2_normalized_vectors_for_candidate_generation",
            "rerank_metric": "exact_normalized_inner_product_maxsim",
            "missing_contribution": 0.0,
        },
        "documents": {
            "count": len(doc_ids),
            "values_shape": list(document_pack["values"].shape),
            "offsets": document_pack["offsets"].tolist(),
        },
        "queries": {
            "count": len(normalized_queries),
            "values_shape": list(query_pack["values"].shape),
            "offsets": query_pack["offsets"].tolist(),
        },
        "work_reference": {
            "full_exact_document_comparisons": full_exact_comparisons,
            "total_document_token_vectors": total_document_token_vectors,
            "total_query_token_vectors": total_query_token_vectors,
        },
        "token_index": {
            "flat_values_shape": list(flat_tokens.shape),
            "token_to_doc_index": token_to_doc_index.tolist(),
            "token_to_token_position": token_to_token_position.tolist(),
            "index_build_seconds": index_build_time,
        },
        "l_values": L_VALUES,
        "c_values": C_VALUES,
        "hybrid_alphas": HYBRID_ALPHAS,
        "runs": runs,
        "note": (
            "Comparison ratios count exact MaxSim document rerank calls only. "
            "They are not end-to-end speedup claims."
        ),
    }
    save_json(OUTPUT_PATH, output)

    print("Realish candidate selection study")
    print(f"PDX supported metrics: {supported_metrics}")
    print(f"documents: {len(doc_ids)}")
    print(f"queries: {len(normalized_queries)}")
    print(f"flat token matrix: {flat_tokens.shape}")
    print(f"index build time: {index_build_time:.6f}s")
    print(f"full exact document comparisons: {full_exact_comparisons}")

    print("\nCandidate pool coverage by L")
    for run in runs:
        summary = run["candidate_pool_coverage"]["summary"]
        all_unique = run["all_unique_candidates"]
        print(
            f"L={run['top_l']}: "
            f"pool docs mean={summary['mean_unique_candidate_documents']:.2f}, "
            f"pool top1/top3/top5/top10="
            f"{summary['exact_top1_in_pool']}/"
            f"{summary['exact_top3_all_in_pool']}/"
            f"{summary['exact_top5_all_in_pool']}/"
            f"{summary['exact_top10_all_in_pool']}, "
            f"all_unique top10={all_unique['exact_top10_all_recovered_count']}/"
            f"{len(normalized_queries)}, "
            f"comparison ratio={all_unique['reranked_comparison_ratio']:.4f}"
        )

    def best_for(top_c: int, recall_key: str):
        candidates = []
        for run in runs:
            for result in run["policy_results"]:
                if result["top_c"] == top_c:
                    candidates.append((run["top_l"], result))
        return max(
            candidates,
            key=lambda item: (
                item[1]["mean_exact_recall_at_k"][recall_key],
                item[1]["exact_top10_all_recovered_count"],
                -item[1]["reranked_comparison_ratio"],
            ),
        )

    best_c20 = best_for(20, "5")
    best_c50 = best_for(50, "10")
    print("\nBest policy for exact recall@5 with C=20")
    print(
        f"L={best_c20[0]}, policy={best_c20[1]['policy']}, "
        f"recall@5={best_c20[1]['mean_exact_recall_at_k']['5']:.4f}, "
        f"top5 all={best_c20[1]['exact_top5_all_recovered_count']}/"
        f"{len(normalized_queries)}, "
        f"ratio={best_c20[1]['reranked_comparison_ratio']:.4f}, "
        f"loss@5={best_c20[1]['selector_loss']['5']}"
    )

    print("\nBest policy for exact recall@10 with C=50")
    print(
        f"L={best_c50[0]}, policy={best_c50[1]['policy']}, "
        f"recall@10={best_c50[1]['mean_exact_recall_at_k']['10']:.4f}, "
        f"top10 all={best_c50[1]['exact_top10_all_recovered_count']}/"
        f"{len(normalized_queries)}, "
        f"ratio={best_c50[1]['reranked_comparison_ratio']:.4f}, "
        f"loss@10={best_c50[1]['selector_loss']['10']}"
    )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
