"""Coverage-preserving candidate selection for realish ColBERT/PDX retrieval.

Iteration 7 studies whether top-C candidate selection can preserve more exact
MaxSim top-k documents after PDX token retrieval. It does not change PDX,
re-encode embeddings, scale the corpus, or claim end-to-end speedup.
"""

from __future__ import annotations

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
    maxsim_scores_for_candidate_docs,
    rank_documents,
    save_json,
    topk_overlap,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
EXACT_PATH = RESULTS_DIR / "realish_colbert_maxsim_numpy.json"
OUTPUT_PATH = RESULTS_DIR / "realish_coverage_preserving_selection.json"

L_VALUES = [10, 20, 50, 100]
C_VALUES = [20, 30, 50, 75]
REPORT_K_VALUES = [1, 3, 5, 10]

ITERATION_6_BEST_C20_RECALL5 = 0.9800
ITERATION_6_BEST_C50_RECALL10 = 0.9367


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


def build_candidate_features(
    query_token_hits: list[list[tuple[int, float]]],
    token_to_doc_index: np.ndarray,
    num_documents: int,
) -> dict[str, Any]:
    """Build per-document candidate features from token retrieval hits."""
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
    matched_query_token_count = matched_mask.sum(axis=0).astype(np.int32)
    approx_scores = np.where(matched_mask, per_token_doc_best, 0.0).sum(axis=0)

    return {
        "candidate_doc_indices": candidate_doc_indices,
        "per_token_doc_best": per_token_doc_best,
        "approx_scores": approx_scores.astype(np.float32),
        "retrieved_token_count": retrieved_token_count,
        "matched_query_token_count": matched_query_token_count,
        "retrieved_token_vectors": sum(len(hits) for hits in query_token_hits),
    }


def sort_by_approx(features: dict[str, Any]) -> list[int]:
    candidates = sorted(features["candidate_doc_indices"])
    approx = features["approx_scores"]
    return sorted(candidates, key=lambda doc: (-float(approx[doc]), doc))


def sort_by_retrieved_count(features: dict[str, Any]) -> list[int]:
    candidates = sorted(features["candidate_doc_indices"])
    approx = features["approx_scores"]
    retrieved = features["retrieved_token_count"]
    return sorted(
        candidates,
        key=lambda doc: (-int(retrieved[doc]), -float(approx[doc]), doc),
    )


def sort_by_matched_coverage(features: dict[str, Any]) -> list[int]:
    candidates = sorted(features["candidate_doc_indices"])
    approx = features["approx_scores"]
    matched = features["matched_query_token_count"]
    return sorted(
        candidates,
        key=lambda doc: (-int(matched[doc]), -float(approx[doc]), doc),
    )


def trim_union_selection(
    selected: set[int],
    approx_seed: set[int],
    count_seed: set[int],
    features: dict[str, Any],
    top_c: int,
) -> list[int]:
    """Trim a union selection with deterministic preference for overlap docs."""
    approx = features["approx_scores"]
    matched = features["matched_query_token_count"]
    ordered = sorted(
        selected,
        key=lambda doc: (
            -(int(doc in approx_seed and doc in count_seed)),
            -int(matched[doc]),
            -float(approx[doc]),
            doc,
        ),
    )
    return ordered[:top_c]


def select_union_approx_and_count(features: dict[str, Any], top_c: int) -> list[int]:
    """Take about half from approximate score and half from token-hit count."""
    approx_order = sort_by_approx(features)
    count_order = sort_by_retrieved_count(features)
    first_take = math.ceil(top_c / 2)
    second_take = top_c - first_take
    approx_seed = set(approx_order[:first_take])
    count_seed = set(count_order[:second_take])
    selected = set(approx_seed).union(count_seed)

    for doc_index in approx_order:
        if len(selected) >= top_c:
            break
        selected.add(doc_index)

    return trim_union_selection(selected, approx_seed, count_seed, features, top_c)


def select_round_robin_query_token_coverage(
    features: dict[str, Any],
    top_c: int,
) -> list[int]:
    """Select documents in rounds across query-token-specific rankings."""
    per_token_doc_best = features["per_token_doc_best"]
    approx = features["approx_scores"]
    token_rankings = []
    for query_token_index in range(per_token_doc_best.shape[0]):
        docs = [
            doc_index
            for doc_index in features["candidate_doc_indices"]
            if np.isfinite(per_token_doc_best[query_token_index, doc_index])
        ]
        token_rankings.append(
            sorted(
                docs,
                key=lambda doc: (
                    -float(per_token_doc_best[query_token_index, doc]),
                    -float(approx[doc]),
                    doc,
                ),
            )
        )

    selected = []
    seen = set()
    max_depth = max((len(ranking) for ranking in token_rankings), default=0)
    for depth in range(max_depth):
        for ranking in token_rankings:
            if depth >= len(ranking):
                continue
            doc_index = ranking[depth]
            if doc_index not in seen:
                selected.append(doc_index)
                seen.add(doc_index)
                if len(selected) >= top_c:
                    return selected

    for doc_index in sort_by_approx(features):
        if doc_index not in seen:
            selected.append(doc_index)
            seen.add(doc_index)
            if len(selected) >= top_c:
                break
    return selected


def select_candidates(features: dict[str, Any], top_c: int | None, policy: str) -> list[int]:
    """Select candidates using one of the Iteration 7 policies."""
    if top_c is None:
        return sort_by_approx(features)

    if policy == "baseline_approx_score":
        return sort_by_approx(features)[:top_c]
    if policy == "baseline_retrieved_token_count":
        return sort_by_retrieved_count(features)[:top_c]
    if policy == "union_approx_and_count":
        return select_union_approx_and_count(features, top_c)
    if policy == "round_robin_query_token_coverage":
        return select_round_robin_query_token_coverage(features, top_c)
    if policy == "diversified_coverage_then_score":
        return sort_by_matched_coverage(features)[:top_c]
    if policy == "oracle_pool_upper_bound":
        return sort_by_approx(features)
    raise ValueError(f"Unknown policy: {policy}")


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
    if not candidate_scores:
        return [], perf_counter() - start

    ordered_candidates = list(candidate_scores)
    scores = np.array(
        [candidate_scores[index] for index in ordered_candidates],
        dtype=np.float32,
    )
    local_ranking = rank_documents(scores)
    return [ordered_candidates[index] for index in local_ranking], perf_counter() - start


def coverage_for_set(reference_ranking: list[str], doc_ids: set[str]) -> dict[str, Any]:
    """Measure whether an unordered candidate pool contains exact top-k docs."""
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
    """Compare a reranked candidate list to the full exact MaxSim ranking."""
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
    """Classify whether top-k recovery failed due to pool or selector."""
    exact_top_k = set(reference_ranking[:k])
    if exact_top_k.issubset(selected_doc_ids):
        return "none"
    if not exact_top_k.issubset(pool_doc_ids):
        return "pool_coverage_failure"
    return "selector_failure"


def candidate_diagnostics(
    *,
    selected_doc_indices: list[int],
    features: dict[str, Any],
    exact_ranking: list[str],
    doc_ids: list[str],
    query_token_count: int,
) -> list[dict[str, Any]]:
    """Create per-selected-candidate diagnostic records."""
    exact_top_sets = {
        "1": set(exact_ranking[:1]),
        "3": set(exact_ranking[:3]),
        "5": set(exact_ranking[:5]),
        "10": set(exact_ranking[:10]),
    }
    records = []
    per_token_doc_best = features["per_token_doc_best"]
    for doc_index in selected_doc_indices:
        doc_id = doc_ids[doc_index]
        best_values = per_token_doc_best[:, doc_index]
        matched_values = best_values[np.isfinite(best_values)]
        if len(matched_values):
            average_best = float(np.mean(matched_values))
            min_best = float(np.min(matched_values))
            max_best = float(np.max(matched_values))
        else:
            average_best = 0.0
            min_best = 0.0
            max_best = 0.0
        matched_count = int(features["matched_query_token_count"][doc_index])
        records.append(
            {
                "doc_index": int(doc_index),
                "doc_id": doc_id,
                "approx_score": float(features["approx_scores"][doc_index]),
                "retrieved_token_count": int(features["retrieved_token_count"][doc_index]),
                "matched_query_token_count": matched_count,
                "matched_query_token_fraction": matched_count / query_token_count,
                "average_best_token_similarity": average_best,
                "min_best_token_similarity": min_best,
                "max_best_token_similarity": max_best,
                "in_exact_top1": doc_id in exact_top_sets["1"],
                "in_exact_top3": doc_id in exact_top_sets["3"],
                "in_exact_top5": doc_id in exact_top_sets["5"],
                "in_exact_top10": doc_id in exact_top_sets["10"],
            }
        )
    return records


def summarize_pool_coverage(pool_query_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate candidate-pool coverage over all queries."""
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


def aggregate_policy_result(
    *,
    policy: str,
    top_l: int,
    top_c: int | None,
    generated: list[dict[str, Any]],
    normalized_queries: list[np.ndarray],
    normalized_documents: list[np.ndarray],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
    qrels: dict[str, list[str]],
    full_exact_comparisons: int,
    candidate_generation_seconds: float,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Select, rerank, compare, and aggregate one policy/L/C setting."""
    query_results = []
    rankings_by_query = {}
    total_rerank_seconds = 0.0
    reranked_comparisons = 0
    total_candidate_pool_size = 0
    loss_counts = {
        "5": {"pool_coverage_failure": 0, "selector_failure": 0, "none": 0},
        "10": {"pool_coverage_failure": 0, "selector_failure": 0, "none": 0},
    }

    for item in generated:
        query_id = item["query_id"]
        query_index = item["query_index"]
        features = item["features"]
        selected = select_candidates(features, top_c, policy)
        reranked, rerank_seconds = rerank_candidates(
            normalized_queries[query_index],
            normalized_documents,
            selected,
        )
        total_rerank_seconds += rerank_seconds
        reranked_comparisons += len(selected)
        total_candidate_pool_size += len(features["candidate_doc_indices"])

        pool_doc_ids = {doc_ids[index] for index in features["candidate_doc_indices"]}
        selected_doc_ids = {doc_ids[index] for index in selected}
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

        query_result = {
            "query_id": query_id,
            "candidate_pool_size": len(features["candidate_doc_indices"]),
            "selected_candidate_count": len(selected),
            "selected_doc_ids": [doc_ids[index] for index in selected],
            "reranked_doc_ids": reranked_doc_ids,
            "exact_rerank_seconds": rerank_seconds,
            "comparison_to_exact": compare_reranked(
                exact_rankings[query_id],
                reranked_doc_ids,
            ),
        }
        if include_diagnostics:
            query_result["selected_candidate_diagnostics"] = candidate_diagnostics(
                selected_doc_indices=selected,
                features=features,
                exact_ranking=exact_rankings[query_id],
                doc_ids=doc_ids,
                query_token_count=len(normalized_queries[query_index]),
            )
        query_results.append(query_result)

    qrels_metrics = compute_qrels_metrics(
        rankings_by_query,
        qrels,
        k_values=(1, 3, 5, 10),
        mrr_k=10,
    )
    query_count = len(generated)
    return {
        "policy": policy,
        "top_l": top_l,
        "top_c": top_c,
        "candidate_generation_seconds": candidate_generation_seconds,
        "total_rerank_seconds": total_rerank_seconds,
        "full_exact_comparisons": full_exact_comparisons,
        "reranked_comparisons": reranked_comparisons,
        "reranked_comparison_ratio": reranked_comparisons / full_exact_comparisons,
        "mean_candidate_pool_size": total_candidate_pool_size / query_count,
        "mean_selected_candidate_count": reranked_comparisons / query_count,
        "mean_exact_recall_at_k": {
            str(k): float(np.mean([
                result["comparison_to_exact"]["exact_recall_at_k"][str(k)]
                for result in query_results
            ]))
            for k in REPORT_K_VALUES
        },
        "exact_top1_match_count": sum(
            1 for result in query_results if result["comparison_to_exact"]["exact_top1_match"]
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
        "queries": query_results,
    }


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
    full_exact_comparisons = len(normalized_queries) * len(normalized_documents)
    total_query_token_vectors = int(sum(len(query) for query in normalized_queries))

    np.random.seed(0)
    index = IndexPDXBONDFlat(ndim=flat_tokens.shape[1])
    start = perf_counter()
    index.add_load(flat_tokens)
    index_build_seconds = perf_counter() - start

    policies = [
        "baseline_approx_score",
        "baseline_retrieved_token_count",
        "union_approx_and_count",
        "round_robin_query_token_coverage",
        "diversified_coverage_then_score",
    ]

    runs = []
    for top_l in L_VALUES:
        generated = []
        total_generation_seconds = 0.0
        total_retrieved_token_vectors = 0
        pool_query_results = []

        for query_index, query_matrix in enumerate(normalized_queries):
            query_id = query_pack["ids"][query_index]
            query_token_hits, generation_seconds = retrieve_query_token_hits(
                index,
                query_matrix,
                top_l,
            )
            features = build_candidate_features(
                query_token_hits,
                token_to_doc_index,
                num_documents=len(doc_ids),
            )
            candidate_doc_ids = {
                doc_ids[index] for index in features["candidate_doc_indices"]
            }
            coverage = coverage_for_set(exact_rankings[query_id], candidate_doc_ids)
            generated.append(
                {
                    "query_id": query_id,
                    "query_index": query_index,
                    "features": features,
                    "candidate_generation_seconds": generation_seconds,
                    "coverage": coverage,
                }
            )
            total_generation_seconds += generation_seconds
            total_retrieved_token_vectors += features["retrieved_token_vectors"]
            pool_query_results.append(
                {
                    "query_id": query_id,
                    "unique_candidate_documents": len(features["candidate_doc_indices"]),
                    "coverage": coverage,
                }
            )

        pool_summary = summarize_pool_coverage(pool_query_results)
        policy_results = []
        for policy in policies:
            for top_c in C_VALUES:
                policy_results.append(
                    aggregate_policy_result(
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
                        candidate_generation_seconds=total_generation_seconds,
                    )
                )

        oracle_result = aggregate_policy_result(
            policy="oracle_pool_upper_bound",
            top_l=top_l,
            top_c=None,
            generated=generated,
            normalized_queries=normalized_queries,
            normalized_documents=normalized_documents,
            exact_rankings=exact_rankings,
            doc_ids=doc_ids,
            qrels=qrels,
            full_exact_comparisons=full_exact_comparisons,
            candidate_generation_seconds=total_generation_seconds,
        )

        runs.append(
            {
                "top_l": top_l,
                "candidate_generation_seconds": total_generation_seconds,
                "total_retrieved_token_vectors": total_retrieved_token_vectors,
                "token_retrieval_estimate": {
                    "total_query_token_vectors": total_query_token_vectors,
                    "total_document_token_vectors": int(flat_tokens.shape[0]),
                    "mean_retrieved_token_vectors_per_query": (
                        total_retrieved_token_vectors / len(normalized_queries)
                    ),
                    "mean_retrieved_token_fraction_per_query": (
                        (total_retrieved_token_vectors / len(normalized_queries))
                        / int(flat_tokens.shape[0])
                    ),
                },
                "candidate_pool_coverage": {
                    "summary": pool_summary,
                    "queries": pool_query_results,
                },
                "policy_results": policy_results,
                "oracle_pool_upper_bound": oracle_result,
            }
        )

    def best_policy(top_c: int, recall_key: str) -> tuple[int, dict[str, Any]]:
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
                item[1]["exact_top5_all_recovered_count"],
                -item[1]["reranked_comparison_ratio"],
            ),
        )

    best_c20_recall5 = best_policy(20, "5")
    best_c50_recall10 = best_policy(50, "10")
    tolerance = 1e-6
    improved_over_iteration_6 = {
        "c20_recall5": (
            best_c20_recall5[1]["mean_exact_recall_at_k"]["5"]
            > ITERATION_6_BEST_C20_RECALL5 + tolerance
        ),
        "c50_recall10": (
            best_c50_recall10[1]["mean_exact_recall_at_k"]["10"]
            > ITERATION_6_BEST_C50_RECALL10 + tolerance
        ),
    }

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "used": "l2sq_on_l2_normalized_vectors_for_candidate_generation",
            "rerank_metric": "exact_normalized_inner_product_maxsim",
            "missing_contribution_for_approx_score": 0.0,
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
            "total_document_token_vectors": int(flat_tokens.shape[0]),
            "total_query_token_vectors": total_query_token_vectors,
        },
        "token_index": {
            "flat_values_shape": list(flat_tokens.shape),
            "token_to_doc_index": token_to_doc_index.tolist(),
            "token_to_token_position": token_to_token_position.tolist(),
            "index_build_seconds": index_build_seconds,
        },
        "l_values": L_VALUES,
        "c_values": C_VALUES,
        "policies": policies + ["oracle_pool_upper_bound"],
        "iteration_6_reference": {
            "best_c20_recall5": ITERATION_6_BEST_C20_RECALL5,
            "best_c50_recall10": ITERATION_6_BEST_C50_RECALL10,
        },
        "best_results": {
            "c20_recall5": {
                "top_l": best_c20_recall5[0],
                "result": {
                    key: value
                    for key, value in best_c20_recall5[1].items()
                    if key != "queries"
                },
            },
            "c50_recall10": {
                "top_l": best_c50_recall10[0],
                "result": {
                    key: value
                    for key, value in best_c50_recall10[1].items()
                    if key != "queries"
                },
            },
            "improved_over_iteration_6": improved_over_iteration_6,
        },
        "runs": runs,
        "note": (
            "Comparison ratios count exact MaxSim document rerank calls only. "
            "They are not end-to-end speedup claims."
        ),
    }
    save_json(OUTPUT_PATH, output)

    print("Realish coverage-preserving candidate selection")
    print(f"PDX supported metrics: {supported_metrics}")
    print(f"documents: {len(doc_ids)}")
    print(f"queries: {len(normalized_queries)}")
    print(f"flat token matrix: {flat_tokens.shape}")
    print(f"index build time: {index_build_seconds:.6f}s")
    print(f"full exact document comparisons: {full_exact_comparisons}")

    print("\nCandidate pool coverage by L")
    for run in runs:
        summary = run["candidate_pool_coverage"]["summary"]
        oracle = run["oracle_pool_upper_bound"]
        print(
            f"L={run['top_l']}: "
            f"pool docs mean={summary['mean_unique_candidate_documents']:.2f}, "
            f"pool top1/top3/top5/top10="
            f"{summary['exact_top1_in_pool']}/"
            f"{summary['exact_top3_all_in_pool']}/"
            f"{summary['exact_top5_all_in_pool']}/"
            f"{summary['exact_top10_all_in_pool']}, "
            f"oracle top10={oracle['exact_top10_all_recovered_count']}/"
            f"{len(normalized_queries)}, "
            f"oracle ratio={oracle['reranked_comparison_ratio']:.4f}"
        )

    print("\nBest policy for C=20 by exact recall@5")
    print(
        f"L={best_c20_recall5[0]}, policy={best_c20_recall5[1]['policy']}, "
        f"recall@5={best_c20_recall5[1]['mean_exact_recall_at_k']['5']:.4f}, "
        f"top5 all={best_c20_recall5[1]['exact_top5_all_recovered_count']}/"
        f"{len(normalized_queries)}, "
        f"ratio={best_c20_recall5[1]['reranked_comparison_ratio']:.4f}, "
        f"loss@5={best_c20_recall5[1]['selector_loss']['5']}"
    )

    print("\nBest policy for C=50 by exact recall@10")
    print(
        f"L={best_c50_recall10[0]}, policy={best_c50_recall10[1]['policy']}, "
        f"recall@10={best_c50_recall10[1]['mean_exact_recall_at_k']['10']:.4f}, "
        f"top10 all={best_c50_recall10[1]['exact_top10_all_recovered_count']}/"
        f"{len(normalized_queries)}, "
        f"ratio={best_c50_recall10[1]['reranked_comparison_ratio']:.4f}, "
        f"loss@10={best_c50_recall10[1]['selector_loss']['10']}"
    )

    print("\nImproves over Iteration 6")
    print(f"C=20 recall@5: {improved_over_iteration_6['c20_recall5']}")
    print(f"C=50 recall@10: {improved_over_iteration_6['c50_recall10']}")

    if best_c20_recall5[1]["selector_loss"]["5"]["selector_failure"] > 0:
        failure_note = "remaining top-5 misses are selector failures"
    else:
        failure_note = "top-5 misses are not selector-driven at the best C=20 setting"
    print(f"\nFailure mode: {failure_note}")
    print(
        "Recommended policy: "
        f"{best_c20_recall5[1]['policy']} for C=20 when prioritizing top-5, "
        f"{best_c50_recall10[1]['policy']} for C=50 when prioritizing top-10."
    )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
