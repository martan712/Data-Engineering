"""PDX candidate generation + exact MaxSim reranking for SciFact subset."""

from __future__ import annotations

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
EXACT_PATH = RESULTS_DIR / "scifact_colbert_maxsim_numpy.json"
OUTPUT_PATH = RESULTS_DIR / "scifact_pdx_candidates_rerank.json"
L_VALUES = [10, 20, 50, 100]
C_VALUES = [20, 50, 100]
REPORT_K_VALUES = [1, 3, 5, 10]
POLICIES = [
    "approx_score",
    "retrieved_token_count",
    "union_approx_and_count",
]


def import_pdx_api():
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


def select_candidates(features: dict[str, Any], top_c: int | None, policy: str) -> list[int]:
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
        selected = set(approx_order[:first_take]).union(count_order[: top_c - first_take])
        for doc_index in approx_order:
            if len(selected) >= top_c:
                break
            selected.add(doc_index)
        return sorted(
            selected,
            key=lambda doc: (
                -(doc in set(approx_order[:first_take]) and doc in set(count_order[: top_c - first_take])),
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


def compare_exact(reference: list[str], reranked: list[str], selected: set[str], pool: set[str]) -> dict:
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
    top_c: int | None,
    generated: list[dict[str, Any]],
    normalized_queries: list[np.ndarray],
    normalized_documents: list[np.ndarray],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
    qrels: dict[str, list[str]],
    full_exact_comparisons: int,
    candidate_generation_seconds: float,
) -> dict[str, Any]:
    query_results = []
    rankings_by_query = {}
    total_rerank_seconds = 0.0
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
        selected_doc_ids = {doc_ids[index] for index in selected}
        pool_doc_ids = {doc_ids[index] for index in features["candidate_doc_indices"]}
        reranked_doc_ids = [doc_ids[index] for index in reranked]
        rankings_by_query[query_id] = reranked_doc_ids
        query_results.append(
            {
                "query_id": query_id,
                "candidate_pool_size": len(features["candidate_doc_indices"]),
                "selected_candidate_count": len(selected),
                "reranked_doc_ids": reranked_doc_ids,
                "comparison_to_exact": compare_exact(
                    exact_rankings[query_id],
                    reranked_doc_ids,
                    selected_doc_ids,
                    pool_doc_ids,
                ),
            }
        )

    reranked_comparisons = sum(result["selected_candidate_count"] for result in query_results)
    return {
        "policy": policy,
        "top_l": top_l,
        "top_c": top_c,
        "candidate_generation_seconds": candidate_generation_seconds,
        "total_rerank_seconds": total_rerank_seconds,
        "full_exact_comparisons": full_exact_comparisons,
        "reranked_comparisons": reranked_comparisons,
        "reranked_comparison_ratio": reranked_comparisons / full_exact_comparisons,
        "mean_candidate_pool_size": float(np.mean([q["candidate_pool_size"] for q in query_results])),
        "mean_selected_candidate_count": reranked_comparisons / len(query_results),
        "mean_exact_recall_at_k": {
            str(k): float(np.mean([
                q["comparison_to_exact"]["exact_recall_at_k"][str(k)]
                for q in query_results
            ]))
            for k in REPORT_K_VALUES
        },
        "exact_top1_match_count": sum(q["comparison_to_exact"]["exact_top1_match"] for q in query_results),
        "exact_top3_all_recovered_count": sum(q["comparison_to_exact"]["exact_top3_all_recovered"] for q in query_results),
        "exact_top5_all_recovered_count": sum(q["comparison_to_exact"]["exact_top5_all_recovered"] for q in query_results),
        "exact_top10_all_recovered_count": sum(q["comparison_to_exact"]["exact_top10_all_recovered"] for q in query_results),
        "pool_top5_all_count": sum(q["comparison_to_exact"]["pool_contains_top5"] for q in query_results),
        "pool_top10_all_count": sum(q["comparison_to_exact"]["pool_contains_top10"] for q in query_results),
        "selector_loss": {
            "5": {
                "pool_coverage_failure": sum(q["comparison_to_exact"]["selector_loss_at_5"] == "pool_coverage_failure" for q in query_results),
                "selector_failure": sum(q["comparison_to_exact"]["selector_loss_at_5"] == "selector_failure" for q in query_results),
                "none": sum(q["comparison_to_exact"]["selector_loss_at_5"] == "none" for q in query_results),
            },
            "10": {
                "pool_coverage_failure": sum(q["comparison_to_exact"]["selector_loss_at_10"] == "pool_coverage_failure" for q in query_results),
                "selector_failure": sum(q["comparison_to_exact"]["selector_loss_at_10"] == "selector_failure" for q in query_results),
                "none": sum(q["comparison_to_exact"]["selector_loss_at_10"] == "none" for q in query_results),
            },
        },
        "qrels_metrics": compute_qrels_metrics(rankings_by_query, qrels, k_values=(1, 3, 5, 10), mrr_k=10),
        "queries": query_results,
    }


def main() -> None:
    IndexPDXBONDFlat, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")
    if not EXACT_PATH.exists():
        raise SystemExit("Run 21_scifact_colbert_maxsim_numpy.py before this script.")

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
    flat_tokens, token_to_doc_index, token_to_token_position = flatten_document_embeddings(normalized_documents)
    doc_ids = document_pack["ids"]
    exact_rankings = {result["query_id"]: result["ranking_doc_ids"] for result in exact["query_results"]}

    np.random.seed(0)
    index = IndexPDXBONDFlat(ndim=flat_tokens.shape[1])
    start = perf_counter()
    index.add_load(flat_tokens)
    index_build_seconds = perf_counter() - start

    full_exact_comparisons = len(normalized_queries) * len(normalized_documents)
    runs = []
    for top_l in L_VALUES:
        generated = []
        total_generation_seconds = 0.0
        total_retrieved_tokens = 0
        for query_index, query_matrix in enumerate(normalized_queries):
            query_id = query_pack["ids"][query_index]
            query_token_hits, generation_seconds = retrieve_query_token_hits(index, query_matrix, top_l)
            features = build_candidate_features(query_token_hits, token_to_doc_index, len(doc_ids))
            generated.append(
                {
                    "query_id": query_id,
                    "query_index": query_index,
                    "features": features,
                }
            )
            total_generation_seconds += generation_seconds
            total_retrieved_tokens += features["retrieved_token_vectors"]

        policy_results = []
        for policy in POLICIES:
            for top_c in C_VALUES:
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
                        candidate_generation_seconds=total_generation_seconds,
                    )
                )
        oracle = summarize_policy(
            policy="approx_score",
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
        oracle["policy"] = "oracle_pool_upper_bound"
        runs.append(
            {
                "top_l": top_l,
                "candidate_generation_seconds": total_generation_seconds,
                "total_retrieved_token_vectors": total_retrieved_tokens,
                "mean_candidate_pool_size": oracle["mean_candidate_pool_size"],
                "oracle_pool_upper_bound": oracle,
                "policy_results": policy_results,
            }
        )

    all_policy_results = [result for run in runs for result in run["policy_results"]]
    best_c20_recall5 = max(
        (result for result in all_policy_results if result["top_c"] == 20),
        key=lambda result: (result["mean_exact_recall_at_k"]["5"], result["exact_top10_all_recovered_count"]),
    )
    best_c50_recall10 = max(
        (result for result in all_policy_results if result["top_c"] == 50),
        key=lambda result: (result["mean_exact_recall_at_k"]["10"], result["exact_top10_all_recovered_count"]),
    )

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "used": "l2sq_on_l2_normalized_vectors_for_candidate_generation",
            "rerank_metric": "exact_normalized_inner_product_maxsim",
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
            "total_query_token_vectors": int(sum(len(query) for query in normalized_queries)),
        },
        "token_index": {
            "flat_values_shape": list(flat_tokens.shape),
            "index_build_seconds": index_build_seconds,
            "token_to_doc_index": token_to_doc_index.tolist(),
            "token_to_token_position": token_to_token_position.tolist(),
        },
        "l_values": L_VALUES,
        "c_values": C_VALUES,
        "policies": POLICIES,
        "best_results": {
            "c20_recall5": {k: v for k, v in best_c20_recall5.items() if k != "queries"},
            "c50_recall10": {k: v for k, v in best_c50_recall10.items() if k != "queries"},
        },
        "runs": runs,
        "note": "Comparison ratios count exact MaxSim rerank calls only, not end-to-end speedup.",
    }
    save_json(OUTPUT_PATH, output)

    print("SciFact PDX candidate generation + exact rerank")
    print(f"documents: {len(doc_ids)}")
    print(f"queries: {len(normalized_queries)}")
    print(f"flat token matrix: {flat_tokens.shape}")
    print(f"index build time: {index_build_seconds:.6f}s")
    print(f"full exact document comparisons: {full_exact_comparisons}")
    for run in runs:
        oracle = run["oracle_pool_upper_bound"]
        print(
            f"L={run['top_l']}: gen={run['candidate_generation_seconds']:.6f}s "
            f"pool={run['mean_candidate_pool_size']:.2f} "
            f"oracle top5/top10={oracle['exact_top5_all_recovered_count']}/"
            f"{oracle['exact_top10_all_recovered_count']}"
        )
    print("\nBest C=20 by exact recall@5")
    print(
        f"L={best_c20_recall5['top_l']} policy={best_c20_recall5['policy']} "
        f"recall@5={best_c20_recall5['mean_exact_recall_at_k']['5']:.4f} "
        f"ratio={best_c20_recall5['reranked_comparison_ratio']:.4f}"
    )
    print("\nBest C=50 by exact recall@10")
    print(
        f"L={best_c50_recall10['top_l']} policy={best_c50_recall10['policy']} "
        f"recall@10={best_c50_recall10['mean_exact_recall_at_k']['10']:.4f} "
        f"ratio={best_c50_recall10['reranked_comparison_ratio']:.4f}"
    )
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
