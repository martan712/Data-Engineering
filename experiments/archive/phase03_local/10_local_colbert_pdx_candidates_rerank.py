"""PDX candidate generation followed by exact ColBERT MaxSim reranking.

This experiment treats PDX-BOND as token-level candidate generation. It then
computes exact normalized MaxSim only for the selected candidate documents.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from time import perf_counter

import numpy as np

from utils_colbert import (
    aggregate_token_retrieval_scores,
    compute_qrels_metrics,
    l2_normalize,
    load_packed_embeddings,
    maxsim_scores_for_candidate_docs,
    rank_documents,
    save_json,
    topk_overlap,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "local_corpus"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "local_corpus_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
EXACT_PATH = RESULTS_DIR / "local_colbert_maxsim_numpy.json"
OUTPUT_PATH = RESULTS_DIR / "local_colbert_pdx_candidates_rerank.json"
L_VALUES = [5, 10, 20, 50, 100]
C_VALUES = [5, 10, 20, 50]
REPORT_K_VALUES = [1, 3, 5]


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


def flatten_documents(
    documents: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_vectors = []
    token_to_doc_index = []
    token_to_token_position = []

    for doc_index, document_matrix in enumerate(documents):
        flat_vectors.append(document_matrix)
        token_count = len(document_matrix)
        token_to_doc_index.extend([doc_index] * token_count)
        token_to_token_position.extend(range(token_count))

    return (
        np.ascontiguousarray(np.vstack(flat_vectors), dtype=np.float32),
        np.array(token_to_doc_index, dtype=np.int64),
        np.array(token_to_token_position, dtype=np.int64),
    )


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


def select_top_candidates(scores: np.ndarray, candidate_doc_indices: set[int], top_c: int) -> list[int]:
    """Select top-C candidate docs by approximate aggregate score."""
    if not candidate_doc_indices:
        return []
    candidates = np.array(sorted(candidate_doc_indices), dtype=np.int64)
    candidate_scores = scores[candidates]
    order = np.lexsort((candidates, -candidate_scores))
    return candidates[order[:top_c]].astype(int).tolist()


def rerank_candidates(
    query_matrix: np.ndarray,
    documents: list[np.ndarray],
    candidate_doc_indices: list[int],
) -> tuple[list[int], dict[int, float], float]:
    """Run exact MaxSim only over selected candidates."""
    start = perf_counter()
    candidate_scores = maxsim_scores_for_candidate_docs(
        query_matrix,
        documents,
        candidate_doc_indices,
        normalize=False,
    )
    elapsed = perf_counter() - start
    if not candidate_scores:
        return [], {}, elapsed

    ordered_candidates = list(candidate_scores)
    scores = np.array([candidate_scores[index] for index in ordered_candidates], dtype=np.float32)
    local_ranking = rank_documents(scores)
    ranking = [ordered_candidates[index] for index in local_ranking]
    return ranking, candidate_scores, elapsed


def compare_to_exact(reference_ranking: list[str], reranked: list[str], candidates: list[str]) -> dict:
    return {
        "exact_top1_in_candidates": bool(reference_ranking[:1] and reference_ranking[0] in candidates),
        "exact_top3_all_in_candidates": set(reference_ranking[:3]).issubset(set(candidates)),
        "exact_recall_at_k": {
            str(k): topk_overlap(reference_ranking, reranked, k)["ratio"]
            for k in REPORT_K_VALUES
        },
        "top10_overlap": topk_overlap(reference_ranking, reranked, 10),
    }


def main() -> None:
    IndexPDXBONDFlat, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")
    if not EXACT_PATH.exists():
        raise SystemExit("Run 08_local_colbert_maxsim_numpy.py before this script.")

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
    flat_tokens, token_to_doc_index, token_to_token_position = flatten_documents(
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

    runs = []
    for requested_l in L_VALUES:
        if requested_l > len(flat_tokens):
            continue

        generated = []
        total_retrieved_tokens = 0
        total_generation_time = 0.0
        for query_index, query_matrix in enumerate(normalized_queries):
            query_id = query_pack["ids"][query_index]
            query_token_hits, retrieval_time = retrieve_query_token_hits(
                index,
                query_matrix,
                requested_l,
            )
            scores, candidate_doc_indices, retrieved_tokens = aggregate_token_retrieval_scores(
                query_token_hits,
                token_to_doc_index,
                num_documents=len(doc_ids),
                missing_value=0.0,
            )
            generated.append(
                {
                    "query_id": query_id,
                    "query_index": query_index,
                    "approx_scores": scores,
                    "candidate_doc_indices": candidate_doc_indices,
                    "retrieved_token_vectors": retrieved_tokens,
                    "generation_seconds": retrieval_time,
                }
            )
            total_retrieved_tokens += retrieved_tokens
            total_generation_time += retrieval_time

        c_runs = []
        for requested_c in C_VALUES:
            query_runs = []
            rankings_by_query = {}
            total_rerank_time = 0.0

            for generated_query in generated:
                query_id = generated_query["query_id"]
                query_index = generated_query["query_index"]
                selected_indices = select_top_candidates(
                    generated_query["approx_scores"],
                    generated_query["candidate_doc_indices"],
                    min(requested_c, len(generated_query["candidate_doc_indices"])),
                )
                reranked_indices, candidate_scores, rerank_time = rerank_candidates(
                    normalized_queries[query_index],
                    normalized_documents,
                    selected_indices,
                )
                total_rerank_time += rerank_time

                selected_doc_ids = [doc_ids[index] for index in selected_indices]
                reranked_doc_ids = [doc_ids[index] for index in reranked_indices]
                rankings_by_query[query_id] = reranked_doc_ids
                query_runs.append(
                    {
                        "query_id": query_id,
                        "candidate_count": len(selected_indices),
                        "candidate_doc_ids": selected_doc_ids,
                        "reranked_doc_ids": reranked_doc_ids,
                        "candidate_scores": {
                            doc_ids[index]: float(score)
                            for index, score in candidate_scores.items()
                        },
                        "rerank_seconds": rerank_time,
                        "comparison_to_exact": compare_to_exact(
                            exact_rankings[query_id],
                            reranked_doc_ids,
                            selected_doc_ids,
                        ),
                    }
                )

            qrels_metrics = compute_qrels_metrics(rankings_by_query, qrels)
            c_runs.append(
                {
                    "top_c": requested_c,
                    "total_rerank_seconds": total_rerank_time,
                    "qrels_metrics": qrels_metrics,
                    "mean_exact_recall_at_k": {
                        str(k): float(np.mean([
                            query_run["comparison_to_exact"]["exact_recall_at_k"][str(k)]
                            for query_run in query_runs
                        ]))
                        for k in REPORT_K_VALUES
                    },
                    "exact_top1_in_candidates": sum(
                        1
                        for query_run in query_runs
                        if query_run["comparison_to_exact"]["exact_top1_in_candidates"]
                    ),
                    "exact_top3_all_in_candidates": sum(
                        1
                        for query_run in query_runs
                        if query_run["comparison_to_exact"]["exact_top3_all_in_candidates"]
                    ),
                    "mean_top10_overlap_ratio": float(np.mean([
                        query_run["comparison_to_exact"]["top10_overlap"]["ratio"]
                        for query_run in query_runs
                    ])),
                    "queries": query_runs,
                }
            )

        runs.append(
            {
                "top_l": requested_l,
                "total_retrieved_token_vectors": total_retrieved_tokens,
                "mean_candidate_pool_documents": float(np.mean([
                    len(generated_query["candidate_doc_indices"])
                    for generated_query in generated
                ])),
                "candidate_generation_seconds": total_generation_time,
                "candidate_limits": c_runs,
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
            "count": len(queries),
            "values_shape": list(query_pack["values"].shape),
            "offsets": query_pack["offsets"].tolist(),
        },
        "token_index": {
            "flat_values_shape": list(flat_tokens.shape),
            "token_to_doc_index": token_to_doc_index.tolist(),
            "token_to_token_position": token_to_token_position.tolist(),
            "index_build_seconds": index_build_time,
        },
        "runs": runs,
    }
    save_json(OUTPUT_PATH, output)

    print("Local PDX candidate generation + exact MaxSim rerank")
    print(f"PDX supported metrics: {supported_metrics}")
    print(f"documents: {len(doc_ids)}")
    print(f"flat token matrix: {flat_tokens.shape}")
    print(f"index build time: {index_build_time:.6f}s")
    for run in runs:
        print(f"\nL={run['top_l']}")
        print(f"  total token vectors retrieved: {run['total_retrieved_token_vectors']}")
        print(
            "  mean candidate pool documents: "
            f"{run['mean_candidate_pool_documents']:.2f}"
        )
        for c_run in run["candidate_limits"]:
            print(
                f"  C={c_run['top_c']}: "
                f"top1 in candidates {c_run['exact_top1_in_candidates']}/{len(queries)}, "
                f"top3 all in candidates {c_run['exact_top3_all_in_candidates']}/{len(queries)}, "
                f"mean exact recall@k {c_run['mean_exact_recall_at_k']}, "
                f"qrels {c_run['qrels_metrics']['macro']}"
            )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
