"""PDX token-vector retrieval on the local ColBERT corpus.

This is an approximate document scorer. PDX retrieves the top-L document token
vectors for each query token. The script maps token ids back to document ids,
keeps the best retrieved similarity per query-token/document pair, fills
missing pairs with 0.0, and sums across query tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from utils_colbert import (
    aggregate_token_retrieval_scores,
    compute_qrels_metrics,
    l2_normalize,
    load_packed_embeddings,
    rank_documents,
    save_json,
    topk_overlap,
    topk_pairwise_order_agreement,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "local_corpus"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "local_corpus_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
EXACT_PATH = RESULTS_DIR / "local_colbert_maxsim_numpy.json"
OUTPUT_PATH = RESULTS_DIR / "local_colbert_token_pdx_retrieval.json"
L_VALUES = [5, 10, 20, 50, 100, 200]
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
    """Flatten document token matrices and keep token metadata."""
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


def compare_to_exact(reference_ranking: list[str], candidate_ranking: list[str]) -> dict:
    """Compare approximate document ranking to exact MaxSim ranking."""
    return {
        "exact_top1_match": reference_ranking[:1] == candidate_ranking[:1],
        "exact_recall_at_k": {
            str(k): topk_overlap(reference_ranking, candidate_ranking, k)["ratio"]
            for k in REPORT_K_VALUES
        },
        "top10_overlap": topk_overlap(reference_ranking, candidate_ranking, 10),
        "top10_pairwise_order_agreement": topk_pairwise_order_agreement(
            reference_ranking,
            candidate_ranking,
            k=10,
        ),
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
        query_runs = []
        rankings_by_query = {}
        total_retrieved_tokens = 0
        total_unique_candidates = 0
        total_query_time = 0.0

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
            ranking = rank_documents(scores)
            ranking_doc_ids = [doc_ids[doc_index] for doc_index in ranking]
            rankings_by_query[query_id] = ranking_doc_ids

            total_retrieved_tokens += retrieved_tokens
            total_unique_candidates += len(candidate_doc_indices)
            total_query_time += retrieval_time
            query_runs.append(
                {
                    "query_id": query_id,
                    "query_text": query_pack["texts"][query_index],
                    "retrieved_token_vectors": retrieved_tokens,
                    "unique_candidate_documents": len(candidate_doc_indices),
                    "elapsed_seconds": retrieval_time,
                    "scores": scores.tolist(),
                    "ranking": ranking,
                    "ranking_doc_ids": ranking_doc_ids,
                    "top10": ranking_doc_ids[:10],
                    "comparison_to_exact": compare_to_exact(
                        exact_rankings[query_id],
                        ranking_doc_ids,
                    ),
                }
            )

        qrels_metrics = compute_qrels_metrics(rankings_by_query, qrels)
        runs.append(
            {
                "top_l": requested_l,
                "total_retrieved_token_vectors": total_retrieved_tokens,
                "mean_unique_candidate_documents": total_unique_candidates / len(queries),
                "elapsed_seconds": total_query_time,
                "qrels_metrics": qrels_metrics,
                "mean_exact_recall_at_k": {
                    str(k): float(np.mean([
                        query_run["comparison_to_exact"]["exact_recall_at_k"][str(k)]
                        for query_run in query_runs
                    ]))
                    for k in REPORT_K_VALUES
                },
                "mean_top10_overlap_ratio": float(np.mean([
                    query_run["comparison_to_exact"]["top10_overlap"]["ratio"]
                    for query_run in query_runs
                ])),
                "mean_top10_pairwise_order_agreement": float(np.mean([
                    query_run["comparison_to_exact"]["top10_pairwise_order_agreement"]
                    for query_run in query_runs
                ])),
                "top1_matches": sum(
                    1
                    for query_run in query_runs
                    if query_run["comparison_to_exact"]["exact_top1_match"]
                ),
                "queries": query_runs,
            }
        )

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "used": "l2sq_on_l2_normalized_vectors",
            "similarity_interpretation": "cosine_similarity = 1 - squared_l2 / 2",
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
            "token_to_doc_id": [doc_ids[int(doc_index)] for doc_index in token_to_doc_index],
            "token_to_doc_index": token_to_doc_index.tolist(),
            "token_to_token_position": token_to_token_position.tolist(),
            "index_build_seconds": index_build_time,
        },
        "runs": runs,
        "limitation": (
            "Missing query-token/document contributions are filled with 0.0. "
            "The ranking is approximate unless top-L retrieves the best token "
            "for each important document and query token."
        ),
    }
    save_json(OUTPUT_PATH, output)

    print("Local ColBERT token-vector PDX retrieval")
    print(f"PDX supported metrics: {supported_metrics}")
    print(f"documents: {len(doc_ids)}")
    print(f"flat token matrix: {flat_tokens.shape}")
    print(f"index build time: {index_build_time:.6f}s")
    for run in runs:
        print(f"\nL={run['top_l']}")
        print(f"  total token vectors retrieved: {run['total_retrieved_token_vectors']}")
        print(
            "  mean unique candidate documents: "
            f"{run['mean_unique_candidate_documents']:.2f}"
        )
        print(f"  mean exact recall@k: {run['mean_exact_recall_at_k']}")
        print(f"  exact top1 matches: {run['top1_matches']}/{len(queries)}")
        print(f"  qrels macro: {run['qrels_metrics']['macro']}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
