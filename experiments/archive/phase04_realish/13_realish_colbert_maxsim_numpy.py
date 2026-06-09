"""Exact NumPy ColBERT MaxSim reference for the realish corpus."""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from time import perf_counter

import numpy as np

from utils_colbert import (
    compute_qrels_metrics,
    l2_normalize,
    load_packed_embeddings,
    rank_documents,
    save_json,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
OUTPUT_PATH = RESULTS_DIR / "realish_colbert_maxsim_numpy.json"


def maxsim_scores(query: np.ndarray, documents: list[np.ndarray]) -> np.ndarray:
    """Score one normalized query matrix against all normalized documents."""
    scores = []
    for document in documents:
        similarities = query @ document.T
        scores.append(float(np.max(similarities, axis=1).sum()))
    return np.array(scores, dtype=np.float32)


def main() -> None:
    document_pack = load_packed_embeddings(EMBEDDINGS_DIR / "documents_packed.npz")
    query_pack = load_packed_embeddings(EMBEDDINGS_DIR / "queries_packed.npz")
    with (CORPUS_DIR / "qrels.json").open("r", encoding="utf-8") as file:
        qrels = json.load(file)

    documents = unpack_embeddings(document_pack["values"], document_pack["offsets"])
    queries = unpack_embeddings(query_pack["values"], query_pack["offsets"])
    normalized_documents = [l2_normalize(document) for document in documents]
    normalized_queries = [l2_normalize(query) for query in queries]

    start = perf_counter()
    query_results = []
    rankings_by_query = {}
    for query_index, query_matrix in enumerate(normalized_queries):
        scores = maxsim_scores(query_matrix, normalized_documents)
        ranking = rank_documents(scores)
        ranking_doc_ids = [document_pack["ids"][doc_index] for doc_index in ranking]
        query_id = query_pack["ids"][query_index]
        rankings_by_query[query_id] = ranking_doc_ids
        query_results.append(
            {
                "query_id": query_id,
                "query_text": query_pack["texts"][query_index],
                "scores": scores.tolist(),
                "ranking": ranking,
                "ranking_doc_ids": ranking_doc_ids,
                "top10": ranking_doc_ids[:10],
            }
        )
    elapsed = perf_counter() - start

    qrels_metrics = compute_qrels_metrics(
        rankings_by_query,
        qrels,
        k_values=(1, 3, 5, 10),
        mrr_k=10,
    )

    output = {
        "metric": "normalized_inner_product_maxsim",
        "elapsed_seconds": elapsed,
        "documents": {
            "count": len(documents),
            "ids": document_pack["ids"],
            "values_shape": list(document_pack["values"].shape),
            "offsets": document_pack["offsets"].tolist(),
            "item_shapes": document_pack["shapes"].tolist(),
        },
        "queries": {
            "count": len(queries),
            "ids": query_pack["ids"],
            "values_shape": list(query_pack["values"].shape),
            "offsets": query_pack["offsets"].tolist(),
            "item_shapes": query_pack["shapes"].tolist(),
        },
        "qrels_metrics": qrels_metrics,
        "query_results": query_results,
    }
    save_json(OUTPUT_PATH, output)

    print("Realish exact ColBERT MaxSim")
    print(f"documents: {len(documents)}")
    print(f"queries: {len(queries)}")
    print(f"document token vectors: {document_pack['values'].shape[0]}")
    print(f"query token vectors: {query_pack['values'].shape[0]}")
    print(f"elapsed: {elapsed:.6f}s")
    print(f"qrels macro metrics: {qrels_metrics['macro']}")
    for result in query_results[:5]:
        print(f"{result['query_id']} top10: {result['top10']}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
