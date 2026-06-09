"""Exact NumPy ColBERT MaxSim reference on the toy PyLate embeddings.

This script intentionally does not use PDX. It is the correctness reference for
later experiments that approximate or accelerate ColBERT-style late interaction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from time import perf_counter

import numpy as np

from utils_colbert import (
    load_packed_embeddings,
    maxsim_score,
    rank_documents,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "pylate_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
OUTPUT_PATH = RESULTS_DIR / "colbert_maxsim_numpy.json"


def score_all_queries(
    queries: list[np.ndarray],
    documents: list[np.ndarray],
    normalize: bool,
) -> tuple[list[dict], float]:
    """Compute exact MaxSim scores/rankings for all query-document pairs."""
    start = perf_counter()
    query_results = []

    for query_index, query_matrix in enumerate(queries):
        scores = np.array(
            [
                maxsim_score(
                    query_matrix=query_matrix,
                    document_matrix=document_matrix,
                    normalize=normalize,
                )
                for document_matrix in documents
            ],
            dtype=np.float32,
        )
        ranking = rank_documents(scores)
        query_results.append(
            {
                "query_index": query_index,
                "scores": scores.tolist(),
                "ranking": ranking,
            }
        )

    return query_results, perf_counter() - start


def print_rankings(
    label: str,
    query_results: list[dict],
    query_ids: list[str],
    query_texts: list[str],
    document_ids: list[str],
) -> None:
    print(f"\n{label}")
    for result in query_results:
        query_index = result["query_index"]
        ranking = result["ranking"]
        scores = result["scores"]
        print(f"\nQuery {query_ids[query_index]}: {query_texts[query_index]}")
        for rank, document_index in enumerate(ranking, start=1):
            print(
                f"  {rank}. {document_ids[document_index]} "
                f"score={scores[document_index]:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Print normalized inner-product rankings as the primary view.",
    )
    args = parser.parse_args()

    document_pack = load_packed_embeddings(EMBEDDINGS_DIR / "toy_document_embeddings.npz")
    query_pack = load_packed_embeddings(EMBEDDINGS_DIR / "toy_query_embeddings.npz")

    documents = unpack_embeddings(document_pack["values"], document_pack["offsets"])
    queries = unpack_embeddings(query_pack["values"], query_pack["offsets"])

    print("Loaded packed PyLate embeddings")
    print(f"document values shape: {document_pack['values'].shape}")
    print(f"document offsets: {document_pack['offsets'].tolist()}")
    print(f"query values shape: {query_pack['values'].shape}")
    print(f"query offsets: {query_pack['offsets'].tolist()}")

    raw_results, raw_time = score_all_queries(
        queries=queries,
        documents=documents,
        normalize=False,
    )
    normalized_results, normalized_time = score_all_queries(
        queries=queries,
        documents=documents,
        normalize=True,
    )

    primary_label = (
        "Exact MaxSim, L2-normalized inner product"
        if args.normalize
        else "Exact MaxSim, raw inner product"
    )
    primary_results = normalized_results if args.normalize else raw_results
    print_rankings(
        primary_label,
        primary_results,
        query_pack["ids"],
        query_pack["texts"],
        document_pack["ids"],
    )

    print("\nTimings")
    print(f"raw inner product MaxSim: {raw_time:.6f}s")
    print(f"normalized inner product MaxSim: {normalized_time:.6f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "documents": {
            "ids": document_pack["ids"],
            "texts": document_pack["texts"],
            "values_shape": list(document_pack["values"].shape),
            "offsets": document_pack["offsets"].tolist(),
            "item_shapes": document_pack["shapes"].tolist(),
        },
        "queries": {
            "ids": query_pack["ids"],
            "texts": query_pack["texts"],
            "values_shape": list(query_pack["values"].shape),
            "offsets": query_pack["offsets"].tolist(),
            "item_shapes": query_pack["shapes"].tolist(),
        },
        "results": {
            "inner_product": {
                "normalize": False,
                "similarity": "inner_product",
                "elapsed_seconds": raw_time,
                "queries": raw_results,
            },
            "normalized_inner_product": {
                "normalize": True,
                "similarity": "inner_product_after_l2_normalization",
                "elapsed_seconds": normalized_time,
                "queries": normalized_results,
            },
        },
        "primary_result": (
            "normalized_inner_product" if args.normalize else "inner_product"
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
