"""Minimal PyLate / ColBERT baseline for a tiny toy collection.

This script intentionally keeps the collection tiny. The goal is to verify that
PyLate can load a ColBERT model, encode documents and queries, build a PLAID
index, and return scored retrieval results before doing larger experiments.
"""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import torch
from pylate import indexes, models, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDEX_FOLDER = PROJECT_ROOT / "artifacts" / "pylate_baseline"

# The professor suggested PyLate; its current examples recommend this trained
# ColBERT model. Override with PYLATE_MODEL for smaller smoke tests if needed.
MODEL_NAME = os.environ.get("PYLATE_MODEL", "lightonai/GTE-ModernColBERT-v1")
DEVICE = os.environ.get("PYLATE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

DOCUMENTS = {
    "doc-1": "ColBERT represents each passage with multiple token-level vectors for late interaction retrieval.",
    "doc-2": "BOND accelerates k-nearest neighbor search by scanning dimensions incrementally and pruning candidates.",
    "doc-3": "PDX stores vectors in a dimension-oriented layout to improve locality for vector search.",
    "doc-4": "ADSampling adaptively samples vector dimensions to speed up distance comparisons.",
    "doc-5": "CoRECT is useful for evaluating retrieval and embedding experiments.",
}

QUERIES = [
    "late interaction retrieval with token vectors",
    "dimension pruning for vector search",
    "adaptive dimension sampling for distance computations",
]


def timed_step(label: str, func):
    """Run a step and print its wall-clock duration."""
    start = perf_counter()
    result = func()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return result, elapsed


def main() -> None:
    print(f"PyLate model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    # Load the ColBERT model. On first run this may download model files from
    # Hugging Face, so model load time is reported separately from indexing.
    model, _ = timed_step(
        "Load model",
        lambda: models.ColBERT(model_name_or_path=MODEL_NAME, device=DEVICE),
    )

    document_ids = list(DOCUMENTS.keys())
    document_texts = list(DOCUMENTS.values())

    # Encode documents as multi-vector embeddings. is_query=False applies the
    # document-side ColBERT tokenization/prefix behavior.
    document_embeddings, document_encode_time = timed_step(
        "Encode documents",
        lambda: model.encode(
            document_texts,
            is_query=False,
            batch_size=1,
            show_progress_bar=False,
        ),
    )

    # Build a small FastPLAID-backed index. The parameters are intentionally
    # tiny for a smoke test, not tuned for retrieval quality.
    index = indexes.PLAID(
        index_folder=str(INDEX_FOLDER),
        index_name="toy_colbert",
        override=True,
        nbits=2,
        kmeans_niters=1,
        n_samples_kmeans=32,
        n_ivf_probe=2,
        n_full_scores=16,
        show_progress=False,
        device=DEVICE,
        use_triton=False,
    )

    _, plaid_index_time = timed_step(
        "Build PLAID index",
        lambda: index.add_documents(
            documents_ids=document_ids,
            documents_embeddings=document_embeddings,
        ),
    )
    print(f"Total indexing time, including document encoding: {document_encode_time + plaid_index_time:.3f}s")

    # Encode queries with query-side ColBERT tokenization/prefix behavior.
    query_embeddings, query_encode_time = timed_step(
        "Encode queries",
        lambda: model.encode(
            QUERIES,
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
        ),
    )

    retriever = retrieve.ColBERT(index=index)
    results, retrieve_time = timed_step(
        "Retrieve",
        lambda: retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=3,
        ),
    )
    print(f"Total query time, including query encoding: {query_encode_time + retrieve_time:.3f}s")

    for query, hits in zip(QUERIES, results):
        print(f"\nQuery: {query}")
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit["id"]
            score = hit["score"]
            print(f"  {rank}. {doc_id} | score={score:.4f} | {DOCUMENTS[doc_id]}")


if __name__ == "__main__":
    main()
