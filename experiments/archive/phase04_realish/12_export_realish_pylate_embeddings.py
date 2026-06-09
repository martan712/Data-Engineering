"""Export PyLate / ColBERT embeddings for the Iteration 5 realish corpus."""

from __future__ import annotations

import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from time import perf_counter

import numpy as np
import torch
from pylate import models

from utils_colbert import (
    as_list_of_arrays,
    load_jsonl,
    save_json,
    save_packed_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus_embeddings"
MODEL_NAME = os.environ.get("PYLATE_MODEL", "lightonai/GTE-ModernColBERT-v1")
DEVICE = os.environ.get("PYLATE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("PYLATE_BATCH_SIZE", "8"))


def timed_step(label: str, func):
    start = perf_counter()
    result = func()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return result, elapsed


def describe(label: str, arrays: list[np.ndarray]) -> dict:
    token_counts = [array.shape[0] for array in arrays]
    dimensions = sorted({array.shape[1] for array in arrays})
    return {
        "label": label,
        "item_count": len(arrays),
        "total_token_vectors": int(sum(token_counts)),
        "min_token_vectors": int(min(token_counts)),
        "max_token_vectors": int(max(token_counts)),
        "average_token_vectors": float(np.mean(token_counts)),
        "embedding_dimensions": dimensions,
    }


def main() -> None:
    documents = load_jsonl(CORPUS_DIR / "documents.jsonl")
    queries = load_jsonl(CORPUS_DIR / "queries.jsonl")
    if not documents or not queries:
        raise SystemExit("Realish corpus is empty. Run 11_create_small_realish_corpus.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document_ids = [document["id"] for document in documents]
    document_texts = [document["text"] for document in documents]
    query_ids = [query["id"] for query in queries]
    query_texts = [query["text"] for query in queries]

    print(f"PyLate model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")

    model, load_time = timed_step(
        "Load model",
        lambda: models.ColBERT(model_name_or_path=MODEL_NAME, device=DEVICE),
    )

    document_embeddings, document_encode_time = timed_step(
        "Encode documents",
        lambda: model.encode(
            document_texts,
            is_query=False,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        ),
    )
    query_embeddings, query_encode_time = timed_step(
        "Encode queries",
        lambda: model.encode(
            query_texts,
            is_query=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        ),
    )

    document_arrays = as_list_of_arrays(document_embeddings)
    query_arrays = as_list_of_arrays(query_embeddings)
    document_summary = describe("documents", document_arrays)
    query_summary = describe("queries", query_arrays)

    save_packed_embeddings(
        OUTPUT_DIR / "documents_packed.npz",
        document_ids,
        document_texts,
        document_arrays,
    )
    save_packed_embeddings(
        OUTPUT_DIR / "queries_packed.npz",
        query_ids,
        query_texts,
        query_arrays,
    )
    save_json(OUTPUT_DIR / "document_ids.json", document_ids)
    save_json(OUTPUT_DIR / "query_ids.json", query_ids)
    save_json(
        OUTPUT_DIR / "metadata.json",
        {
            "model": MODEL_NAME,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "load_seconds": load_time,
            "document_encode_seconds": document_encode_time,
            "query_encode_seconds": query_encode_time,
            "documents": document_summary,
            "queries": query_summary,
            "format": "Packed token vectors. Reconstruct item i with values[offsets[i]:offsets[i+1]].",
        },
    )

    embedding_dim = document_summary["embedding_dimensions"][0]
    print("\nExport summary")
    print(f"number of documents: {len(documents)}")
    print(f"number of queries: {len(queries)}")
    print(f"total document token vectors: {document_summary['total_token_vectors']}")
    print(f"total query token vectors: {query_summary['total_token_vectors']}")
    print(
        "average document token vectors per document: "
        f"{document_summary['average_token_vectors']:.2f}"
    )
    print(
        "average query token vectors per query: "
        f"{query_summary['average_token_vectors']:.2f}"
    )
    print(f"embedding dimension: {embedding_dim}")
    print(f"saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
