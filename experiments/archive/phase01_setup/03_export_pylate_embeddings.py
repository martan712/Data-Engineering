"""Export toy PyLate / ColBERT multi-vector embeddings for later inspection.

This script does not feed ColBERT embeddings into PDX. It only saves the
variable-length token-vector matrices so the next iteration can inspect layout
requirements deliberately.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from pylate import models


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "pylate_embeddings"
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
    start = perf_counter()
    result = func()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return result


def as_list_of_arrays(embeddings) -> list[np.ndarray]:
    if isinstance(embeddings, np.ndarray):
        if embeddings.dtype == object:
            return [np.asarray(item, dtype=np.float32) for item in embeddings.tolist()]
        if embeddings.ndim == 3:
            return [np.asarray(embeddings[i], dtype=np.float32) for i in range(embeddings.shape[0])]
        if embeddings.ndim == 2:
            return [np.asarray(embeddings, dtype=np.float32)]

    return [np.asarray(item, dtype=np.float32) for item in embeddings]


def describe(name: str, embeddings) -> list[np.ndarray]:
    arrays = as_list_of_arrays(embeddings)
    shapes = [array.shape for array in arrays]
    token_lengths = [shape[0] for shape in shapes]
    vector_dims = sorted({shape[1] for shape in shapes if len(shape) == 2})
    variable_length = len(set(token_lengths)) > 1

    print(f"\n{name} embeddings")
    print(f"raw type: {type(embeddings).__name__}")
    if isinstance(embeddings, np.ndarray):
        print(f"raw ndarray shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"items: {len(arrays)}")
    print(f"item shapes: {shapes}")
    print(f"token lengths: {token_lengths}")
    print(f"vector dimensions: {vector_dims}")
    print(f"variable-length token matrices: {variable_length}")

    return arrays


def save_packed(path: Path, ids: list[str], texts: list[str], arrays: list[np.ndarray]) -> None:
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for idx, array in enumerate(arrays):
        offsets[idx + 1] = offsets[idx] + array.shape[0]

    values = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    np.savez_compressed(
        path,
        ids=np.array(ids),
        texts=np.array(texts),
        values=values,
        offsets=offsets,
        shapes=np.array([array.shape for array in arrays], dtype=np.int64),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PyLate model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    model = timed_step(
        "Load model",
        lambda: models.ColBERT(model_name_or_path=MODEL_NAME, device=DEVICE),
    )

    document_ids = list(DOCUMENTS.keys())
    document_texts = list(DOCUMENTS.values())

    document_embeddings = timed_step(
        "Encode documents",
        lambda: model.encode(
            document_texts,
            is_query=False,
            batch_size=1,
            show_progress_bar=False,
        ),
    )
    query_embeddings = timed_step(
        "Encode queries",
        lambda: model.encode(
            QUERIES,
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
        ),
    )

    document_arrays = describe("Document", document_embeddings)
    query_arrays = describe("Query", query_embeddings)

    documents_path = OUTPUT_DIR / "toy_document_embeddings.npz"
    queries_path = OUTPUT_DIR / "toy_query_embeddings.npz"
    metadata_path = OUTPUT_DIR / "metadata.json"

    save_packed(documents_path, document_ids, document_texts, document_arrays)
    save_packed(queries_path, [f"query-{i + 1}" for i in range(len(QUERIES))], QUERIES, query_arrays)

    metadata = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "document_count": len(document_arrays),
        "query_count": len(query_arrays),
        "document_shapes": [list(array.shape) for array in document_arrays],
        "query_shapes": [list(array.shape) for array in query_arrays],
        "format": "Packed token vectors. Reconstruct item i with values[offsets[i]:offsets[i+1]].",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved documents: {documents_path}")
    print(f"Saved queries: {queries_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
