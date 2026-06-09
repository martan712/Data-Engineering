"""Export PyLate / ColBERT embeddings for any prepared BEIR benchmark.

Generalizes experiment 20 with configurable corpus and output directories so it
can encode arbitrary corpora produced by 29_prepare_beir_benchmark.py.

Example
-------
    python 30_export_beir_pylate_embeddings.py \
        --corpus-dir artifacts/scifact_full_benchmark \
        --output-dir artifacts/scifact_full_benchmark_embeddings
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from pylate import models

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PROJECT_ROOT, setup_imports

setup_imports()
from utils_colbert import (
    as_list_of_arrays,
    load_jsonl,
    save_json,
    save_packed_embeddings,
)
MODEL_NAME = os.environ.get("PYLATE_MODEL", "lightonai/GTE-ModernColBERT-v1")
DEVICE = os.environ.get("PYLATE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH_SIZE = int(os.environ.get("PYLATE_BATCH_SIZE", "8"))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = args.corpus_dir if args.corpus_dir.is_absolute() else PROJECT_ROOT / args.corpus_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    documents = load_jsonl(corpus_dir / "documents.jsonl")
    queries = load_jsonl(corpus_dir / "queries.jsonl")
    if not documents or not queries:
        raise SystemExit(f"Benchmark at {corpus_dir} is empty. Run 29_prepare_beir_benchmark.py first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    document_ids = [document["id"] for document in documents]
    document_texts = [document["text"] for document in documents]
    query_ids = [query["id"] for query in queries]
    query_texts = [query["text"] for query in queries]

    print(f"PyLate model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Corpus dir: {corpus_dir}")
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
            batch_size=args.batch_size,
            show_progress_bar=True,
        ),
    )
    query_embeddings, query_encode_time = timed_step(
        "Encode queries",
        lambda: model.encode(
            query_texts,
            is_query=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        ),
    )

    document_arrays = as_list_of_arrays(document_embeddings)
    query_arrays = as_list_of_arrays(query_embeddings)
    document_summary = describe("documents", document_arrays)
    query_summary = describe("queries", query_arrays)

    save_packed_embeddings(output_dir / "documents_packed.npz", document_ids, document_texts, document_arrays)
    save_packed_embeddings(output_dir / "queries_packed.npz", query_ids, query_texts, query_arrays)
    save_json(output_dir / "document_ids.json", document_ids)
    save_json(output_dir / "query_ids.json", query_ids)
    save_json(
        output_dir / "metadata.json",
        {
            "model": MODEL_NAME,
            "device": DEVICE,
            "batch_size": args.batch_size,
            "corpus_dir": str(corpus_dir),
            "load_seconds": load_time,
            "document_encode_seconds": document_encode_time,
            "query_encode_seconds": query_encode_time,
            "documents": document_summary,
            "queries": query_summary,
            "format": "Packed token vectors. Reconstruct item i with values[offsets[i]:offsets[i+1]].",
        },
    )

    print("\nExport summary")
    print(f"documents: {len(documents)}  queries: {len(queries)}")
    print(f"total document token vectors: {document_summary['total_token_vectors']}")
    print(f"total query token vectors: {query_summary['total_token_vectors']}")
    print(f"embedding dimension: {document_summary['embedding_dimensions'][0]}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
