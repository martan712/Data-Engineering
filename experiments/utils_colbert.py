"""Small helpers for ColBERT multi-vector experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _validate_offsets(
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    source: str,
) -> None:
    """Check the packed representation before it reaches a scoring kernel."""
    if values.ndim != 2:
        raise ValueError(f"{source}: values must be a 2D matrix")
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError(f"{source}: offsets must be a 1D terminal-offset array")
    if int(offsets[0]) != 0:
        raise ValueError(f"{source}: offsets must start at zero")

    lengths = np.diff(offsets)
    if np.any(lengths <= 0):
        raise ValueError(f"{source}: every item must contain at least one token vector")
    if int(offsets[-1]) != values.shape[0]:
        raise ValueError(
            f"{source}: final offset {int(offsets[-1])} does not match "
            f"the {values.shape[0]} stored vectors"
        )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records."""
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def save_json(path: str | Path, data: Any) -> None:
    """Write JSON with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def as_list_of_arrays(embeddings: Any) -> list[np.ndarray]:
    """Normalize model embedding outputs to a list of float32 2D arrays."""
    if isinstance(embeddings, np.ndarray):
        if embeddings.dtype == object:
            return [np.asarray(item, dtype=np.float32) for item in embeddings.tolist()]
        if embeddings.ndim == 3:
            return [
                np.asarray(embeddings[index], dtype=np.float32)
                for index in range(embeddings.shape[0])
            ]
        if embeddings.ndim == 2:
            return [np.asarray(embeddings, dtype=np.float32)]

    return [np.asarray(item, dtype=np.float32) for item in embeddings]


def save_packed_embeddings(
    path: str | Path,
    ids: list[str],
    texts: list[str],
    arrays: list[np.ndarray],
) -> None:
    """Save variable-length token-vector matrices in the project NPZ format."""
    if not arrays:
        raise ValueError("At least one embedding matrix is required")
    if len(ids) != len(texts) or len(ids) != len(arrays):
        raise ValueError("ids, texts, and embedding matrices must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("Embedding IDs must be unique")

    matrices = [np.asarray(array, dtype=np.float32) for array in arrays]
    embedding_dimension: int | None = None
    for index, matrix in enumerate(matrices):
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(
                f"Embedding matrix {index} must be a non-empty 2D token matrix"
            )
        if embedding_dimension is None:
            embedding_dimension = int(matrix.shape[1])
        elif matrix.shape[1] != embedding_dimension:
            raise ValueError(
                f"Embedding matrix {index} has dimension {matrix.shape[1]}; "
                f"expected {embedding_dimension}"
            )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(matrices) + 1, dtype=np.int64)
    for index, matrix in enumerate(matrices):
        offsets[index + 1] = offsets[index] + matrix.shape[0]

    values = np.ascontiguousarray(
        np.concatenate(matrices, axis=0),
        dtype=np.float32,
    )
    np.savez_compressed(
        output_path,
        ids=np.array(ids),
        texts=np.array(texts),
        values=values,
        offsets=offsets,
        shapes=np.array([matrix.shape for matrix in matrices], dtype=np.int64),
    )


def load_packed_embeddings(path: str | Path) -> dict[str, Any]:
    """Load packed variable-length embeddings from an `.npz` file.

    The export format stores all token vectors in one `values` matrix and uses
    `offsets` to mark the slice for each document or query.
    """
    input_path = Path(path)
    required_fields = {"ids", "texts", "values", "offsets", "shapes"}
    with np.load(input_path, allow_pickle=False) as data:
        missing_fields = required_fields.difference(data.files)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"{input_path}: missing packed fields: {missing}")

        ids = data["ids"].astype(str).tolist()
        texts = data["texts"].astype(str).tolist()
        values = np.ascontiguousarray(data["values"], dtype=np.float32)
        offsets = np.ascontiguousarray(data["offsets"], dtype=np.int64)
        shapes = np.ascontiguousarray(data["shapes"], dtype=np.int64)

    source = str(input_path)
    _validate_offsets(values, offsets, source=source)
    item_count = len(ids)
    if len(texts) != item_count or offsets.size != item_count + 1:
        raise ValueError(
            f"{source}: ids, texts, and offsets describe different item counts"
        )
    if len(set(ids)) != item_count:
        raise ValueError(f"{source}: embedding IDs must be unique")
    if shapes.shape != (item_count, 2):
        raise ValueError(
            f"{source}: shapes must have one (tokens, dimension) row per item"
        )

    expected_shapes = np.column_stack(
        (
            np.diff(offsets),
            np.full(item_count, values.shape[1], dtype=np.int64),
        )
    )
    if not np.array_equal(shapes, expected_shapes):
        raise ValueError(f"{source}: shapes do not agree with values and offsets")

    return {
        "ids": ids,
        "texts": texts,
        "values": values,
        "offsets": offsets,
        "shapes": shapes,
    }


def unpack_embeddings(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Reconstruct a list of token-vector matrices from packed storage."""
    values = np.asarray(values)
    offsets = np.asarray(offsets)
    _validate_offsets(values, offsets, source="packed embeddings")
    return [
        np.ascontiguousarray(values[offsets[i] : offsets[i + 1]], dtype=np.float32)
        for i in range(len(offsets) - 1)
    ]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise L2-normalized copy of a vector matrix."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D vector matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def flatten_document_embeddings(
    documents: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten document token matrices and keep token-to-document metadata."""
    if not documents:
        raise ValueError("At least one document embedding is required")

    flat_vectors = []
    token_to_doc_index = []
    token_to_token_position = []
    embedding_dimension: int | None = None

    for doc_index, document_matrix in enumerate(documents):
        document_matrix = np.asarray(document_matrix)
        if document_matrix.ndim != 2 or document_matrix.shape[0] == 0:
            raise ValueError(
                f"Document {doc_index} must contain a non-empty 2D token matrix"
            )
        if embedding_dimension is None:
            embedding_dimension = int(document_matrix.shape[1])
        elif document_matrix.shape[1] != embedding_dimension:
            raise ValueError(
                f"Document {doc_index} has dimension {document_matrix.shape[1]}; "
                f"expected {embedding_dimension}"
            )
        flat_vectors.append(document_matrix)
        token_count = len(document_matrix)
        token_to_doc_index.extend([doc_index] * token_count)
        token_to_token_position.extend(range(token_count))

    return (
        np.ascontiguousarray(np.vstack(flat_vectors), dtype=np.float32),
        np.array(token_to_doc_index, dtype=np.int64),
        np.array(token_to_token_position, dtype=np.int64),
    )


def maxsim_score(
    query_matrix: np.ndarray,
    document_matrix: np.ndarray,
    normalize: bool = False,
) -> float:
    """Compute exact ColBERT-style MaxSim with inner product similarity.

    Score(q, d) = sum_i max_j dot(q_i, d_j), where query/document matrices are
    token-vector matrices. If `normalize=True`, each token vector is first
    L2-normalized, making inner product equivalent to cosine similarity.
    """
    query_matrix = np.asarray(query_matrix)
    document_matrix = np.asarray(document_matrix)
    if query_matrix.ndim != 2 or document_matrix.ndim != 2:
        raise ValueError("MaxSim expects 2D query and document token matrices")
    if query_matrix.shape[0] == 0 or document_matrix.shape[0] == 0:
        raise ValueError("MaxSim requires at least one query and document token")
    if query_matrix.shape[1] != document_matrix.shape[1]:
        raise ValueError(
            "Query and document token vectors must have the same dimension"
        )

    if normalize:
        query_matrix = l2_normalize(query_matrix)
        document_matrix = l2_normalize(document_matrix)

    similarities = query_matrix @ document_matrix.T
    return float(np.max(similarities, axis=1).sum())


def rank_documents(scores: np.ndarray) -> list[int]:
    """Rank document indices by descending score with deterministic tie breaks."""
    return np.lexsort((np.arange(len(scores)), -scores)).astype(int).tolist()


def recall_at_k(
    reference_ranking: list[int],
    candidate_ranking: list[int],
    k: int,
) -> float:
    """Compute set recall@k between a reference and candidate ranking."""
    reference_top_k = set(reference_ranking[:k])
    candidate_top_k = set(candidate_ranking[:k])
    if not reference_top_k:
        return 0.0
    return len(reference_top_k.intersection(candidate_top_k)) / len(reference_top_k)


def topk_overlap(
    reference_ranking: list[str],
    candidate_ranking: list[str],
    k: int,
) -> dict[str, float | int]:
    """Return top-k set overlap count and ratio."""
    reference_top_k = set(reference_ranking[:k])
    candidate_top_k = set(candidate_ranking[:k])
    if not reference_top_k:
        return {"count": 0, "ratio": 0.0}
    overlap = len(reference_top_k.intersection(candidate_top_k))
    return {"count": overlap, "ratio": overlap / len(reference_top_k)}


def mrr_at_k(ranking: list[str], relevant_ids: Iterable[str], k: int) -> float:
    """Compute reciprocal rank for the first relevant item in the top-k."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def compute_qrels_metrics(
    rankings_by_query: dict[str, list[str]],
    qrels: dict[str, list[str]],
    k_values: Iterable[int] = (1, 3, 5),
    mrr_k: int = 10,
) -> dict[str, Any]:
    """Compute macro recall@k and MRR@k against manual qrels."""
    per_query = {}
    recall_sums = {int(k): 0.0 for k in k_values}
    mrr_sum = 0.0
    query_count = 0

    for query_id, relevant_ids in qrels.items():
        ranking = rankings_by_query.get(query_id, [])
        relevant = set(relevant_ids)
        query_metrics = {}
        for k in recall_sums:
            if not relevant:
                recall = 0.0
            else:
                recall = len(set(ranking[:k]).intersection(relevant)) / len(relevant)
            query_metrics[f"recall@{k}"] = recall
            recall_sums[k] += recall

        mrr = mrr_at_k(ranking, relevant, mrr_k)
        query_metrics[f"mrr@{mrr_k}"] = mrr
        mrr_sum += mrr
        per_query[query_id] = query_metrics
        query_count += 1

    if query_count == 0:
        macro = {f"recall@{k}": 0.0 for k in recall_sums}
        macro[f"mrr@{mrr_k}"] = 0.0
    else:
        macro = {f"recall@{k}": recall_sums[k] / query_count for k in recall_sums}
        macro[f"mrr@{mrr_k}"] = mrr_sum / query_count

    return {
        "macro": macro,
        "per_query": per_query,
        "query_count": query_count,
    }


def maxsim_scores_for_candidate_docs(
    query_matrix: np.ndarray,
    documents: list[np.ndarray],
    candidate_doc_indices: Iterable[int],
    normalize: bool = False,
) -> dict[int, float]:
    """Compute exact MaxSim only for selected document indices."""
    scores = {}
    for doc_index in candidate_doc_indices:
        scores[int(doc_index)] = maxsim_score(
            query_matrix,
            documents[int(doc_index)],
            normalize=normalize,
        )
    return scores


def aggregate_token_retrieval_scores(
    query_token_hits: list[list[tuple[int, float]]],
    token_to_doc_index: np.ndarray,
    num_documents: int,
    missing_value: float = 0.0,
) -> tuple[np.ndarray, set[int], int]:
    """Aggregate token-level hits into approximate document scores.

    Each inner hit list contains `(token_index, similarity)` entries for one
    query token. For each query token and document, only the best retrieved
    similarity is kept. Missing query-token/document pairs are filled with
    `missing_value`; current experiments use `0.0` and report that limitation.
    """
    per_token_doc_max = np.full(
        (len(query_token_hits), num_documents),
        -np.inf,
        dtype=np.float32,
    )
    candidate_doc_indices: set[int] = set()
    retrieved_token_count = 0

    for query_token_index, hits in enumerate(query_token_hits):
        retrieved_token_count += len(hits)
        for token_index, similarity in hits:
            doc_index = int(token_to_doc_index[int(token_index)])
            candidate_doc_indices.add(doc_index)
            if similarity > per_token_doc_max[query_token_index, doc_index]:
                per_token_doc_max[query_token_index, doc_index] = similarity

    filled = np.where(np.isfinite(per_token_doc_max), per_token_doc_max, missing_value)
    return filled.sum(axis=0).astype(np.float32), candidate_doc_indices, retrieved_token_count


def topk_pairwise_order_agreement(
    reference_ranking: list[str],
    candidate_ranking: list[str],
    k: int = 10,
) -> float:
    """Compute pairwise order agreement over the union of two top-k lists."""
    docs = list(dict.fromkeys(reference_ranking[:k] + candidate_ranking[:k]))
    if len(docs) < 2:
        return 1.0

    missing_position = len(docs)
    reference_pos = {doc_id: rank for rank, doc_id in enumerate(reference_ranking[:k])}
    candidate_pos = {doc_id: rank for rank, doc_id in enumerate(candidate_ranking[:k])}

    total = 0
    same_order = 0
    for left_index, left_doc in enumerate(docs):
        for right_doc in docs[left_index + 1 :]:
            reference_order = (
                reference_pos.get(left_doc, missing_position)
                < reference_pos.get(right_doc, missing_position)
            )
            candidate_order = (
                candidate_pos.get(left_doc, missing_position)
                < candidate_pos.get(right_doc, missing_position)
            )
            same_order += int(reference_order == candidate_order)
            total += 1

    return same_order / total if total else 1.0
