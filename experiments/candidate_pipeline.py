"""Shared token-hit aggregation and deterministic candidate selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateFeatures:
    """Document-level features derived from token-level ANN hits."""

    candidate_doc_indices: np.ndarray
    approx_scores: np.ndarray
    retrieved_token_count: np.ndarray
    matched_query_token_count: np.ndarray

    @property
    def pool_size(self) -> int:
        return int(self.candidate_doc_indices.size)


def build_candidate_features(
    query_token_hits: Sequence[Sequence[tuple[int, float]]],
    token_to_doc_index: np.ndarray,
    num_documents: int,
) -> CandidateFeatures:
    """Map token hits to document candidates and aggregation features.

    Approximate score follows the historical Mikel pipeline: for each query
    token, retain the best retrieved similarity for a document and use zero for
    missing query-token/document pairs. Exact MaxSim is applied after selection.
    """
    if num_documents <= 0:
        raise ValueError("num_documents must be positive")

    token_to_doc = np.asarray(token_to_doc_index, dtype=np.int64)
    per_token_doc_best = np.full(
        (len(query_token_hits), num_documents),
        -np.inf,
        dtype=np.float32,
    )
    retrieved_token_count = np.zeros(num_documents, dtype=np.int32)
    candidate_mask = np.zeros(num_documents, dtype=bool)

    for query_token_index, hits in enumerate(query_token_hits):
        for token_index, similarity in hits:
            token = int(token_index)
            if token < 0 or token >= token_to_doc.size:
                raise IndexError(f"Token index out of bounds: {token}")
            doc_index = int(token_to_doc[token])
            if doc_index < 0 or doc_index >= num_documents:
                raise IndexError(f"Document index out of bounds: {doc_index}")

            candidate_mask[doc_index] = True
            retrieved_token_count[doc_index] += 1
            if similarity > per_token_doc_best[query_token_index, doc_index]:
                per_token_doc_best[query_token_index, doc_index] = similarity

    matched = np.isfinite(per_token_doc_best)
    approx_scores = np.where(matched, per_token_doc_best, 0.0).sum(axis=0)
    return CandidateFeatures(
        candidate_doc_indices=np.flatnonzero(candidate_mask).astype(np.int64),
        approx_scores=np.asarray(approx_scores, dtype=np.float32),
        retrieved_token_count=retrieved_token_count,
        matched_query_token_count=matched.sum(axis=0).astype(np.int32),
    )


def _approx_order(features: CandidateFeatures) -> list[int]:
    return sorted(
        features.candidate_doc_indices.tolist(),
        key=lambda doc: (-float(features.approx_scores[doc]), int(doc)),
    )


def _count_order(features: CandidateFeatures) -> list[int]:
    return sorted(
        features.candidate_doc_indices.tolist(),
        key=lambda doc: (
            -int(features.retrieved_token_count[doc]),
            -float(features.approx_scores[doc]),
            int(doc),
        ),
    )


def select_candidates(
    features: CandidateFeatures,
    top_c: int,
    policy: str = "approx_score",
) -> list[int]:
    """Select at most `top_c` documents with a deterministic policy."""
    if top_c < 0:
        raise ValueError("top_c must be non-negative")
    if features.pool_size == 0:
        return []

    approx_order = _approx_order(features)
    if top_c == 0 or top_c >= features.pool_size:
        return approx_order

    if policy == "approx_score":
        return approx_order[:top_c]

    count_order = _count_order(features)
    if policy == "retrieved_token_count":
        return count_order[:top_c]

    if policy == "matched_query_token_count":
        return sorted(
            features.candidate_doc_indices.tolist(),
            key=lambda doc: (
                -int(features.matched_query_token_count[doc]),
                -float(features.approx_scores[doc]),
                int(doc),
            ),
        )[:top_c]

    if policy == "union_approx_and_count":
        approx_take = math.ceil(top_c / 2)
        count_take = top_c - approx_take
        approx_head = set(approx_order[:approx_take])
        count_head = set(count_order[:count_take])
        selected = approx_head | count_head

        for doc_index in approx_order:
            if len(selected) >= top_c:
                break
            selected.add(doc_index)

        return sorted(
            selected,
            key=lambda doc: (
                -(doc in approx_head and doc in count_head),
                -int(features.matched_query_token_count[doc]),
                -float(features.approx_scores[doc]),
                int(doc),
            ),
        )[:top_c]

    raise ValueError(f"Unknown candidate selection policy: {policy}")


def actual_rerank_count(selected_by_query: Iterable[Sequence[int]]) -> int:
    """Count the query-document pairs passed to exact MaxSim reranking."""
    return sum(len(selected) for selected in selected_by_query)

