"""Artifact-ready Stage 5 ranking validation and quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from bondmaxsim.eval.corect import (
    compute_corect_standard_metrics,
    corect_metric_crosscheck,
)
from bondmaxsim.eval.qrels import compute_quality_metrics, per_query_ndcg_at_10
from bondmaxsim.oracle.agreement import (
    AgreementBatchResult,
    batch_agreement_result,
    recall_at_k,
    validate_boundary_tie_equivalence,
)


@dataclass(frozen=True)
class RankedQueryResult:
    """One emitted ranking plus optional candidate-work accounting."""

    query_id: str
    ids: np.ndarray
    scores: np.ndarray
    candidate_work: Mapping[str, Any] | None = None
    accounting: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RetrievalPassResult:
    """One complete, timed pass over a fixed ordered query workload."""

    queries: tuple[RankedQueryResult, ...]


def validate_ranked_pass(
    result: RetrievalPassResult,
    *,
    query_ids: Sequence[str],
    num_documents: int,
) -> None:
    """Reject malformed, duplicated, misordered, or misaligned rankings."""
    if tuple(row.query_id for row in result.queries) != tuple(query_ids):
        raise ValueError("retrieval pass does not match the ordered query workload")
    for row in result.queries:
        ids = np.asarray(row.ids)
        scores = np.asarray(row.scores)
        if ids.ndim != 1 or scores.ndim != 1 or len(ids) != len(scores):
            raise ValueError(f"{row.query_id}: ranking arrays must be aligned vectors")
        if len(ids) and (
            not np.issubdtype(ids.dtype, np.integer)
            or int(ids.min()) < 0
            or int(ids.max()) >= num_documents
        ):
            raise ValueError(f"{row.query_id}: document IDs are out of range")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{row.query_id}: duplicate document IDs")
        if not np.isfinite(scores).all():
            raise ValueError(f"{row.query_id}: ranking contains non-finite scores")
        if len(scores) > 1 and np.any(scores[:-1] < scores[1:]):
            raise ValueError(f"{row.query_id}: scores are not descending")


def validate_exact_pass(
    result: RetrievalPassResult,
    *,
    query_ids: Sequence[str],
    exact_ids: Sequence[np.ndarray],
    exact_scores: Sequence[np.ndarray],
    exact_full_scores: Sequence[np.ndarray],
    num_documents: int,
    k: int,
) -> AgreementBatchResult:
    """Apply the shared strict/boundary-tie exactness gate to every query."""
    validate_ranked_pass(result, query_ids=query_ids, num_documents=num_documents)
    checks = []
    for row, oracle_ids, oracle_scores, full_scores in zip(
        result.queries, exact_ids, exact_scores, exact_full_scores, strict=True
    ):
        checks.append(
            validate_boundary_tie_equivalence(
                np.asarray(row.ids[:k], dtype=np.int64),
                np.asarray(oracle_ids[:k], dtype=np.int64),
                np.asarray(oracle_scores[:k], dtype=np.float32),
                k=k,
                num_documents=num_documents,
                exact_scores_by_id=np.asarray(full_scores, dtype=np.float32),
                returned_scores=np.asarray(row.scores[:k], dtype=np.float32),
                ranked_tie_break="none",
            )
        )
    return batch_agreement_result(checks)


def rank_run(
    result: RetrievalPassResult,
    corpus_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Convert rankings to evaluator input without reintroducing score ties."""
    run: dict[str, dict[str, float]] = {}
    for row in result.queries:
        size = len(row.ids)
        run[row.query_id] = {
            corpus_ids[int(document_id)]: float(size - rank)
            for rank, document_id in enumerate(row.ids)
        }
    return run


def serialize_rankings(result: RetrievalPassResult) -> list[dict[str, Any]]:
    """Retain per-query rankings so downstream analysis never reruns retrieval."""
    return [
        {
            "query_id": row.query_id,
            "document_ids": [int(value) for value in row.ids],
            "scores": [float(value) for value in row.scores],
        }
        for row in result.queries
    ]


def evaluate_pass(
    result: RetrievalPassResult,
    *,
    corpus_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    exact_ids: Sequence[np.ndarray],
    k_eval: int,
    crosscheck_corect: bool = False,
) -> dict[str, Any]:
    """Compute standard metrics, exact recall, and retained per-query values."""
    run = rank_run(result, corpus_ids)
    if crosscheck_corect:
        corect_metric_crosscheck(run, dict(qrels))
    standard = compute_quality_metrics(run, dict(qrels))
    per_query = per_query_ndcg_at_10(run, dict(qrels))
    recalls = [
        recall_at_k(row.ids[:k_eval], oracle[:k_eval])
        for row, oracle in zip(result.queries, exact_ids, strict=True)
    ]
    return {
        "ndcg_at_10": standard["nDCG_at_10"],
        "recall_at_100": standard["recall_at_100"],
        "mrr_at_10": standard["MRR_at_10"],
        "recall_vs_oracle_set": float(np.mean(recalls)),
        "corect_standard_metrics": compute_corect_standard_metrics(run, dict(qrels)),
        "per_query_ndcg_at_10": per_query,
        "rankings": serialize_rankings(result),
    }
