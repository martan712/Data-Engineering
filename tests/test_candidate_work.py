from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline
from bondmaxsim.experiments.candidate_work import (
    CandidateWorkObservation,
    CountObservation,
    summarize_candidate_work,
)


class _FakeFaissIndex:
    def search(self, query: np.ndarray, k: int):
        assert k == 2
        scores = np.array([[0.9, 0.8], [0.7, 0.0]], dtype=np.float32)
        token_ids = np.array([[0, 2], [3, -1]], dtype=np.int64)
        return scores, token_ids


def _baseline() -> FaissIVFBaseline:
    tokens = np.eye(4, dtype=np.float32)
    starts = np.arange(4, dtype=np.int64)
    baseline = FaissIVFBaseline(tokens, starts, n_lists=2)
    baseline.index = _FakeFaissIndex()
    return baseline


def test_faiss_candidate_work_distinguishes_generated_and_admitted():
    baseline = _baseline()
    candidates, work = baseline.candidates_with_work(
        np.eye(2, 4, dtype=np.float32), candidate_budget=2, k_token=2
    )
    assert len(candidates) == 2
    assert work.unique_candidates_generated.value == 3
    assert work.documents_admitted_to_scoring.value == 2
    assert work.documents_fully_scored.quality == "unavailable"
    assert work.token_hits_inspected.value == 3


def test_faiss_topk_reports_exact_full_score_work():
    _, _, work = _baseline().topk_with_work(
        np.eye(2, 4, dtype=np.float32), k=1, candidate_budget=2, k_token=2
    )
    assert work.documents_fully_scored == CountObservation.exact(
        2, "candidate IDs passed once to exact MaxSim reranker"
    )
    work.require_equal_work_reranking()


def test_unavailable_counts_cannot_support_equal_work_claim():
    work = CandidateWorkObservation.empty()
    with pytest.raises(ValueError, match="documents_fully_scored"):
        work.require_equal_work_reranking()
    with pytest.raises(ValueError, match="value=None"):
        CountObservation(1, "unavailable", "bad")


def test_candidate_work_summary_retains_unavailable_count():
    first = CandidateWorkObservation.empty()
    second = replace(
        first,
        documents_probed=CountObservation.exact(12, "partition offsets"),
    )
    summary = summarize_candidate_work([first, second])["documents_probed"]
    assert summary == {
        "count": 2,
        "min": 12,
        "median": 12.0,
        "mean": 12.0,
        "max": 12,
        "unavailable": 1,
    }
