from __future__ import annotations

import numpy as np
import pytest

from bondmaxsim.baselines.plaid import PLAIDBaseline
from bondmaxsim.compat.plaid import inspect_plaid_capabilities


class _PublicPlaid:
    def __init__(self, index_folder, index_name, n_ivf_probe, n_full_scores):
        del index_folder, index_name, n_ivf_probe, n_full_scores

    def __call__(self, queries, k=10):
        del queries, k


def test_plaid_pinned_public_features_are_checked():
    capabilities = inspect_plaid_capabilities(
        _PublicPlaid, installed_version="1.6.0"
    )
    assert capabilities.public_search_settings
    assert not capabilities.actual_full_score_count
    with pytest.raises(RuntimeError, match="1.6.0"):
        inspect_plaid_capabilities(_PublicPlaid, installed_version="1.5.0")


class _FakeLoadedPlaid:
    def __call__(self, queries, k):
        return [[{"id": "3", "score": 0.75}][:k] for _ in queries]


def test_plaid_work_marks_actual_full_scores_unavailable():
    baseline = PLAIDBaseline("fixture", n_full_scores=100)
    baseline._plaid = _FakeLoadedPlaid()
    baseline._capabilities = inspect_plaid_capabilities(
        _PublicPlaid, installed_version="1.6.0"
    )
    results, work = baseline.search_with_work(
        [np.eye(2, dtype=np.float32)], k=1
    )
    assert results[0][0].tolist() == [3]
    assert work[0].configured_full_score_cap.value == 100
    assert work[0].documents_fully_scored.value is None
    assert work[0].documents_fully_scored.quality == "unavailable"
    with pytest.raises(ValueError, match="system_cap"):
        baseline.search_with_work(
            [np.eye(2, dtype=np.float32)], comparison_scope="equal_work_reranking"
        )
