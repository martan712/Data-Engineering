"""Tests for bondmaxsim.oracle.agreement."""

import numpy as np
import pytest

from bondmaxsim.oracle.agreement import (
    recall_at_k,
    exact_agreement,
    assert_exact_agreement,
)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k_identical_sets():
    ids = np.array([0, 1, 2, 3, 4])
    assert recall_at_k(ids, ids) == 1.0


def test_recall_at_k_full_overlap():
    retrieved = np.array([3, 1, 4, 1, 5])  # note duplicates — treated as set
    relevant = np.array([1, 3, 5])
    assert recall_at_k(retrieved, relevant) == 1.0


def test_recall_at_k_partial_overlap():
    retrieved = np.array([0, 1, 2, 3])
    relevant = np.array([0, 1, 4, 5])  # 2 out of 4 overlap
    result = recall_at_k(retrieved, relevant)
    assert abs(result - 0.5) < 1e-9


def test_recall_at_k_no_overlap():
    retrieved = np.array([0, 1, 2])
    relevant = np.array([3, 4, 5])
    assert recall_at_k(retrieved, relevant) == 0.0


def test_recall_at_k_empty_relevant():
    retrieved = np.array([0, 1, 2])
    relevant = np.array([], dtype=np.int64)
    assert recall_at_k(retrieved, relevant) == 1.0


def test_recall_at_k_with_truncation():
    # Only the first k elements should be considered
    retrieved = np.array([99, 1, 2, 3])   # first element is irrelevant
    relevant = np.array([1, 2, 3, 99])
    # Without truncation: all 4 overlap -> recall = 1.0
    assert recall_at_k(retrieved, relevant) == 1.0
    # With k=1: only retrieved[:1]=[99] vs relevant[:1]=[1] -> 0 overlap
    assert recall_at_k(retrieved, relevant, k=1) == 0.0


def test_recall_at_k_correct_fraction():
    retrieved = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    relevant = np.array([0, 10, 20, 30])  # only 0 is in retrieved
    result = recall_at_k(retrieved, relevant)
    assert abs(result - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# exact_agreement
# ---------------------------------------------------------------------------

def test_exact_agreement_identical():
    ids = np.array([5, 3, 1, 9, 7])
    assert exact_agreement(ids, ids) == 1.0


def test_exact_agreement_partial():
    pruned = np.array([0, 1, 2, 99])
    exact = np.array([0, 1, 2, 3])
    # 3 out of 4 overlap
    result = exact_agreement(pruned, exact)
    assert abs(result - 0.75) < 1e-9


def test_exact_agreement_uses_len_exact_ids_as_k():
    # exact_ids has 3 elements; pruned has 5 but only first 3 should count
    exact = np.array([10, 20, 30])
    pruned = np.array([10, 20, 30, 40, 50])
    # recall_at_k(pruned, exact, k=3) -> pruned[:3]={10,20,30} vs exact={10,20,30} -> 1.0
    assert exact_agreement(pruned, exact) == 1.0


# ---------------------------------------------------------------------------
# assert_exact_agreement
# ---------------------------------------------------------------------------

def test_assert_exact_agreement_passes_identical():
    ids = np.array([0, 1, 2, 3, 4])
    assert_exact_agreement(ids, ids)  # should not raise


def test_assert_exact_agreement_raises_on_mismatch():
    pruned = np.array([0, 1, 2, 99])
    exact = np.array([0, 1, 2, 3])
    with pytest.raises(AssertionError) as exc_info:
        assert_exact_agreement(pruned, exact)
    msg = str(exc_info.value)
    assert "agreement" in msg.lower()


def test_assert_exact_agreement_passes_with_tol():
    pruned = np.array([0, 1, 2, 99])   # 3/4 = 0.75 agreement
    exact = np.array([0, 1, 2, 3])
    # tol=0.3 means threshold = 0.7 < 0.75 -> should pass
    assert_exact_agreement(pruned, exact, tol=0.3)


def test_assert_exact_agreement_raises_below_tol():
    pruned = np.array([0, 1, 2, 99])   # 3/4 = 0.75 agreement
    exact = np.array([0, 1, 2, 3])
    # tol=0.2 means threshold = 0.8 > 0.75 -> should raise
    with pytest.raises(AssertionError):
        assert_exact_agreement(pruned, exact, tol=0.2)


def test_assert_exact_agreement_hard_gate_raises_on_any_mismatch():
    pruned = np.array([0, 1, 2, 3, 99])  # 4 out of 5 relevant
    exact = np.array([0, 1, 2, 3, 4])
    with pytest.raises(AssertionError):
        assert_exact_agreement(pruned, exact, tol=0.0)
