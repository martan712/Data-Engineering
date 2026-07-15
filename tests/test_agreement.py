"""Tests for bondmaxsim.oracle.agreement."""

import numpy as np
import pytest

from bondmaxsim.oracle.agreement import (
    FP32_TIE_MAX_ULPS,
    assert_exact_agreement,
    exact_agreement,
    fp32_boundary_equal,
    recall_at_k,
    strict_top_k_set_equal,
    validate_boundary_tie_equivalence,
    validate_ranked_order,
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


# ---------------------------------------------------------------------------
# Strict and independently verified boundary-tie validation
# ---------------------------------------------------------------------------

def _validate(returned, oracle=(1, 2, 3), scores=(10.0, 9.0, 8.0), **kwargs):
    return validate_boundary_tie_equivalence(
        np.asarray(returned),
        np.asarray(oracle),
        np.asarray(scores, dtype=np.float32),
        k=len(oracle),
        num_documents=100,
        **kwargs,
    )


def test_strict_set_equality_is_named_and_rank_independent():
    assert strict_top_k_set_equal(
        np.array([3, 1, 2]), np.array([1, 2, 3]), k=3, num_documents=4
    )
    assert not strict_top_k_set_equal(
        np.array([1, 2, 3, 4]), np.array([1, 2, 3]), k=3, num_documents=5
    )


def test_historical_false_pass_is_rejected_without_replacement_evidence():
    result = _validate([1, 2, 99])
    assert not result.strict_top_k_set_equal
    assert not result.boundary_tie_equivalent
    assert result.recall_vs_oracle_set == pytest.approx(2 / 3)
    assert "missing_exact_replacement_score" in result.failure_codes
    assert exact_agreement(
        np.array([1, 2, 99]),
        np.array([1, 2, 3]),
        np.array([10.0, 9.0, 8.0]),
    ) == pytest.approx(2 / 3)


def test_below_boundary_replacement_is_rejected():
    result = _validate([1, 2, 99], exact_scores_by_id={99: 7.5})
    assert not result.exact_gate_passed
    assert result.failure_codes == ("below_boundary_replacement",)


def test_genuine_boundary_replacement_is_accepted_but_not_strict():
    result = _validate([1, 2, 99], exact_scores_by_id={99: 8.0})
    assert not result.strict_top_k_set_equal
    assert result.boundary_tie_equivalent
    assert result.exact_gate_passed
    assert result.recall_vs_oracle_set == pytest.approx(2 / 3)
    assert result.failure_codes == ()
    assert exact_agreement(
        np.array([1, 2, 99]),
        np.array([1, 2, 3]),
        np.array([10.0, 9.0, 8.0]),
        num_documents=100,
        exact_scores_by_id={99: 8.0},
    ) == 1.0


def test_missing_above_boundary_document_is_rejected():
    result = _validate([1, 3, 99], exact_scores_by_id={99: 8.0})
    assert not result.boundary_tie_equivalent
    assert "above_boundary_miss" in result.failure_codes


def test_multiple_boundary_substitutions_are_verified_independently():
    result = _validate(
        [1, 98, 99],
        oracle=(1, 2, 3),
        scores=(10.0, 8.0, 8.0),
        exact_scores_by_id={98: 8.0, 99: 8.0},
    )
    assert result.boundary_tie_equivalent

    failed = _validate(
        [1, 98, 99],
        oracle=(1, 2, 3),
        scores=(10.0, 8.0, 8.0),
        exact_scores_by_id={98: 8.0, 99: 7.0},
    )
    assert not failed.boundary_tie_equivalent
    assert "below_boundary_replacement" in failed.failure_codes


@pytest.mark.parametrize(
    "returned, failure",
    [
        ([1, 2], "malformed_returned_ids"),
        ([1, 2, 3, 4], "malformed_returned_ids"),
        ([1, 1, 3], "malformed_returned_ids"),
        ([1, 2, -1], "malformed_returned_ids"),
        ([1, 2, 100], "malformed_returned_ids"),
        ([1.0, 2.0, 3.0], "malformed_returned_ids"),
    ],
)
def test_malformed_returned_lists_never_pass(returned, failure):
    result = _validate(returned, exact_scores_by_id={})
    assert not result.strict_top_k_set_equal
    assert not result.boundary_tie_equivalent
    assert failure in result.failure_codes


@pytest.mark.parametrize(
    "scores, failure",
    [
        ([10.0, 9.0], "malformed_oracle_scores"),
        ([10.0, np.nan, 8.0], "non_finite_oracle_score"),
        ([10.0, 8.0, 9.0], "oracle_scores_not_non_increasing"),
    ],
)
def test_malformed_oracle_scores_never_pass(scores, failure):
    result = _validate([1, 2, 3], scores=scores)
    assert not result.boundary_tie_equivalent
    assert not result.exact_gate_passed
    assert failure in result.failure_codes


def _step_float32(value: float, direction: float, count: int) -> np.float32:
    result = np.float32(value)
    for _ in range(count):
        result = np.nextafter(result, np.float32(direction), dtype=np.float32)
    return result


def test_fp32_boundary_tolerance_is_symmetric_and_limited_to_eight_ulps():
    plus_eight = _step_float32(8.0, np.inf, FP32_TIE_MAX_ULPS)
    minus_eight = _step_float32(8.0, -np.inf, FP32_TIE_MAX_ULPS)
    plus_nine = _step_float32(8.0, np.inf, FP32_TIE_MAX_ULPS + 1)
    assert fp32_boundary_equal(8.0, plus_eight)
    assert fp32_boundary_equal(plus_eight, 8.0)
    assert fp32_boundary_equal(8.0, minus_eight)
    assert not fp32_boundary_equal(8.0, plus_nine)
    assert not fp32_boundary_equal(8.0, 8.001)


def test_exact_scorer_is_accepted_as_independent_evidence():
    result = _validate([1, 2, 99], exact_scorer=lambda document_id: 8.0)
    assert result.boundary_tie_equivalent


def test_non_finite_replacement_score_is_rejected():
    result = _validate([1, 2, 99], exact_scores_by_id={99: np.inf})
    assert not result.boundary_tie_equivalent
    assert "non_finite_replacement_score" in result.failure_codes


def test_ranked_order_validation_is_separate():
    assert validate_ranked_order(
        np.array([1, 2, 3]), np.array([10.0, 9.0, 8.0]),
        k=3, num_documents=4,
    )
    assert not validate_ranked_order(
        np.array([1, 2, 3]), np.array([10.0, 11.0, 8.0]),
        k=3, num_documents=4,
    )
    assert not validate_ranked_order(
        np.array([2, 1, 3]), np.array([10.0, 10.0, 8.0]),
        k=3, num_documents=4,
    )


def test_result_serialization_uses_contract_fields():
    result = _validate([1, 2, 99], exact_scores_by_id={99: 8.0})
    serialized = result.to_dict()
    assert serialized["strict_top_k_set_equal"] is False
    assert serialized["boundary_tie_equivalent"] is True
    assert serialized["fp32_tie_max_ulps"] == 8
    assert serialized["failure_codes"] == []


def test_small_population_specification_property():
    oracle = np.array([0, 1, 2])
    oracle_scores = np.array([4.0, 3.0, 2.0], dtype=np.float32)
    population_scores = np.array([4.0, 3.0, 2.0, 2.0, 1.0], dtype=np.float32)
    from itertools import combinations

    for returned_tuple in combinations(range(5), 3):
        returned = np.array(returned_tuple)
        result = validate_boundary_tie_equivalence(
            returned,
            oracle,
            oracle_scores,
            k=3,
            num_documents=5,
            exact_scores_by_id=population_scores,
        )
        expected = 0 in returned and 1 in returned and all(
            population_scores[doc] == 2.0 for doc in returned if doc not in oracle
        )
        assert result.exact_gate_passed is expected
