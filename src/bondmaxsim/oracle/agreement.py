"""Strict, tie-aware, recall, and ranked-order retrieval validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import numpy as np

FP32_TIE_MAX_ULPS = 8
_FP32_MIN_SUBNORMAL = np.nextafter(
    np.float32(0.0), np.float32(1.0), dtype=np.float32
)


@dataclass(frozen=True)
class AgreementResult:
    strict_top_k_set_equal: bool
    boundary_tie_equivalent: bool
    recall_vs_oracle_set: float
    ranked_order_valid: bool | None
    k: int
    fp32_tie_max_ulps: int
    failure_codes: tuple[str, ...]

    @property
    def exact_gate_passed(self) -> bool:
        correctness_failures = [
            code for code in self.failure_codes if code != "ranked_order_failure"
        ]
        return (
            self.strict_top_k_set_equal or self.boundary_tie_equivalent
        ) and not correctness_failures

    def to_dict(self) -> dict:
        result = asdict(self)
        result["failure_codes"] = list(self.failure_codes)
        return result


@dataclass(frozen=True)
class AgreementBatchResult:
    strict_top_k_set_equal: bool
    boundary_tie_equivalent: bool
    recall_vs_oracle_set_mean: float
    query_count: int
    failure_codes: tuple[str, ...]

    @property
    def exact_gate_passed(self) -> bool:
        return self.query_count > 0 and self.boundary_tie_equivalent

    def to_dict(self) -> dict:
        result = asdict(self)
        result["failure_codes"] = list(self.failure_codes)
        return result


def batch_agreement_result(results: list[AgreementResult]) -> AgreementBatchResult:
    """Retain strict/tie distinctions while aggregating one workload pass."""
    if not results:
        raise ValueError("agreement batch must contain at least one query")
    return AgreementBatchResult(
        strict_top_k_set_equal=all(row.strict_top_k_set_equal for row in results),
        boundary_tie_equivalent=all(row.exact_gate_passed for row in results),
        recall_vs_oracle_set_mean=float(np.mean([
            row.recall_vs_oracle_set for row in results
        ])),
        query_count=len(results),
        failure_codes=tuple(sorted({
            code for row in results for code in row.failure_codes
        })),
    )


def aggregate_agreement_results(results: list[AgreementResult]) -> dict:
    """Aggregate per-query gates without collapsing strict and tie outcomes."""
    return {
        "strict_top_k_set_equal": bool(results) and all(
            result.strict_top_k_set_equal for result in results
        ),
        "boundary_tie_equivalent": bool(results) and all(
            result.exact_gate_passed for result in results
        ),
        "agreement_failure_codes": sorted(
            {code for result in results for code in result.failure_codes}
        ),
    }


def recall_at_k(
    retrieved: np.ndarray,
    relevant: np.ndarray,
    k: int | None = None,
) -> float:
    """Set recall against one oracle tie-breaking choice.

    This is an overlap metric, not proof of strict or tie-equivalent exactness.
    """
    retrieved = np.asarray(retrieved).reshape(-1)
    relevant = np.asarray(relevant).reshape(-1)
    if k is not None:
        retrieved = retrieved[:k]
        relevant = relevant[:k]
    try:
        rel_set = set(int(x) for x in relevant)
        ret_set = set(int(x) for x in retrieved)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not rel_set:
        return 1.0
    return len(ret_set & rel_set) / len(rel_set)


def recall_vs_oracle_set(
    returned_ids: np.ndarray, oracle_ids: np.ndarray, k: int
) -> float:
    """Ordinary set recall at exactly ``k`` positions."""
    return recall_at_k(returned_ids, oracle_ids, k=k)


def fp32_boundary_equal(a: float, b: float) -> bool:
    """Return whether two finite scores are within the symmetric 8-ULP rule."""
    a32, b32 = np.float32(a), np.float32(b)
    if not np.isfinite(a32) or not np.isfinite(b32):
        return False
    spacing = max(
        float(np.spacing(np.abs(a32))),
        float(np.spacing(np.abs(b32))),
        float(_FP32_MIN_SUBNORMAL),
    )
    return abs(float(a32) - float(b32)) <= FP32_TIE_MAX_ULPS * spacing


def _integer_ids(
    name: str, values, k: int, num_documents: int
) -> tuple[np.ndarray | None, list[str]]:
    array = np.asarray(values)
    code = f"malformed_{name}_ids"
    if array.ndim != 1 or len(array) != k or not np.issubdtype(array.dtype, np.integer):
        return None, [code]
    try:
        ids = np.ascontiguousarray(array, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None, [code]
    if len(np.unique(ids)) != k or np.any(ids < 0) or np.any(ids >= num_documents):
        return None, [code]
    return ids, []


def strict_top_k_set_equal(
    returned_ids, oracle_ids, *, k: int, num_documents: int
) -> bool:
    """Exact equality of two structurally valid unique top-k ID sets."""
    returned, returned_failures = _integer_ids(
        "returned", returned_ids, k, num_documents
    )
    oracle, oracle_failures = _integer_ids("oracle", oracle_ids, k, num_documents)
    return not returned_failures and not oracle_failures and set(returned) == set(oracle)


def validate_ranked_order(
    returned_ids,
    returned_scores,
    *,
    k: int,
    num_documents: int,
    tie_break: str = "id_ascending",
) -> bool:
    """Validate finite non-increasing scores and an explicit tie-break policy."""
    ids, failures = _integer_ids("returned", returned_ids, k, num_documents)
    scores = np.asarray(returned_scores)
    if failures or scores.ndim != 1 or len(scores) != k:
        return False
    try:
        scores = np.ascontiguousarray(scores, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        return False
    if not np.isfinite(scores).all() or np.any(scores[1:] > scores[:-1]):
        return False
    if tie_break not in ("id_ascending", "none"):
        raise ValueError("tie_break must be 'id_ascending' or 'none'")
    if tie_break == "id_ascending":
        for index in range(1, k):
            if fp32_boundary_equal(scores[index - 1], scores[index]) and (
                ids[index - 1] > ids[index]
            ):
                return False
    return True


def _replacement_score(
    document_id: int,
    exact_scores_by_id: Mapping[int, float] | np.ndarray | None,
    exact_scorer: Callable[[int], float] | None,
) -> tuple[float | None, str | None]:
    try:
        if exact_scores_by_id is not None:
            if isinstance(exact_scores_by_id, Mapping):
                if document_id not in exact_scores_by_id:
                    return None, "missing_exact_replacement_score"
                return float(exact_scores_by_id[document_id]), None
            values = np.asarray(exact_scores_by_id)
            if values.ndim != 1 or document_id >= len(values):
                return None, "missing_exact_replacement_score"
            return float(values[document_id]), None
        if exact_scorer is not None:
            return float(exact_scorer(document_id)), None
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None, "missing_exact_replacement_score"
    return None, "missing_exact_replacement_score"


def validate_boundary_tie_equivalence(
    returned_ids,
    oracle_ids,
    oracle_scores,
    *,
    k: int,
    num_documents: int,
    exact_scores_by_id: Mapping[int, float] | np.ndarray | None = None,
    exact_scorer: Callable[[int], float] | None = None,
    returned_scores=None,
    ranked_tie_break: str = "id_ascending",
) -> AgreementResult:
    """Validate strict equality or independently exact-scored boundary ties."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")
    if (
        isinstance(num_documents, bool)
        or not isinstance(num_documents, (int, np.integer))
        or num_documents < k
    ):
        raise ValueError("num_documents must be an integer greater than or equal to k")
    k, num_documents = int(k), int(num_documents)
    failures: list[str] = []
    returned, returned_failures = _integer_ids(
        "returned", returned_ids, k, num_documents
    )
    oracle, oracle_failures = _integer_ids("oracle", oracle_ids, k, num_documents)
    failures.extend(returned_failures)
    failures.extend(oracle_failures)

    recall = recall_vs_oracle_set(returned_ids, oracle_ids, k)
    ranked_valid = None
    if returned_scores is not None:
        ranked_valid = validate_ranked_order(
            returned_ids,
            returned_scores,
            k=k,
            num_documents=num_documents,
            tie_break=ranked_tie_break,
        )
        if not ranked_valid:
            failures.append("ranked_order_failure")

    scores = np.asarray(oracle_scores)
    if scores.ndim != 1 or len(scores) != k:
        failures.append("malformed_oracle_scores")
        scores32 = None
    else:
        try:
            scores32 = np.ascontiguousarray(scores, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            scores32 = None
            failures.append("malformed_oracle_scores")
    if scores32 is not None:
        if not np.isfinite(scores32).all():
            failures.append("non_finite_oracle_score")
        elif np.any(scores32[1:] > scores32[:-1]):
            failures.append("oracle_scores_not_non_increasing")

    structural_ok = not returned_failures and not oracle_failures and scores32 is not None
    score_ok = structural_ok and not any(
        code in failures
        for code in ("non_finite_oracle_score", "oracle_scores_not_non_increasing")
    )
    strict = bool(structural_ok and set(returned) == set(oracle))
    boundary = bool(strict and score_ok)

    if score_ok and not strict:
        tau = float(scores32[-1])
        oracle_score = {int(doc): float(score) for doc, score in zip(oracle, scores32)}
        missing = set(int(x) for x in oracle) - set(int(x) for x in returned)
        replacements = set(int(x) for x in returned) - set(int(x) for x in oracle)

        if any(not fp32_boundary_equal(oracle_score[doc], tau) for doc in missing):
            failures.append("above_boundary_miss")

        for document_id in sorted(replacements):
            score, error = _replacement_score(
                document_id, exact_scores_by_id, exact_scorer
            )
            if error is not None:
                failures.append(error)
                continue
            if not np.isfinite(np.float32(score)):
                failures.append("non_finite_replacement_score")
            elif not fp32_boundary_equal(score, tau):
                failures.append(
                    "below_boundary_replacement" if score < tau
                    else "above_boundary_replacement"
                )
        boundary = not any(code != "ranked_order_failure" for code in failures)

    return AgreementResult(
        strict_top_k_set_equal=strict,
        boundary_tie_equivalent=boundary,
        recall_vs_oracle_set=recall,
        ranked_order_valid=ranked_valid,
        k=k,
        fp32_tie_max_ulps=FP32_TIE_MAX_ULPS,
        failure_codes=tuple(dict.fromkeys(failures)),
    )


def exact_agreement(
    pruned_ids: np.ndarray,
    exact_ids: np.ndarray,
    exact_scores: np.ndarray | None = None,
    *,
    num_documents: int | None = None,
    exact_scores_by_id: Mapping[int, float] | np.ndarray | None = None,
    exact_scorer: Callable[[int], float] | None = None,
) -> float:
    """Backward-compatible scalar overlap, conservatively tie-aware.

    Without independent replacement scores, differing sets return ordinary
    recall and can no longer receive a false ``1.0`` tie pass.
    """
    k = len(np.asarray(exact_ids).reshape(-1))
    basic = recall_at_k(pruned_ids, exact_ids, k=k)
    if exact_scores is None or basic == 1.0 or k == 0:
        return basic
    if num_documents is None:
        candidates = np.concatenate(
            [np.asarray(pruned_ids).reshape(-1), np.asarray(exact_ids).reshape(-1)]
        )
        if not np.issubdtype(candidates.dtype, np.integer) or candidates.size == 0:
            return basic
        num_documents = int(np.max(candidates)) + 1
    result = validate_boundary_tie_equivalence(
        pruned_ids,
        exact_ids,
        exact_scores,
        k=k,
        num_documents=num_documents,
        exact_scores_by_id=exact_scores_by_id,
        exact_scorer=exact_scorer,
    )
    return 1.0 if result.exact_gate_passed else basic


def assert_exact_agreement(
    pruned_ids: np.ndarray,
    exact_ids: np.ndarray,
    tol: float = 0.0,
) -> None:
    """Raise when ordinary set recall falls below ``1 - tol``."""
    agreement = exact_agreement(pruned_ids, exact_ids)
    threshold = 1.0 - tol
    if agreement < threshold:
        raise AssertionError(
            f"Oracle set-agreement recall check failed: recall={agreement:.6f} < "
            f"threshold={threshold:.6f} (tol={tol}). "
            f"returned={list(pruned_ids)}, oracle={list(exact_ids)}"
        )
