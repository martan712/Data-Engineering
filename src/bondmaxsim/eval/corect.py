"""Standard IR metric cross-validation through the pinned CoRECT checkout.

This module calls CoRECT's ordinary qrels-based ``evaluate_results`` function.
It does not run CoRECT's separate Relevance Composition evaluation.
"""

from __future__ import annotations

import warnings
from typing import Any

from bondmaxsim.compat.corect import load_corect_evaluate_results

CORECT_K_VALUES: tuple[int, ...] = (10, 100)
"""Default standard metric cutoffs, matched to the local qrels metrics."""


def compute_corect_standard_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = CORECT_K_VALUES,
) -> dict[str, Any]:
    """Compute ordinary NDCG/MAP/Recall/P/MRR via CoRECT."""
    evaluate_results = load_corect_evaluate_results()

    judged = {qid: docs for qid, docs in run.items() if qid in qrels}
    if not judged:
        raise ValueError(
            "No overlap between run queries and qrels — check the ID sidecar "
            "(bondmaxsim.data.beir_ids) and test-query encoding."
        )
    sub_qrels = {qid: qrels[qid] for qid in judged}
    metric_dicts = evaluate_results(sub_qrels, judged, list(k_values))

    flat: dict[str, float] = {}
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            flat[key] = float(value)
    return flat


def corect_metric_crosscheck(
    run: dict[str, dict[str, float]] | None = None,
    qrels: dict[str, dict[str, int]] | None = None,
    atol: float = 5e-5,
) -> bool:
    """Verify CoRECT standard metrics agree with local ranx-backed metrics."""
    from bondmaxsim.eval.qrels import compute_quality_metrics

    if run is None or qrels is None:
        qrels = {
            "q1": {"d1": 2, "d3": 1},
            "q2": {"d2": 1},
            "q3": {"d5": 1, "d6": 1},
        }
        run = {
            "q1": {"d1": 0.9, "d2": 0.8, "d3": 0.1},
            "q2": {"d4": 0.7, "d2": 0.6},
            "q3": {"d6": 0.5, "d7": 0.4, "d1": 0.3},
        }

    ours = compute_quality_metrics(run, qrels)
    corect_standard = compute_corect_standard_metrics(run, qrels)
    pairs = [
        ("nDCG_at_10", "NDCG@10"),
        ("recall_at_100", "Recall@100"),
        ("MRR_at_10", "MRR@10"),
    ]
    for local_key, corect_key in pairs:
        local_value = ours[local_key]
        corect_value = corect_standard[corect_key]
        assert abs(local_value - corect_value) <= atol, (
            f"CoRECT metric crosscheck failed: {corect_key}={corect_value} "
            f"vs local {local_key}={local_value}"
        )
    return True


# Historical API compatibility. New code must use the standard-metric names.
RC_K_VALUES = CORECT_K_VALUES


def compute_rc_metrics(*args, **kwargs):
    warnings.warn(
        "compute_rc_metrics is deprecated; use compute_corect_standard_metrics",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_corect_standard_metrics(*args, **kwargs)


def corect_smoke_test(*args, **kwargs):
    warnings.warn(
        "corect_smoke_test is deprecated; use corect_metric_crosscheck",
        DeprecationWarning,
        stacklevel=2,
    )
    return corect_metric_crosscheck(*args, **kwargs)
