"""CoRECT IR evaluation framework wrapper.

Single responsibility: wrap extern/CoRECT/ to compute CoRECT RC metrics for a
retrieval run and populate the CoRECT_RC_metrics field of ResultRecord.

Ported artifact: CoRECT framework from
  extern/CoRECT/ (pinned commit fedf8bb2, padas-lab-de/CoRECT); we call its
  corect.utils.evaluate_results (pytrec_eval-based NDCG/MAP/Recall/P/MRR at
  cutoffs) via a sys.path shim — the repo is referenced in place, never
  copied.  Its CoRE corpus pools (100k-1M scale axis) are future work with R7.
Stage 1 reference: docs/project_b_analysis_and_research_plan.md Stage 5 section
  (CoRECT RC metrics are a primary quality signal alongside nDCG@10; CoRECT
  wrapper smoke test required before scaling).
"""

from __future__ import annotations

import sys
from typing import Any

from bondmaxsim.config import EXTERN_DIR

CORECT_DIR = EXTERN_DIR / "CoRECT"

RC_K_VALUES: tuple[int, ...] = (10, 100)
"""Default RC metric cutoffs — matched to the qrels metrics (@10, @100)."""


def _import_evaluate_results():
    """Import corect.utils.evaluate_results from the pinned checkout.

    corect.utils and corect.model_wrappers import each other; importing
    model_wrappers FIRST binds AbstractModelWrapper before utils' back-edge
    runs, which breaks the cycle (the order CoRECT's own CLI happens to use).
    """
    src = str(CORECT_DIR / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import corect.model_wrappers  # noqa: F401  (must precede corect.utils)
    from corect.utils import evaluate_results

    return evaluate_results


def compute_rc_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = RC_K_VALUES,
) -> dict[str, Any]:
    """Compute CoRECT RC metrics for a retrieval run.

    Parameters
    ----------
    run      : {query_id: {doc_id: score}} — ranked results
    qrels    : {query_id: {doc_id: relevance}} — relevance judgments
    k_values : metric cutoffs (CoRECT computes NDCG/MAP/Recall/P/MRR at each)

    Returns
    -------
    flat dict of CoRECT RC metric values, keys following CoRECT conventions
    (e.g. "NDCG@10", "MAP@10", "Recall@100", "P@10", "MRR@10")
    """
    evaluate_results = _import_evaluate_results()

    judged = {qid: docs for qid, docs in run.items() if qid in qrels}
    if not judged:
        raise ValueError(
            "No overlap between run queries and qrels — check the ID sidecar "
            "(bondmaxsim.data.beir_ids) and test-query encoding."
        )
    # Restrict qrels to the judged run so pytrec_eval averages over the same
    # query subset as ranx (bondmaxsim.eval.qrels); otherwise unretrieved qrels
    # queries score zero and the two metric sources diverge on a subset run.
    sub_qrels = {qid: qrels[qid] for qid in judged}
    metric_dicts = evaluate_results(sub_qrels, judged, list(k_values))

    flat: dict[str, float] = {}
    for d in metric_dicts:
        for key, value in d.items():
            flat[key] = float(value)
    return flat


def corect_smoke_test(
    run: dict[str, dict[str, float]] | None = None,
    qrels: dict[str, dict[str, int]] | None = None,
    atol: float = 5e-5,
) -> bool:
    """Verify CoRECT metrics agree with our ranx qrels metrics on one run.

    With no arguments, checks a small deterministic synthetic run; the Stage 5
    driver calls it again with the real dense_fused run before scaling (the
    plan's "CoRECT wrapper smoke test" gate).

    Returns True if nDCG@10, recall@100 and MRR@10 agree within atol; raises
    AssertionError (with both values) otherwise.  The default atol absorbs
    CoRECT's round(..., 5) on each metric while still catching definitional
    mismatches.
    """
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
    rc = compute_rc_metrics(run, qrels)

    pairs = [
        ("nDCG_at_10", "NDCG@10"),
        ("recall_at_100", "Recall@100"),
        ("MRR_at_10", "MRR@10"),
    ]
    for ranx_key, rc_key in pairs:
        a, b = ours[ranx_key], rc[rc_key]
        assert abs(a - b) <= atol, (
            f"CoRECT smoke test FAILED: {rc_key}={b} vs ranx {ranx_key}={a}"
        )
    return True
