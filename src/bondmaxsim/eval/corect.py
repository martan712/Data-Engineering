"""CoRECT IR evaluation framework wrapper.

Single responsibility: wrap extern/CoRECT/ to compute CoRECT RC metrics for a
retrieval run and populate the CoRECT_RC_metrics field of ResultRecord.

Ported artifact: CoRECT framework from
  extern/CoRECT/ (pinned commit fedf8bb2, padas-lab-de/CoRECT).
Stage 1 reference: docs/project_b_analysis_and_research_plan.md Stage 5 section
  (CoRECT RC metrics are a primary quality signal alongside nDCG@10; CoRECT
  wrapper smoke test required before scaling), Stage 0 (CoRECT evaluation
  framework from Univ. Passau / OWS.EU partners must be central to the final
  evaluation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bondmaxsim.config import EXTERN_DIR

CORECT_DIR = EXTERN_DIR / "CoRECT"


def compute_rc_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    dataset: str,
) -> dict[str, Any]:
    """Compute CoRECT RC metrics for a retrieval run.

    Parameters
    ----------
    run     : {query_id: {doc_id: score}} — ranked results
    qrels   : {query_id: {doc_id: relevance}} — relevance judgments
    dataset : str — BEIR corpus name (used to locate corpus-level stats)

    Returns
    -------
    dict of CoRECT RC metric values (key names follow CoRECT conventions)
    """
    raise NotImplementedError


def corect_smoke_test(dataset: str = "scifact") -> bool:
    """Verify CoRECT produces expected metrics on a small ColBERT run.

    Returns True if smoke test passes.  Must be run before any Stage 5 scaling.
    """
    raise NotImplementedError
