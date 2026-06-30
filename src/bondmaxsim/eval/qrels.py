"""Standard IR evaluation metrics against qrels (nDCG@10, recall@100, MRR@10).

Single responsibility: given ranked results and BEIR-format qrels, compute
standard IR metrics and populate the quality fields of ResultRecord.

Ported artifact: metric plumbing from
  research/colbert/02b_corect_bruteforce.py and research/colbert/03_beir_comparison.py
  (BEIR-style qrels metric computation); uses ranx for metric computation.
Stage 1 reference: docs/project_b_analysis_and_research_plan.md Stage 5 section
  (fairness controls: same machine, OS, thread count, repeated runs; nDCG@10,
  recall@100, MRR@10 are the primary qrels metrics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ndcg_at_10(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> float:
    """Compute nDCG@10 using ranx.

    Parameters
    ----------
    run   : {query_id: {doc_id: score}} — ranked results for each query
    qrels : {query_id: {doc_id: relevance}} — relevance judgments

    Returns
    -------
    mean nDCG@10 across queries
    """
    raise NotImplementedError


def recall_at_100(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> float:
    """Compute recall@100.

    Parameters
    ----------
    run, qrels : same format as ndcg_at_10

    Returns
    -------
    mean recall@100 across queries
    """
    raise NotImplementedError


def mrr_at_10(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> float:
    """Compute MRR@10.

    Parameters
    ----------
    run, qrels : same format as ndcg_at_10

    Returns
    -------
    mean MRR@10 across queries
    """
    raise NotImplementedError


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load BEIR-format qrels TSV (query_id, doc_id, relevance).

    Returns
    -------
    {query_id: {doc_id: relevance_int}}
    """
    raise NotImplementedError
