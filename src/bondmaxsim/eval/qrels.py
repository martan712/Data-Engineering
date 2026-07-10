"""Standard IR evaluation metrics against qrels (nDCG@10, recall@100, MRR@10).

Single responsibility: given ranked results and BEIR-format qrels, compute
standard IR metrics and populate the quality fields of ResultRecord.

Ported artifact: metric plumbing from
  archive/colbert_scripts/02b_corect_bruteforce.py and 03_beir_comparison.py
  (BEIR-style qrels metric computation); uses ranx for metric computation.
Stage 1 reference: docs/project_b_analysis_and_research_plan.md Stage 5 section
  (fairness controls: same machine, OS, thread count, repeated runs; nDCG@10,
  recall@100, MRR@10 are the primary qrels metrics).

recall_vs_exact@10 is NOT computed here — it needs doc indices, not qrels; use
bondmaxsim.oracle.agreement.recall_at_k against the dense_fused ranking.
"""

from __future__ import annotations

from pathlib import Path

from bondmaxsim.data.beir_ids import load_qrels_tsv


def _evaluate(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    metrics: list[str],
) -> dict[str, float]:
    """Evaluate a run against qrels with ranx, restricted to judged queries.

    Metrics are averaged over the queries that are both in the run and have
    judgments (BEIR convention).  Run queries without judgments are dropped;
    qrels queries not in the run are dropped too, so a subset run is scored on
    its own queries rather than the full benchmark.  Raises ValueError if the
    run shares no query with the qrels.
    """
    from ranx import Qrels, Run, evaluate  # [retrieval] extra

    judged = {qid: docs for qid, docs in run.items() if qid in qrels}
    if not judged:
        raise ValueError(
            "No overlap between run queries and qrels — check the ID sidecar "
            "(bondmaxsim.data.beir_ids) and test-query encoding."
        )
    sub_qrels = {qid: qrels[qid] for qid in judged}
    out = evaluate(Qrels(sub_qrels), Run(judged), metrics)
    if isinstance(out, float):  # single-metric convenience form
        return {metrics[0]: out}
    return out


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
    mean nDCG@10 across judged queries
    """
    return float(_evaluate(run, qrels, ["ndcg@10"])["ndcg@10"])


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
    mean recall@100 across judged queries
    """
    return float(_evaluate(run, qrels, ["recall@100"])["recall@100"])


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
    mean MRR@10 across judged queries
    """
    return float(_evaluate(run, qrels, ["mrr@10"])["mrr@10"])


def compute_quality_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Compute all ResultRecord qrels-quality fields in one ranx pass.

    Returns
    -------
    {"nDCG_at_10": ..., "recall_at_100": ..., "MRR_at_10": ...}
    (keys spelled as the ResultRecord fields)
    """
    out = _evaluate(run, qrels, ["ndcg@10", "recall@100", "mrr@10"])
    return {
        "nDCG_at_10": float(out["ndcg@10"]),
        "recall_at_100": float(out["recall@100"]),
        "MRR_at_10": float(out["mrr@10"]),
    }


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load BEIR-format qrels TSV (query_id, doc_id, relevance).

    Thin re-export of bondmaxsim.data.beir_ids.load_qrels_tsv (the data layer
    owns the TSV format; eval callers import from here).

    Returns
    -------
    {query_id: {doc_id: relevance_int}}
    """
    return load_qrels_tsv(Path(path))
