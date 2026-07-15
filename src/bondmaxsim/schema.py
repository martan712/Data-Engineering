"""Shared result schema for all bondmaxsim experiments.

Single responsibility: define ResultRecord — the single JSON-serialisable
dataclass that every experiment driver writes into results/json/.  Provides
to_json / from_json / to_dict helpers.

Ported artifact: schema definition from
  docs/project_b_analysis_and_research_plan.md ("Shared Result Schema" section).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §3 (exact/approx
  arm separation must be visible in `method` and `threshold_policy`), §6 (cost
  model: cells_scanned_pct vs ms_per_query must not be conflated).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ResultRecord:
    """One experiment result row.

    Exact and approximate arms are distinguished by `method` and
    `threshold_policy` (Stage 1 §3 / Convention 4 in project_structure.md).
    Accounting metrics (cells_scanned_pct, bound_checks_per_query) and
    throughput metrics (ms_per_query, qps) are always reported separately
    (Stage 1 §6 / Convention 5).  Use None for fields that genuinely do not
    apply to a given method arm; never leave quality metrics at 0 in final
    results.
    """

    # ------------------------------------------------------------------
    # Dataset / run identity
    # ------------------------------------------------------------------
    dataset: str
    """BEIR corpus name, e.g. 'scifact', 'nfcorpus', 'arguana', 'scidocs'."""

    num_docs: int
    """Number of documents in the scored candidate set."""

    num_queries: int
    """Number of queries evaluated."""

    # ------------------------------------------------------------------
    # Method description
    # ------------------------------------------------------------------
    method: str
    """Method identifier, e.g. 'bond_pdx_maxsim_exact_safe', 'faiss_ivf',
    'exact_maxsim', 'plaid'.  Must map to a concrete algorithm (Convention 2)."""

    candidate_budget: Optional[int]
    """IVF/PLAID candidate set size, or None for exhaustive methods."""

    dimension_order: str
    """Dimension-order signal: 'natural', 'bond_q2', 'bond_dtm', 'bond_q2_var',
    'pca_rotation', or another named signal (Stage 1 §4.5, M6 in methodology)."""

    threshold_policy: str
    """Pruning threshold policy: 'self_bound', 'oracle', 'seed', or
    'exact_safe_topk'.  Must distinguish exact from approximate arms (Stage 1
    §4.4, §3)."""

    # ------------------------------------------------------------------
    # Retrieval quality
    # ------------------------------------------------------------------
    recall_vs_exact_at_10: Optional[float]
    """Set recall@10 against one exact-MaxSim top-k tie-breaking choice.
    This is not strict equality or verified tie equivalence. None when exact
    MaxSim is not the reference."""

    nDCG_at_10: Optional[float]
    """nDCG@10 against qrels.  None when qrels not available."""

    recall_at_100: Optional[float]
    """Recall@100 against qrels.  None when qrels not available."""

    MRR_at_10: Optional[float]
    """MRR@10 against qrels.  None when qrels not available."""

    corect_standard_metrics: Optional[dict[str, Any]]
    """Ordinary qrels metrics cross-validated through CoRECT, or None."""

    # ------------------------------------------------------------------
    # Throughput (throughput-mode kernel, Stage 1 §6)
    # ------------------------------------------------------------------
    ms_per_query: Optional[float]
    """Wall-clock latency in milliseconds per query (min-of-repeats after
    warmup).  Produced by throughput-mode kernel (exp-10 style)."""

    qps: Optional[float]
    """Queries per second.  Derived from ms_per_query when set."""

    # ------------------------------------------------------------------
    # Algorithmic work (accounting-mode kernel, Stage 1 §6)
    # ------------------------------------------------------------------
    cells_scanned_pct: Optional[float]
    """Percentage of (query-token, doc-token, dimension) multiply-adds actually
    performed vs brute force.  Produced by accounting-mode kernel (exp-09 style;
    counts only live-set operations)."""

    pruned_docs_pct: Optional[float]
    """Percentage of documents pruned before full scoring."""

    bound_checks_per_query: Optional[int]
    """Number of document upper-bound comparisons per query."""

    # ------------------------------------------------------------------
    # Hardware / environment
    # ------------------------------------------------------------------
    machine: str
    """Hostname or CPU model string for fair-comparison tracking."""

    os: str
    """Operating system, e.g. 'linux', 'windows'."""

    thread_count: int
    """Number of threads used."""

    # ------------------------------------------------------------------
    # Fields below carry a default so they must come last for dataclass
    # positional-argument ordering; conceptually they belong to the groups
    # named in their docstrings, not to a "trailing" group of their own.
    # ------------------------------------------------------------------

    shrink: Optional[float] = field(default=None)
    """Residual shrink factor (Method description group): 1.0 = exact-safe
    arm, < 1.0 = approximate arm.  Machine-separable companion to `method`/
    `threshold_policy` for the exact/approximate split (Stage 1 §3)."""

    tokens_pruned_pct: Optional[float] = field(default=None)
    """Percentage of document tokens removed from per-document live sets
    before full scoring (Algorithmic work / accounting group, conceptually
    next to `pruned_docs_pct`).  Produced by accounting-mode kernel
    (`stats[2]`, Stage 1 §5.3)."""

    strict_top_k_set_equal: Optional[bool] = field(default=None)
    """Whether every validated query returned exactly the oracle top-k ID set."""

    boundary_tie_equivalent: Optional[bool] = field(default=None)
    """Whether every query was strict or differed only by independently
    exact-scored substitutions at the fp32 top-k boundary."""

    agreement_failure_codes: Optional[list[str]] = field(default=None)
    """Stable failure codes from the repaired correctness validator."""

    # ------------------------------------------------------------------
    # Free-text notes
    # ------------------------------------------------------------------
    notes: Optional[str] = field(default=None)
    """Free-text annotation (e.g. 'placeholder', 'shrink=0.9 approximate arm')."""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (JSON-compatible types only)."""
        return asdict(self)

    def to_json(self, path: Path | str) -> None:
        """Write this record as pretty-printed JSON to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> "ResultRecord":
        """Load a ResultRecord from a JSON file written by to_json()."""
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "CoRECT_RC_metrics" in data and "corect_standard_metrics" not in data:
            data["corect_standard_metrics"] = data.pop("CoRECT_RC_metrics")
        return cls(**data)
