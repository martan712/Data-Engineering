"""Mechanism testbed runner: accounting and throughput modes.

Single responsibility: the Runner class wires packed embeddings + config into a
reproducible experiment run, calls either the accounting-mode or throughput-mode
kernel, collects ResultRecord-compatible outputs, and optionally writes them to
results/json/.

Ported artifact: benchmark loop pattern from
  research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py (accounting) and
  research/preliminaries/10_maxsim_pruning_opt/maxsim_pruning_bench.py (throughput);
  result collection from archive/reference/05_maxsim_bond_instrumentation.py.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §6 (accounting mode
  must use exp-09 counting; throughput mode must use exp-10 dense-warmup; they
  must never be conflated), §8 items 2–3 (exact-agreement check inside Runner at
  shrink=1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from bondmaxsim.schema import ResultRecord


@dataclass
class RunConfig:
    """Configuration for a single Runner experiment run."""

    dataset: str
    method: str
    dimension_order: str
    threshold_policy: str
    k: int = 10
    shrink: float = 1.0
    """Recall knob: 1.0 = exact-safe (shrink=1), <1.0 = approximate."""
    candidate_budget: int | None = None
    thread_count: int = 1
    machine: str = "unknown"
    os: str = "linux"
    notes: str | None = None


class Runner:
    """Mechanism testbed runner.

    Parameters
    ----------
    flat_tokens : float32 [total_tokens, D] — all document tokens, dim-major packed
    doc_starts  : int64  [num_docs]          — start offset of each document
    queries     : list of float32 [m_q, D]  — query token matrices

    Methods
    -------
    accounting_mode : run with the exp-09-style live-set-only cells counter
    throughput_mode : run with the exp-10-style dense-warmup wall-clock kernel
    """

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
        queries: list[np.ndarray],
    ) -> None:
        raise NotImplementedError

    def accounting_mode(self, config: RunConfig) -> ResultRecord:
        """Run accounting-mode kernel: measure true cells scanned and pruning rates.

        Uses the per-document-oracle kernel (cpp/per_document_oracle/) in
        accounting mode (exp-09 style: scans only the live set from dimension 0).
        Does NOT measure wall-clock; ms_per_query and qps will be None.

        Returns
        -------
        ResultRecord with cells_scanned_pct, pruned_docs_pct,
        bound_checks_per_query, recall_vs_exact_at_10 populated;
        ms_per_query / qps = None.
        """
        raise NotImplementedError

    def throughput_mode(self, config: RunConfig, n_repeats: int = 5) -> ResultRecord:
        """Run throughput-mode kernel: measure realistic wall-clock latency.

        Uses the per-document-oracle kernel in throughput mode (exp-10 style:
        dense warmup over first D/4 dims, then positional survivor scan).
        Reports min-of-repeats ms/query and QPS.  cells_scanned_pct is not the
        true algorithmic work in this mode (see Stage 1 §6).

        Returns
        -------
        ResultRecord with ms_per_query, qps populated;
        cells_scanned_pct = None (not meaningful in throughput mode).
        """
        raise NotImplementedError
