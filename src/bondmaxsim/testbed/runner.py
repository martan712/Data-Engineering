"""Mechanism testbed runner: accounting and throughput modes.

Single responsibility: the Runner class wires packed embeddings + config into a
reproducible experiment run, dispatches to the per-document-oracle or
wide-block kernel's accounting/throughput mode implementation
(bondmaxsim.testbed.oracle_modes / bondmaxsim.testbed.wide_modes), and returns
the resulting ResultRecord.

Ported artifact: benchmark loop pattern from
  archive/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py (accounting) and
  archive/preliminaries/10_maxsim_pruning_opt/maxsim_pruning_bench.py (throughput);
  result collection from archive/reference/05_maxsim_bond_instrumentation.py.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §6 (accounting mode
  must use exp-09 counting; throughput mode must use exp-10 dense-warmup; they
  must never be conflated), §8 items 2–3 (exact-agreement check inside Runner at
  shrink=1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bondmaxsim.kernels.per_document import load_per_document_oracle
from bondmaxsim.kernels.wide_block import load_wide_block_kernel
from bondmaxsim.schema import ResultRecord
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.oracle_modes import run_accounting_mode, run_throughput_mode
from bondmaxsim.testbed.packing_cache import PackingCache
from bondmaxsim.testbed.wide_modes import run_wide_accounting_mode, run_wide_throughput_mode

__all__ = ["Runner", "RunConfig"]


class Runner:
    """Mechanism testbed runner.

    Parameters
    ----------
    flat_tokens : float32 [total_tokens, D] — all document tokens, row-major
    doc_starts  : int64  [num_docs]          — start token offset of each document
    queries     : list of float32 [m_q, D]  — query token matrices

    Methods
    -------
    accounting_mode : run with the exp-09-style live-set-only cells counter
    throughput_mode : run with the exp-10-style dense-warmup wall-clock kernel

    Dispatch on RunConfig.method
    ----------------------------
    "bond_pdx_maxsim_exact_safe" (or any other value): per-document oracle
      kernel (cpp/per_document_oracle/), unchanged Stage 1 behavior.
    "wide_block_maxsim_bond": Stage 2 wide-block kernel
      (cpp/wide_block_maxsim_bond/); the wide dim-major packing is built and
      cached lazily per dimension order.

    Side channel: per-fetch-boundary live-doc / live-token curves
    ---------------------------------------------------------------
    accounting_mode(), when method="wide_block_maxsim_bond", additionally
    populates ``self.last_block_doc_live`` / ``self.last_block_token_live``
    (float64 [len(DEFAULT_FETCH)], mean over queries of the kernel's
    block_doc_live / block_token_live outputs).  ResultRecord itself stays
    schema-compatible (no new fields); read these attributes right after the
    accounting_mode() call that produced them.  None for the oracle path.
    """

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
        queries: list[np.ndarray],
    ) -> None:
        self._packing = PackingCache(flat_tokens, doc_starts)
        self._queries = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]

        # Kernel libraries — loaded lazily.
        self._lib = None
        self._wide_lib = None

        # Side channel populated by accounting_mode() for the wide-block path
        # (Stage 2 e02 hooks); see class docstring.
        self.last_block_doc_live: Optional[np.ndarray] = None
        self.last_block_token_live: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy library accessors
    # ------------------------------------------------------------------

    def _get_lib(self):
        if self._lib is None:
            self._lib = load_per_document_oracle()
        return self._lib

    def _get_wide_lib(self):
        if self._wide_lib is None:
            self._wide_lib = load_wide_block_kernel()
        return self._wide_lib

    # ------------------------------------------------------------------
    # accounting_mode
    # ------------------------------------------------------------------

    def accounting_mode(self, config: RunConfig) -> ResultRecord:
        """Run accounting-mode kernel: measure true cells scanned and pruning rates.

        Does NOT measure wall-clock; ms_per_query and qps will be None.

        Returns
        -------
        ResultRecord with cells_scanned_pct, pruned_docs_pct,
        tokens_pruned_pct, bound_checks_per_query, recall_vs_exact_at_10
        populated; ms_per_query / qps = None.
        """
        if config.method == "wide_block_maxsim_bond":
            record, bdl, btl = run_wide_accounting_mode(
                self._get_wide_lib(), self._packing, self._queries, config
            )
            self.last_block_doc_live = bdl
            self.last_block_token_live = btl
            return record

        return run_accounting_mode(self._get_lib(), self._packing, self._queries, config)

    # ------------------------------------------------------------------
    # throughput_mode
    # ------------------------------------------------------------------

    def throughput_mode(
        self,
        config: RunConfig,
        n_repeats: int = 5,
    ) -> ResultRecord:
        """Run throughput-mode kernel: measure realistic wall-clock latency.

        Reports min-of-repeats ms/query and QPS.  cells_scanned_pct is not the
        true algorithmic work in this mode (see Stage 1 §6).

        Returns
        -------
        ResultRecord with ms_per_query, qps populated;
        cells_scanned_pct = None (not meaningful in throughput mode).
        """
        if config.method == "wide_block_maxsim_bond":
            return run_wide_throughput_mode(
                self._get_wide_lib(), self._packing, self._queries, config,
                n_repeats=n_repeats,
            )

        return run_throughput_mode(
            self._get_lib(), self._packing, self._queries, config,
            n_repeats=n_repeats,
        )
