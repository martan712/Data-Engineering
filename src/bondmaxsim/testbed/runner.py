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

import time
from typing import Optional

import numpy as np

from bondmaxsim.kernels.fused_panel import load_fused_panel_kernel
from bondmaxsim.kernels.per_document import load_per_document_oracle
from bondmaxsim.kernels.wide_block import load_wide_block_kernel
from bondmaxsim.schema import ResultRecord
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.fused_modes import run_fused_bond_mode, run_fused_brute_mode
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
        self._fused_lib = None

        # Exact top-k cache: keyed by k, stores (ids, scores) tuples computed
        # once and shared across all dimension-order calls on the same query
        # set (order doesn't affect the exact result).
        self._exact_oracle_cache: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._exact_full_scores_cache: Optional[list[np.ndarray]] = None

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

    def _get_fused_lib(self):
        if self._fused_lib is None:
            self._fused_lib = load_fused_panel_kernel()
        return self._fused_lib

    def _get_exact_oracle(self, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (ids, scores) pairs for every query's exact top-k, cached by k."""
        from bondmaxsim.oracle.exact_maxsim import topk_from_scores
        if k not in self._exact_oracle_cache:
            self._exact_oracle_cache[k] = [
                topk_from_scores(scores, k) for scores in self._get_exact_full_scores()
            ]
        return self._exact_oracle_cache[k]

    def _get_exact_full_scores(self) -> list[np.ndarray]:
        """Return independently exact scores for every document and query."""
        from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores
        if self._exact_full_scores_cache is None:
            self._exact_full_scores_cache = [
                exact_maxsim_scores(
                    query, self._packing.flat_tokens, self._packing.doc_starts
                )
                for query in self._queries
            ]
        return self._exact_full_scores_cache

    def _get_exact_ids(self, k: int) -> list[np.ndarray]:
        return [ids for ids, _ in self._get_exact_oracle(k)]

    def _get_exact_scores(self, k: int) -> list[np.ndarray]:
        return [scores for _, scores in self._get_exact_oracle(k)]

    # ------------------------------------------------------------------
    # brute_force_mode
    # ------------------------------------------------------------------

    def brute_force_mode(
        self,
        config: RunConfig,
        kind: str = "fused",
        n_repeats: int = 5,
        n_threads: int = 1,
    ) -> ResultRecord:
        """Run a brute-force baseline (no pruning) for comparison with BOND arms.

        Parameters
        ----------
        kind : "fused" — fused_panel_maxsim_brute (Stage 3b): register-tiled
                         panel-major dense scan with fused per-doc max;
                         ``n_threads`` OpenMP threads over groups.  The
                         decision-gate baseline.
               "numpy" — exact_maxsim_topk timed in a throughput loop: plain
                         row-major MatMul baseline via NumPy/BLAS.  BLAS thread
                         count pinned to ``n_threads`` via threadpoolctl
                         (<= 0 = leave OpenBLAS default, i.e. all cores).
        n_threads : thread count (<= 0 = library default, i.e. all cores).

        (The former "pdx" kind — the wide-block dense scan — was removed in
        Stage 3b; the fused kernel supersedes it as the layout baseline.)
        """
        if kind == "fused":
            return run_fused_brute_mode(
                self._get_fused_lib(), self._packing, self._queries, config,
                n_threads=n_threads, n_repeats=n_repeats,
            )
        elif kind == "numpy":
            return self._numpy_brute_force(config, n_repeats, n_threads)
        else:
            raise ValueError(
                f"Unknown brute_force kind: {kind!r}. Expected 'fused' or 'numpy'."
            )

    def _numpy_brute_force(self, config: RunConfig, n_repeats: int,
                           n_threads: int = 0) -> ResultRecord:
        """Time exact_maxsim_topk (row-major NumPy) in a best-of-N loop.

        n_threads > 0 pins the BLAS pool via threadpoolctl for the duration of
        the timed loops (Stage 3b: the threading factor must be an explicit
        arm, not an accident of the OpenBLAS default).
        """
        import contextlib

        from threadpoolctl import threadpool_limits

        from bondmaxsim.oracle.exact_maxsim import exact_maxsim_topk

        K  = config.k
        nq = len(self._queries)

        def _run_all():
            for q in self._queries:
                exact_maxsim_topk(q, self._packing.flat_tokens,
                                  self._packing.doc_starts, k=K)

        blas_limit = (
            threadpool_limits(limits=n_threads, user_api="blas")
            if n_threads > 0 else contextlib.nullcontext()
        )
        with blas_limit:
            # Warmup.
            _run_all()

            best_s = float("inf")
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                _run_all()
                best_s = min(best_s, time.perf_counter() - t0)

        ms_per_query = best_s / nq * 1e3
        qps          = nq / best_s if best_s > 0.0 else float("inf")

        return ResultRecord(
            dataset               = config.dataset,
            num_docs              = self._packing.num_docs,
            num_queries           = nq,
            method                = "numpy_brute",
            candidate_budget      = None,
            dimension_order       = "natural",
            threshold_policy      = "none",
            recall_vs_exact_at_10 = 1.0,
            nDCG_at_10            = None,
            recall_at_100         = None,
            MRR_at_10             = None,
            corect_standard_metrics=None,
            ms_per_query          = ms_per_query,
            qps                   = qps,
            cells_scanned_pct     = None,
            pruned_docs_pct       = None,
            bound_checks_per_query= None,
            machine               = config.machine,
            os                    = config.os,
            thread_count          = n_threads if n_threads > 0 else config.thread_count,
            shrink                = 1.0,
            tokens_pruned_pct     = None,
            strict_top_k_set_equal= True,
            boundary_tie_equivalent=True,
            agreement_failure_codes=[],
            notes                 = (
                "NumPy row-major brute force"
                + (f", BLAS threads={n_threads}" if n_threads > 0 else ", BLAS default threads")
            ),
        )

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
                self._get_wide_lib(), self._packing, self._queries, config,
                exact_ids_list=self._get_exact_ids(config.k),
                exact_scores_list=self._get_exact_scores(config.k),
                exact_full_scores_list=self._get_exact_full_scores(),
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
        n_threads: int = 1,
    ) -> ResultRecord:
        """Run throughput-mode kernel: measure realistic wall-clock latency.

        Reports min-of-repeats ms/query and QPS.  cells_scanned_pct is not the
        true algorithmic work in this mode (see Stage 1 §6).

        Dispatch: config.method == "fused_panel_maxsim_bond" runs the Stage 3b
        fused BOND kernel with document-level pruning;
        "fused_panel_maxsim_bond_cheap" runs the same document-level kernel
        with the query-only cheap bound (e09 instrument,
        docs/bond2002_bound_cost_analysis.md);
        "fused_panel_maxsim_bond_token" runs the same kernel with the token-
        level domination test added (three-arm isolation, Stage 3b §5.8);
        "wide_block_maxsim_bond" runs the legacy wide-block throughput kernel;
        anything else runs the per-document oracle.

        Returns
        -------
        ResultRecord with ms_per_query, qps populated;
        cells_scanned_pct = None (not meaningful in throughput mode).
        """
        if config.method in ("fused_panel_maxsim_bond", "fused_panel_maxsim_bond_cheap",
                             "fused_panel_maxsim_bond_token"):
            level = "token" if config.method.endswith("_token") else "doc"
            bound = "cheap" if config.method.endswith("_cheap") else "tight"
            return run_fused_bond_mode(
                self._get_fused_lib(), self._packing, self._queries, config,
                n_threads=n_threads, n_repeats=n_repeats,
                exact_ids_list=self._get_exact_ids(config.k),
                exact_scores_list=self._get_exact_scores(config.k),
                exact_full_scores_list=self._get_exact_full_scores(),
                level=level, bound=bound,
            )

        if config.method == "wide_block_maxsim_bond":
            return run_wide_throughput_mode(
                self._get_wide_lib(), self._packing, self._queries, config,
                n_repeats=n_repeats,
            )

        return run_throughput_mode(
            self._get_lib(), self._packing, self._queries, config,
            n_repeats=n_repeats,
        )
