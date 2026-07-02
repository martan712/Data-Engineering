"""Mechanism testbed runner: accounting and throughput modes.

Single responsibility: the Runner class wires packed embeddings + config into a
reproducible experiment run, calls either the accounting-mode or throughput-mode
kernel, collects ResultRecord-compatible outputs, and optionally writes them to
results/json/.

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

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from bondmaxsim.data.packing import build_qcum, pack_corpus, pack_corpus_wide
from bondmaxsim.kernels.bindings import (
    load_per_document_oracle,
    load_wide_block_kernel,
    run_accounting,
    run_throughput,
    run_wide_block_accounting,
    run_wide_block_throughput,
)
from bondmaxsim.oracle.agreement import exact_agreement
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, exact_maxsim_topk
from bondmaxsim.ordering.orders import ada_order, bond_order, natural_order
from bondmaxsim.schema import ResultRecord
from bondmaxsim.threshold.policies import oracle_threshold, seed_threshold


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
    candidate_budget: Optional[int] = None
    thread_count: int = 1
    machine: str = field(default_factory=platform.node)
    """Hostname for fair-comparison tracking (override for cross-machine runs)."""
    os: str = "linux"
    notes: Optional[str] = None


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

    # Cheap "first checkpoint" proxy for the seed threshold policy: partial
    # MaxSim scores computed from only the first few dimensions of the
    # per-query scan order (Stage 1 §4.4 "seeded threshold").
    _SEED_CHECKPOINT_DIMS = 32
    _SEED_FRACTION = 0.05

    # Safety margin subtracted from any tau_seed derived from a NumPy exact-
    # score computation (oracle/seed policies) before handing it to the
    # kernel, to absorb cross-implementation fp32 summation-order noise at
    # near-tie thresholds (see _resolve_tau_seed docstring).
    _TAU_SEED_EPS = 1e-3

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
        queries: list[np.ndarray],
    ) -> None:
        # Original token-major corpus for the exact oracle.
        self._flat_tokens = np.ascontiguousarray(flat_tokens, dtype=np.float32)
        self._doc_starts  = np.asarray(doc_starts, dtype=np.int64)
        self._queries     = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]

        T, D = self._flat_tokens.shape
        self._T       = T
        self._D       = D
        self._num_docs = len(self._doc_starts)

        # Reconstruct per-doc list for dim-major packing.
        docs: list[np.ndarray] = []
        for d in range(self._num_docs):
            start = int(self._doc_starts[d])
            end   = int(self._doc_starts[d + 1]) if d + 1 < self._num_docs else T
            docs.append(self._flat_tokens[start:end])
        self._docs = docs  # keep for ada rotated packing

        # Per-doc dim-major packed corpus for the kernel.
        self._flat, self._offs = pack_corpus(docs)

        # Corpus token mean for BOND order.
        self._mu = self._flat_tokens.mean(axis=0).astype(np.float32)

        # Ada rotation and rotated corpus — built lazily.
        self._R: Optional[np.ndarray] = None
        self._flat_rot: Optional[np.ndarray] = None
        self._offs_rot: Optional[np.ndarray] = None
        self._flat_tokens_rot: Optional[np.ndarray] = None  # token-major rotated corpus

        # Kernel library — loaded lazily.
        self._lib = None

        # Wide-block kernel: library + packings — loaded/built lazily.
        self._wide_lib = None
        self._wide: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self._wide_rot: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

        # Side channel populated by accounting_mode() for the wide-block path
        # (Stage 2 e02 hooks); see class docstring.
        self.last_block_doc_live: Optional[np.ndarray] = None
        self.last_block_token_live: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    def _get_lib(self):
        if self._lib is None:
            self._lib = load_per_document_oracle()
        return self._lib

    def _get_ada_rotation(self) -> np.ndarray:
        """Build a deterministic orthogonal rotation (QR of a seeded random matrix)."""
        if self._R is None:
            g = np.random.default_rng(123).standard_normal((self._D, self._D))
            Q_r, _ = np.linalg.qr(g)
            self._R = Q_r.astype(np.float32)
        return self._R

    def _get_flat_rot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (flat_rot, offs) for the rotated corpus (built lazily)."""
        if self._flat_rot is None:
            R = self._get_ada_rotation()
            rotated_docs = [(d @ R).astype(np.float32) for d in self._docs]
            self._flat_rot, self._offs_rot = pack_corpus(rotated_docs)
        return self._flat_rot, self._offs_rot

    def _get_flat_tokens_rot(self) -> np.ndarray:
        """Return the token-major rotated corpus [T, D] (built lazily).

        Orthogonal rotation preserves inner products and unit norm (Stage 1
        §4.5), so this is exact-safe at shrink=1; used both to build the
        wide-block ada packing and as the seed-policy checkpoint corpus.
        """
        if self._flat_tokens_rot is None:
            R = self._get_ada_rotation()
            self._flat_tokens_rot = (self._flat_tokens @ R).astype(np.float32)
        return self._flat_tokens_rot

    # ------------------------------------------------------------------
    # Wide-block kernel: lazy library + packing accessors
    # ------------------------------------------------------------------

    def _get_wide_lib(self):
        if self._wide_lib is None:
            self._wide_lib = load_wide_block_kernel()
        return self._wide_lib

    def _get_wide_packing(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Wide packing of the ORIGINAL (un-rotated) corpus (natural/bond orders)."""
        if self._wide is None:
            self._wide = pack_corpus_wide(self._flat_tokens, self._doc_starts)
        return self._wide

    def _get_wide_packing_rot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Wide packing of the ada-ROTATED corpus."""
        if self._wide_rot is None:
            flat_rot = self._get_flat_tokens_rot()
            self._wide_rot = pack_corpus_wide(flat_rot, self._doc_starts)
        return self._wide_rot

    def _dispatch_order_wide(
        self,
        query: np.ndarray,
        dimension_order: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (group_data, group_offsets, doc_offsets, group_doc_starts,
        Q_eff, order_u32) for the given order, mirroring _dispatch_order but
        for the wide-block packing."""
        if dimension_order == "natural":
            order = natural_order(query).astype(np.uint32)
            group_data, group_offsets, doc_offsets, group_doc_starts = self._get_wide_packing()
            return group_data, group_offsets, doc_offsets, group_doc_starts, query, order

        elif dimension_order == "bond":
            order = bond_order(query, self._mu).astype(np.uint32)
            group_data, group_offsets, doc_offsets, group_doc_starts = self._get_wide_packing()
            return group_data, group_offsets, doc_offsets, group_doc_starts, query, order

        elif dimension_order == "ada":
            R = self._get_ada_rotation()
            group_data, group_offsets, doc_offsets, group_doc_starts = self._get_wide_packing_rot()
            q_rot, order = ada_order(query, R)
            order = order.astype(np.uint32)
            return group_data, group_offsets, doc_offsets, group_doc_starts, q_rot, order

        else:
            raise ValueError(
                f"Unknown dimension_order: {dimension_order!r}. "
                "Expected one of 'natural', 'bond', 'ada'."
            )

    def _resolve_tau_seed(
        self,
        config: "RunConfig",
        query: np.ndarray,
        order: np.ndarray,
        dimension_order: str,
    ) -> float:
        """Resolve the wide-block kernel's tau_seed from RunConfig.threshold_policy.

        Stage 1 §4.4: "self_bound"/"exact_safe_topk" -> -inf (self-bounded;
        relies entirely on staged finalization across groups).  "oracle" ->
        the true k-th best exact score (upper bound on pruning potential;
        rotation-invariant, computed on the original un-rotated corpus).
        "seed" -> bondmaxsim.threshold.policies.seed_threshold using a cheap
        partial-score proxy: exact MaxSim restricted to the first
        _SEED_CHECKPOINT_DIMS dimensions of THIS query's scan order (in the
        same effective space -- rotated for order="ada" -- the kernel itself
        will see first), consistent with the policy's "first checkpoint" intent.

        Both "oracle" and "seed" derive tau from a NumPy exact-score
        computation whose fp32 summation order differs from the kernel's own
        incremental per-dimension-block accumulation (Stage 1 §4.2: fp
        exactness is empirical, not proven).  The theorem only requires tau
        to be a valid *lower* bound on the true k-th best score; a tau set
        to EXACTLY that score is a hairline tie that a cross-implementation
        fp32 mismatch can occasionally break (a document whose kernel-
        internal score computes a few ULPs below the NumPy tau gets
        spuriously pruned).  Subtracting a small safety margin keeps tau a
        strict lower bound in practice without materially loosening pruning.
        """
        policy = config.threshold_policy
        if policy in ("self_bound", "exact_safe_topk"):
            return float("-inf")

        exact_scores = exact_maxsim_scores(query, self._flat_tokens, self._doc_starts)

        if policy == "oracle":
            return oracle_threshold(exact_scores, config.k) - self._TAU_SEED_EPS

        if policy == "seed":
            if dimension_order == "ada":
                flat_eff_tm = self._get_flat_tokens_rot()
                R = self._get_ada_rotation()
                q_eff = (query @ R).astype(np.float32)
            else:
                flat_eff_tm = self._flat_tokens
                q_eff = query
            n_dims = min(self._D, self._SEED_CHECKPOINT_DIMS)
            dims = order[:n_dims]
            partial_scores = exact_maxsim_scores(
                q_eff[:, dims], flat_eff_tm[:, dims], self._doc_starts
            )
            return seed_threshold(
                partial_scores, exact_scores, config.k, seed_fraction=self._SEED_FRACTION
            ) - self._TAU_SEED_EPS

        raise ValueError(
            f"Unknown threshold_policy for wide_block kernel: {policy!r}. "
            "Expected one of 'self_bound', 'exact_safe_topk', 'oracle', 'seed'."
        )

    # ------------------------------------------------------------------
    # Order dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch_order(
        self,
        query: np.ndarray,
        dimension_order: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (flat_eff, offs_eff, Q_eff, order_u32) for the given order."""
        if dimension_order == "natural":
            order = natural_order(query).astype(np.uint32)
            return self._flat, self._offs, query, order

        elif dimension_order == "bond":
            order = bond_order(query, self._mu).astype(np.uint32)
            return self._flat, self._offs, query, order

        elif dimension_order == "ada":
            R = self._get_ada_rotation()
            flat_rot, offs_rot = self._get_flat_rot()
            q_rot, order = ada_order(query, R)
            order = order.astype(np.uint32)
            return flat_rot, offs_rot, q_rot, order

        else:
            raise ValueError(
                f"Unknown dimension_order: {dimension_order!r}. "
                "Expected one of 'natural', 'bond', 'ada'."
            )

    # ------------------------------------------------------------------
    # accounting_mode
    # ------------------------------------------------------------------

    def accounting_mode(self, config: RunConfig) -> ResultRecord:
        """Run accounting-mode kernel: measure true cells scanned and pruning rates.

        Uses the per-document-oracle kernel (cpp/per_document_oracle/) in
        accounting mode (exp-09 style: scans only the live set from dimension 0).
        Does NOT measure wall-clock; ms_per_query and qps will be None.

        Returns
        -------
        ResultRecord with cells_scanned_pct, pruned_docs_pct,
        tokens_pruned_pct, bound_checks_per_query, recall_vs_exact_at_10
        populated; ms_per_query / qps = None.
        """
        if config.method == "wide_block_maxsim_bond":
            return self._wide_accounting(config)

        lib    = self._get_lib()
        K      = config.k
        D      = self._D
        T      = self._T
        n_docs = self._num_docs

        cells_list: list[float]   = []
        dp_list:    list[float]   = []
        tp_list:    list[float]   = []
        recall_list: list[float]  = []

        for query in self._queries:
            flat_eff, offs_eff, Q_eff, order = self._dispatch_order(
                query, config.dimension_order
            )
            m    = Q_eff.shape[0]
            Qcum = build_qcum(Q_eff, order)

            ids, _scores, stats = run_accounting(
                lib, flat_eff, offs_eff, Q_eff, order, Qcum,
                shrink=config.shrink, K=K,
            )

            # Exact oracle uses original (un-rotated) query + token-major flat.
            exact_ids, _ = exact_maxsim_topk(
                query, self._flat_tokens, self._doc_starts, k=K
            )

            recall_list.append(exact_agreement(ids.astype(np.int64), exact_ids))

            total_cells = int(T) * D * m   # brute-force denominator
            cells_list.append(float(stats[0]) / total_cells if total_cells > 0 else 0.0)
            dp_list.append(float(stats[1]) / n_docs if n_docs > 0 else 0.0)
            tp_list.append(float(stats[2]) / T if T > 0 else 0.0)

        cells_scanned_pct   = float(np.mean(cells_list))      * 100.0
        pruned_docs_pct     = float(np.mean(dp_list))         * 100.0
        tokens_pruned_pct   = float(np.mean(tp_list))         * 100.0
        recall_vs_exact     = float(np.mean(recall_list))

        return ResultRecord(
            dataset               = config.dataset,
            num_docs              = n_docs,
            num_queries           = len(self._queries),
            method                = config.method,
            candidate_budget      = config.candidate_budget,
            dimension_order       = config.dimension_order,
            threshold_policy      = config.threshold_policy,
            recall_vs_exact_at_10 = recall_vs_exact,
            nDCG_at_10            = None,
            recall_at_100         = None,
            MRR_at_10             = None,
            CoRECT_RC_metrics     = None,
            ms_per_query          = None,
            qps                   = None,
            cells_scanned_pct     = cells_scanned_pct,
            pruned_docs_pct       = pruned_docs_pct,
            bound_checks_per_query= n_docs,   # every doc gets ≥1 UB check
            machine               = config.machine,
            os                    = config.os,
            thread_count          = config.thread_count,
            shrink                = config.shrink,
            tokens_pruned_pct     = tokens_pruned_pct,
            notes                 = config.notes,
        )

    # ------------------------------------------------------------------
    # throughput_mode
    # ------------------------------------------------------------------

    def throughput_mode(
        self,
        config: RunConfig,
        n_repeats: int = 5,
    ) -> ResultRecord:
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
        if config.method == "wide_block_maxsim_bond":
            return self._wide_throughput(config, n_repeats=n_repeats)

        lib    = self._get_lib()
        K      = config.k
        n_docs = self._num_docs
        nq     = len(self._queries)

        # Pre-build order and Qcum for every query so timing is kernel-only.
        prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for query in self._queries:
            flat_eff, offs_eff, Q_eff, order = self._dispatch_order(
                query, config.dimension_order
            )
            Qcum = build_qcum(Q_eff, order)
            prepared.append((flat_eff, offs_eff, Q_eff, order, Qcum))

        def _run_all():
            for flat_eff, offs_eff, Q_eff, order, Qcum in prepared:
                run_throughput(
                    lib, flat_eff, offs_eff, Q_eff, order, Qcum,
                    shrink=config.shrink, K=K,
                )

        # Warmup pass (not timed).
        _run_all()

        # Best-of-repeats timing.
        best_s = float("inf")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _run_all()
            best_s = min(best_s, time.perf_counter() - t0)

        ms_per_query = best_s / nq * 1e3
        qps          = nq / best_s if best_s > 0.0 else float("inf")

        # Recall check (informational; uses last pass's ids — re-run once to get them).
        recall_list: list[float] = []
        for query, (flat_eff, offs_eff, Q_eff, order, Qcum) in zip(self._queries, prepared):
            ids, _scores, _stats = run_throughput(
                lib, flat_eff, offs_eff, Q_eff, order, Qcum,
                shrink=config.shrink, K=K,
            )
            exact_ids, _ = exact_maxsim_topk(
                query, self._flat_tokens, self._doc_starts, k=K
            )
            recall_list.append(exact_agreement(ids.astype(np.int64), exact_ids))

        recall_vs_exact = float(np.mean(recall_list))

        return ResultRecord(
            dataset               = config.dataset,
            num_docs              = n_docs,
            num_queries           = nq,
            method                = config.method,
            candidate_budget      = config.candidate_budget,
            dimension_order       = config.dimension_order,
            threshold_policy      = config.threshold_policy,
            recall_vs_exact_at_10 = recall_vs_exact,
            nDCG_at_10            = None,
            recall_at_100         = None,
            MRR_at_10             = None,
            CoRECT_RC_metrics     = None,
            ms_per_query          = ms_per_query,
            qps                   = qps,
            cells_scanned_pct     = None,   # not meaningful in throughput mode
            pruned_docs_pct       = None,
            bound_checks_per_query= None,
            machine               = config.machine,
            os                    = config.os,
            thread_count          = config.thread_count,
            shrink                = config.shrink,
            tokens_pruned_pct     = None,   # accounting-only metric
            notes                 = config.notes,
        )

    # ------------------------------------------------------------------
    # wide_block_maxsim_bond dispatch targets (Stage 2 deliverable)
    # ------------------------------------------------------------------

    def _wide_accounting(self, config: RunConfig) -> ResultRecord:
        """accounting_mode() dispatch target for method="wide_block_maxsim_bond".

        Same metric semantics as the oracle's accounting_mode (Stage 1 §6);
        additionally populates self.last_block_doc_live /
        self.last_block_token_live (mean-over-queries e02 curves, see class
        docstring) -- these are NOT part of ResultRecord.
        """
        lib    = self._get_wide_lib()
        K      = config.k
        D      = self._D
        T      = self._T
        n_docs = self._num_docs

        cells_list: list[float]  = []
        dp_list:    list[float]  = []
        tp_list:    list[float]  = []
        recall_list: list[float] = []
        block_doc_live_sum:   Optional[np.ndarray] = None
        block_token_live_sum: Optional[np.ndarray] = None

        for query in self._queries:
            group_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order = (
                self._dispatch_order_wide(query, config.dimension_order)
            )
            m    = Q_eff.shape[0]
            Qcum = build_qcum(Q_eff, order)
            tau_seed = self._resolve_tau_seed(config, query, order, config.dimension_order)

            ids, _scores, stats, bdl, btl = run_wide_block_accounting(
                lib, group_data, group_offsets, doc_offsets, group_doc_starts,
                Q_eff, order, Qcum, shrink=config.shrink, tau_seed=tau_seed, K=K,
                collect_block_stats=True,
            )

            # Exact oracle uses original (un-rotated) query + token-major flat.
            exact_ids, _ = exact_maxsim_topk(
                query, self._flat_tokens, self._doc_starts, k=K
            )
            recall_list.append(exact_agreement(ids.astype(np.int64), exact_ids))

            total_cells = int(T) * D * m   # brute-force denominator
            cells_list.append(float(stats[0]) / total_cells if total_cells > 0 else 0.0)
            dp_list.append(float(stats[1]) / n_docs if n_docs > 0 else 0.0)
            tp_list.append(float(stats[2]) / T if T > 0 else 0.0)

            if bdl is not None:
                if block_doc_live_sum is None:
                    block_doc_live_sum   = bdl.astype(np.float64)
                    block_token_live_sum = btl.astype(np.float64)
                else:
                    block_doc_live_sum   = block_doc_live_sum + bdl
                    block_token_live_sum = block_token_live_sum + btl

        nq = len(self._queries)
        self.last_block_doc_live = (
            block_doc_live_sum / nq if block_doc_live_sum is not None else None
        )
        self.last_block_token_live = (
            block_token_live_sum / nq if block_token_live_sum is not None else None
        )

        cells_scanned_pct   = float(np.mean(cells_list))      * 100.0
        pruned_docs_pct     = float(np.mean(dp_list))         * 100.0
        tokens_pruned_pct   = float(np.mean(tp_list))         * 100.0
        recall_vs_exact     = float(np.mean(recall_list))

        return ResultRecord(
            dataset               = config.dataset,
            num_docs              = n_docs,
            num_queries           = len(self._queries),
            method                = config.method,
            candidate_budget      = config.candidate_budget,
            dimension_order       = config.dimension_order,
            threshold_policy      = config.threshold_policy,
            recall_vs_exact_at_10 = recall_vs_exact,
            nDCG_at_10            = None,
            recall_at_100         = None,
            MRR_at_10             = None,
            CoRECT_RC_metrics     = None,
            ms_per_query          = None,
            qps                   = None,
            cells_scanned_pct     = cells_scanned_pct,
            pruned_docs_pct       = pruned_docs_pct,
            bound_checks_per_query= n_docs,   # every doc gets >=1 UB check
            machine               = config.machine,
            os                    = config.os,
            thread_count          = config.thread_count,
            shrink                = config.shrink,
            tokens_pruned_pct     = tokens_pruned_pct,
            notes                 = config.notes,
        )

    def _wide_throughput(self, config: RunConfig, n_repeats: int = 5) -> ResultRecord:
        """throughput_mode() dispatch target for method="wide_block_maxsim_bond"."""
        lib    = self._get_wide_lib()
        K      = config.k
        n_docs = self._num_docs
        nq     = len(self._queries)

        # Pre-build order, Qcum, and tau_seed for every query so timing is kernel-only.
        prepared: list[tuple[np.ndarray, ...]] = []
        for query in self._queries:
            group_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order = (
                self._dispatch_order_wide(query, config.dimension_order)
            )
            Qcum = build_qcum(Q_eff, order)
            tau_seed = self._resolve_tau_seed(config, query, order, config.dimension_order)
            prepared.append(
                (group_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order, Qcum, tau_seed)
            )

        def _run_all():
            for group_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order, Qcum, tau_seed in prepared:
                run_wide_block_throughput(
                    lib, group_data, group_offsets, doc_offsets, group_doc_starts,
                    Q_eff, order, Qcum, shrink=config.shrink, tau_seed=tau_seed, K=K,
                )

        # Warmup pass (not timed).
        _run_all()

        # Best-of-repeats timing.
        best_s = float("inf")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _run_all()
            best_s = min(best_s, time.perf_counter() - t0)

        ms_per_query = best_s / nq * 1e3
        qps          = nq / best_s if best_s > 0.0 else float("inf")

        # Recall check (informational; uses a fresh pass to get ids).
        recall_list: list[float] = []
        for query, prep in zip(self._queries, prepared):
            group_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order, Qcum, tau_seed = prep
            ids, _scores, _stats, _bdl, _btl = run_wide_block_throughput(
                lib, group_data, group_offsets, doc_offsets, group_doc_starts,
                Q_eff, order, Qcum, shrink=config.shrink, tau_seed=tau_seed, K=K,
            )
            exact_ids, _ = exact_maxsim_topk(
                query, self._flat_tokens, self._doc_starts, k=K
            )
            recall_list.append(exact_agreement(ids.astype(np.int64), exact_ids))

        recall_vs_exact = float(np.mean(recall_list))

        return ResultRecord(
            dataset               = config.dataset,
            num_docs              = n_docs,
            num_queries           = nq,
            method                = config.method,
            candidate_budget      = config.candidate_budget,
            dimension_order       = config.dimension_order,
            threshold_policy      = config.threshold_policy,
            recall_vs_exact_at_10 = recall_vs_exact,
            nDCG_at_10            = None,
            recall_at_100         = None,
            MRR_at_10             = None,
            CoRECT_RC_metrics     = None,
            ms_per_query          = ms_per_query,
            qps                   = qps,
            cells_scanned_pct     = None,   # not meaningful in throughput mode
            pruned_docs_pct       = None,
            bound_checks_per_query= None,
            machine               = config.machine,
            os                    = config.os,
            thread_count          = config.thread_count,
            shrink                = config.shrink,
            tokens_pruned_pct     = None,   # accounting-only metric
            notes                 = config.notes,
        )
