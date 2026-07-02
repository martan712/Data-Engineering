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

from bondmaxsim.data.packing import build_qcum, pack_corpus
from bondmaxsim.kernels.bindings import (
    load_per_document_oracle,
    run_accounting,
    run_throughput,
)
from bondmaxsim.oracle.agreement import exact_agreement
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_topk
from bondmaxsim.ordering.orders import ada_order, bond_order, natural_order
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
    """

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

        # Kernel library — loaded lazily.
        self._lib = None

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
