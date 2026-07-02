"""Wide-block MaxSim BOND kernel accounting/throughput mode implementations
(Stage 2 deliverable).

Single responsibility: run the wide-block kernel (cpp/wide_block_maxsim_bond/)
in accounting mode or throughput mode over a query set, resolving each
query's tau_seed from RunConfig.threshold_policy (bondmaxsim.testbed.thresholds),
and assemble the resulting ResultRecord.  Accounting mode additionally
returns the per-fetch-boundary live-doc/live-token side-channel curves
(Stage 2 e02 hooks; mean over queries) for the caller to expose.

Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.3, §6.
"""

from __future__ import annotations

import ctypes
import time
from typing import Optional

import numpy as np

from bondmaxsim.data.packing import build_qcum
from bondmaxsim.kernels.wide_block import run_wide_block_accounting, run_wide_block_throughput
from bondmaxsim.oracle.agreement import exact_agreement
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_topk
from bondmaxsim.schema import ResultRecord
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.packing_cache import PackingCache
from bondmaxsim.testbed.thresholds import resolve_tau_seed


def run_wide_accounting_mode(
    lib: ctypes.CDLL,
    packing: PackingCache,
    queries: list[np.ndarray],
    config: RunConfig,
) -> tuple[ResultRecord, Optional[np.ndarray], Optional[np.ndarray]]:
    """accounting_mode() dispatch target for method="wide_block_maxsim_bond".

    Same metric semantics as the oracle's accounting mode (Stage 1 §6).

    Returns
    -------
    (record, mean_block_doc_live, mean_block_token_live) — the latter two are
    mean-over-queries e02 curves (None if the kernel does not populate them);
    NOT part of ResultRecord itself.
    """
    K      = config.k
    D      = packing.D
    T      = packing.T
    n_docs = packing.num_docs

    cells_list: list[float]  = []
    dp_list:    list[float]  = []
    tp_list:    list[float]  = []
    recall_list: list[float] = []
    block_doc_live_sum:   Optional[np.ndarray] = None
    block_token_live_sum: Optional[np.ndarray] = None

    for query in queries:
        group_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order = (
            packing.dispatch_order_wide(query, config.dimension_order)
        )
        m    = Q_eff.shape[0]
        Qcum = build_qcum(Q_eff, order)
        tau_seed = resolve_tau_seed(config, query, order, config.dimension_order, packing)

        ids, _scores, stats, bdl, btl = run_wide_block_accounting(
            lib, group_data, group_offsets, doc_offsets, group_doc_starts,
            Q_eff, order, Qcum, shrink=config.shrink, tau_seed=tau_seed, K=K,
            collect_block_stats=True,
        )

        # Exact oracle uses original (un-rotated) query + token-major flat.
        exact_ids, _ = exact_maxsim_topk(
            query, packing.flat_tokens, packing.doc_starts, k=K
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

    nq = len(queries)
    mean_block_doc_live = (
        block_doc_live_sum / nq if block_doc_live_sum is not None else None
    )
    mean_block_token_live = (
        block_token_live_sum / nq if block_token_live_sum is not None else None
    )

    cells_scanned_pct   = float(np.mean(cells_list))      * 100.0
    pruned_docs_pct     = float(np.mean(dp_list))         * 100.0
    tokens_pruned_pct   = float(np.mean(tp_list))         * 100.0
    recall_vs_exact     = float(np.mean(recall_list))

    record = ResultRecord(
        dataset               = config.dataset,
        num_docs              = n_docs,
        num_queries           = len(queries),
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
    return record, mean_block_doc_live, mean_block_token_live


def run_wide_throughput_mode(
    lib: ctypes.CDLL,
    packing: PackingCache,
    queries: list[np.ndarray],
    config: RunConfig,
    n_repeats: int = 5,
) -> ResultRecord:
    """throughput_mode() dispatch target for method="wide_block_maxsim_bond"."""
    K      = config.k
    n_docs = packing.num_docs
    nq     = len(queries)

    # Pre-build order, Qcum, and tau_seed for every query so timing is kernel-only.
    prepared: list[tuple] = []
    for query in queries:
        group_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order = (
            packing.dispatch_order_wide(query, config.dimension_order)
        )
        Qcum = build_qcum(Q_eff, order)
        tau_seed = resolve_tau_seed(config, query, order, config.dimension_order, packing)
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
    for query, prep in zip(queries, prepared):
        group_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order, Qcum, tau_seed = prep
        ids, _scores, _stats, _bdl, _btl = run_wide_block_throughput(
            lib, group_data, group_offsets, doc_offsets, group_doc_starts,
            Q_eff, order, Qcum, shrink=config.shrink, tau_seed=tau_seed, K=K,
        )
        exact_ids, _ = exact_maxsim_topk(
            query, packing.flat_tokens, packing.doc_starts, k=K
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
