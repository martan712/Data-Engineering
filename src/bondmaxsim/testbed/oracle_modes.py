"""Per-document-oracle kernel accounting/throughput mode implementations.

Single responsibility: run the per-document-oracle kernel
(cpp/per_document_oracle/) in accounting mode (exp-09-style live-set-only
cells counter) or throughput mode (exp-10-style dense-warmup wall-clock
kernel) over a query set, and assemble the resulting ResultRecord.

Ported artifact: benchmark loop pattern from
  archive/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py (accounting) and
  archive/preliminaries/10_maxsim_pruning_opt/maxsim_pruning_bench.py (throughput).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §6.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

from bondmaxsim.data.packing import build_qcum
from bondmaxsim.kernels.per_document import (
    run_accounting_validated,
    run_throughput_validated,
)
from bondmaxsim.oracle.agreement import (
    aggregate_agreement_results,
    validate_boundary_tie_equivalence,
)
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores
from bondmaxsim.schema import ResultRecord
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.packing_cache import PackingCache


def run_accounting_mode(
    lib: ctypes.CDLL,
    packing: PackingCache,
    queries: list[np.ndarray],
    config: RunConfig,
) -> ResultRecord:
    """Run accounting-mode kernel: measure true cells scanned and pruning rates.

    Does NOT measure wall-clock; ms_per_query and qps will be None.

    Returns
    -------
    ResultRecord with cells_scanned_pct, pruned_docs_pct,
    tokens_pruned_pct, bound_checks_per_query, recall_vs_exact_at_10
    populated; ms_per_query / qps = None.
    """
    K      = config.k
    D      = packing.D
    T      = packing.T
    n_docs = packing.num_docs

    cells_list: list[float]   = []
    dp_list:    list[float]   = []
    tp_list:    list[float]   = []
    recall_list: list[float]  = []
    agreement_results = []

    for query in queries:
        corpus, Q_eff, order = packing.dispatch_order_corpus(
            query, config.dimension_order
        )
        m    = Q_eff.shape[0]
        Qcum = build_qcum(Q_eff, order)

        ids, _scores, stats = run_accounting_validated(
            lib, corpus, Q_eff, order, Qcum,
            shrink=config.shrink, K=K,
        )

        # Exact oracle uses original (un-rotated) query + token-major flat.
        exact_full_scores = exact_maxsim_scores(
            query, packing.flat_tokens, packing.doc_starts
        )
        exact_ids, exact_scores = topk_from_scores(exact_full_scores, K)
        agreement = validate_boundary_tie_equivalence(
            ids.astype(np.int64), exact_ids, exact_scores,
            k=K, num_documents=n_docs, exact_scores_by_id=exact_full_scores,
        )
        agreement_results.append(agreement)
        recall_list.append(agreement.recall_vs_oracle_set)

        total_cells = int(T) * D * m   # brute-force denominator
        cells_list.append(float(stats[0]) / total_cells if total_cells > 0 else 0.0)
        dp_list.append(float(stats[1]) / n_docs if n_docs > 0 else 0.0)
        tp_list.append(float(stats[2]) / T if T > 0 else 0.0)

    cells_scanned_pct   = float(np.mean(cells_list))      * 100.0
    pruned_docs_pct     = float(np.mean(dp_list))         * 100.0
    tokens_pruned_pct   = float(np.mean(tp_list))         * 100.0
    recall_vs_exact     = float(np.mean(recall_list))
    agreement_summary   = aggregate_agreement_results(agreement_results)

    return ResultRecord(
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
        corect_standard_metrics=None,
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
        **agreement_summary,
    )


def run_throughput_mode(
    lib: ctypes.CDLL,
    packing: PackingCache,
    queries: list[np.ndarray],
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
    K      = config.k
    n_docs = packing.num_docs
    nq     = len(queries)

    # Pre-build order and Qcum for every query so timing is kernel-only.
    prepared: list[tuple] = []
    for query in queries:
        corpus, Q_eff, order = packing.dispatch_order_corpus(
            query, config.dimension_order
        )
        Qcum = build_qcum(Q_eff, order)
        prepared.append((corpus, Q_eff, order, Qcum))

    def _run_all():
        for corpus, Q_eff, order, Qcum in prepared:
            run_throughput_validated(
                lib, corpus, Q_eff, order, Qcum,
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
    agreement_results = []
    for query, (corpus, Q_eff, order, Qcum) in zip(queries, prepared):
        ids, _scores, _stats = run_throughput_validated(
            lib, corpus, Q_eff, order, Qcum,
            shrink=config.shrink, K=K,
        )
        exact_full_scores = exact_maxsim_scores(
            query, packing.flat_tokens, packing.doc_starts
        )
        exact_ids, exact_scores = topk_from_scores(exact_full_scores, K)
        agreement = validate_boundary_tie_equivalence(
            ids.astype(np.int64), exact_ids, exact_scores,
            k=K, num_documents=n_docs, exact_scores_by_id=exact_full_scores,
        )
        agreement_results.append(agreement)
        recall_list.append(agreement.recall_vs_oracle_set)

    recall_vs_exact = float(np.mean(recall_list))
    agreement_summary = aggregate_agreement_results(agreement_results)

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
        corect_standard_metrics=None,
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
        **agreement_summary,
    )
