"""Fused-panel kernel testbed modes (Stage 3b K4/K5).

Single responsibility: wire the PackingCache + RunConfig into the fused panel
MaxSim kernels (cpp/fused_panel_maxsim/) and return ResultRecords, mirroring
bondmaxsim.testbed.wide_modes:
  run_fused_brute_mode — dense scan, the decision-gate baseline
  run_fused_bond_mode  — BOND document-checkpoint pruning on the same
                         microkernel (the wall-clock BOND instrument)

Design and rationale: docs/stage3b_fused_panel_maxsim_kernel.md §5–6.  These
are wall-clock instruments only; algorithmic-work accounting stays on the
wide-block accounting kernel.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

from bondmaxsim.data.packing import build_qcum
from bondmaxsim.kernels.fused_panel import run_fused_panel_bond, run_fused_panel_brute
from bondmaxsim.oracle.agreement import exact_agreement
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_topk
from bondmaxsim.schema import ResultRecord
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.packing_cache import PackingCache
from bondmaxsim.testbed.thresholds import resolve_tau_seed


def run_fused_brute_mode(
    lib: ctypes.CDLL,
    packing: PackingCache,
    queries: list[np.ndarray],
    config: RunConfig,
    n_threads: int = 1,
    n_repeats: int = 5,
) -> ResultRecord:
    """Fused panel-major brute force: register-tiled dense MaxSim scan.

    Same columnar-family layout as the BOND kernels but packed in 16-token
    dim-major panels (the BLAS packed-B format, Stage 3b §3.1) with the
    per-document max fused into the register tile (§3.2).  No bound checks,
    no pruning — the strongest dense baseline for the wall-clock decision
    gate.

    Parameters
    ----------
    n_threads : OpenMP threads over vectorgroups (1 = single-thread arm;
                <= 0 = OpenMP default, i.e. all cores).

    Returns a ResultRecord with ms_per_query / qps populated; all
    pruning/cells fields are None (not meaningful for brute force).
    """
    K      = config.k
    n_docs = packing.num_docs
    nq     = len(queries)

    panel_data, group_offsets, doc_offsets, group_doc_starts, _ = (
        packing._get_panel_packing()
    )

    Qs: list[np.ndarray] = [
        np.ascontiguousarray(q, dtype=np.float32) for q in queries
    ]

    def _run_all():
        for Q in Qs:
            run_fused_panel_brute(
                lib, panel_data, group_offsets, doc_offsets, group_doc_starts,
                Q, K, n_threads=n_threads,
            )

    # Warmup.
    _run_all()

    best_s = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _run_all()
        best_s = min(best_s, time.perf_counter() - t0)

    ms_per_query = best_s / nq * 1e3
    qps          = nq / best_s if best_s > 0.0 else float("inf")

    # Recall check on one pass (dense scan = exact; recall must be 1.0).
    recall_list: list[float] = []
    for Q in Qs:
        exact_ids, _ = exact_maxsim_topk(Q, packing.flat_tokens, packing.doc_starts, k=K)
        ids, _ = run_fused_panel_brute(
            lib, panel_data, group_offsets, doc_offsets, group_doc_starts,
            Q, K, n_threads=n_threads,
        )
        recall_list.append(exact_agreement(ids.astype(np.int64), exact_ids))

    return ResultRecord(
        dataset               = config.dataset,
        num_docs              = n_docs,
        num_queries           = nq,
        method                = "fused_panel_brute",
        candidate_budget      = None,
        dimension_order       = "natural",
        threshold_policy      = "none",
        recall_vs_exact_at_10 = float(np.mean(recall_list)),
        nDCG_at_10            = None,
        recall_at_100         = None,
        MRR_at_10             = None,
        CoRECT_RC_metrics     = None,
        ms_per_query          = ms_per_query,
        qps                   = qps,
        cells_scanned_pct     = None,
        pruned_docs_pct       = None,
        bound_checks_per_query= None,
        machine               = config.machine,
        os                    = config.os,
        thread_count          = n_threads if n_threads > 0 else None,
        shrink                = 1.0,
        tokens_pruned_pct     = None,
        notes                 = f"fused panel brute force, n_threads={n_threads}",
    )


def run_fused_bond_mode(
    lib: ctypes.CDLL,
    packing: PackingCache,
    queries: list[np.ndarray],
    config: RunConfig,
    n_threads: int = 1,
    n_repeats: int = 5,
    exact_ids_list: list[np.ndarray] | None = None,
    level: str = "doc",
) -> ResultRecord:
    """Fused panel BOND: dimension-incremental scan with bound checkpoints at
    dims {32, 64} on the register-tiled microkernel (Stage 3b §5.5).
    level="doc" prunes documents only; level="token" additionally applies the
    Stage 1 §2.4 token domination test (three-arm isolation instrument).
    The wall-clock BOND instrument — compare against run_fused_brute_mode at
    the SAME n_threads.

    Order/policy semantics match the wide-block kernel: dimension_order via
    dispatch_order_panel (natural/bond share the natural packing, pca uses
    the rotated packing), threshold_policy via resolve_tau_seed, shrink=1 is
    exact-safe.

    Returns a ResultRecord with ms_per_query / qps and pruned_docs_pct
    populated; cells_scanned_pct uses the PADDED-token wall-clock convention
    (not comparable to the accounting kernel's live-set counter — reported
    in notes, kept out of cells_scanned_pct to avoid conflation).
    """
    K      = config.k
    n_docs = packing.num_docs
    nq     = len(queries)

    # Pre-build all per-query inputs so timing is kernel-only.
    prepared = []
    for q in queries:
        panel_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order = (
            packing.dispatch_order_panel(q, config.dimension_order)
        )
        Qcum = build_qcum(Q_eff, order)
        tau  = resolve_tau_seed(config, q, order, config.dimension_order, packing)
        prepared.append(
            (panel_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order, Qcum, tau)
        )

    def _run_all():
        for panel_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order, Qcum, tau in prepared:
            run_fused_panel_bond(
                lib, panel_data, group_offsets, doc_offsets, group_doc_starts,
                Q_eff, order, Qcum,
                shrink=config.shrink, tau_seed=tau, K=K, n_threads=n_threads,
                level=level,
            )

    # Warmup.
    _run_all()

    best_s = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _run_all()
        best_s = min(best_s, time.perf_counter() - t0)

    ms_per_query = best_s / nq * 1e3
    qps          = nq / best_s if best_s > 0.0 else float("inf")

    # Recall + pruning stats on one pass.
    if exact_ids_list is None:
        exact_ids_list = [
            exact_maxsim_topk(q, packing.flat_tokens, packing.doc_starts, k=K)[0]
            for q in queries
        ]
    recall_list: list[float] = []
    docs_pruned_total = 0
    tokens_pruned_total = 0
    total_tokens_padded = int(prepared[0][2][-1]) if prepared else 0  # doc_offsets[-1]
    for prep, exact_ids in zip(prepared, exact_ids_list):
        panel_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order, Qcum, tau = prep
        ids, _, stats = run_fused_panel_bond(
            lib, panel_data, group_offsets, doc_offsets, group_doc_starts,
            Q_eff, order, Qcum,
            shrink=config.shrink, tau_seed=tau, K=K, n_threads=n_threads,
            level=level,
        )
        recall_list.append(exact_agreement(ids.astype(np.int64), exact_ids))
        docs_pruned_total += int(stats[1])
        tokens_pruned_total += int(stats[2])

    pruned_docs_pct = 100.0 * docs_pruned_total / (n_docs * nq) if n_docs * nq else 0.0
    tokens_pruned_pct = (100.0 * tokens_pruned_total / (total_tokens_padded * nq)
                         if level == "token" and total_tokens_padded * nq else None)

    return ResultRecord(
        dataset               = config.dataset,
        num_docs              = n_docs,
        num_queries           = nq,
        method                = f"fused_panel_bond_{level}",
        candidate_budget      = config.candidate_budget,
        dimension_order       = config.dimension_order,
        threshold_policy      = config.threshold_policy,
        recall_vs_exact_at_10 = float(np.mean(recall_list)),
        nDCG_at_10            = None,
        recall_at_100         = None,
        MRR_at_10             = None,
        CoRECT_RC_metrics     = None,
        ms_per_query          = ms_per_query,
        qps                   = qps,
        cells_scanned_pct     = None,
        pruned_docs_pct       = pruned_docs_pct,
        bound_checks_per_query= None,
        machine               = config.machine,
        os                    = config.os,
        thread_count          = n_threads if n_threads > 0 else None,
        shrink                = config.shrink,
        tokens_pruned_pct     = tokens_pruned_pct,
        notes                 = (f"fused panel BOND level={level} (checkpoints 32/64), "
                                 f"n_threads={n_threads}"),
    )
