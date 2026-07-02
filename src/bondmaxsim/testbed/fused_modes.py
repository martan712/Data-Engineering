"""Fused-panel kernel testbed mode (Stage 3b K5).

Single responsibility: wire the PackingCache + RunConfig into the fused panel
MaxSim brute kernel (cpp/fused_panel_maxsim/) and return a ResultRecord,
mirroring bondmaxsim.testbed.wide_modes.run_wide_brute_mode.

Design and rationale: docs/stage3b_fused_panel_maxsim_kernel.md §5–6.  This is
a wall-clock instrument only (dense scan, no pruning, no accounting stats).
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

from bondmaxsim.kernels.fused_panel import run_fused_panel_brute
from bondmaxsim.oracle.agreement import exact_agreement
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_topk
from bondmaxsim.schema import ResultRecord
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.packing_cache import PackingCache


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
