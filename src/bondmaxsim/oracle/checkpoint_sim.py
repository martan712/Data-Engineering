"""NumPy checkpoint-accounting simulator for the fused doc-level BOND policy (R2).

Single responsibility: replicate, in exact NumPy arithmetic, the pruning
decisions the fused panel BOND kernel (cpp/fused_panel_maxsim/,
`fused_panel_maxsim_bond`) makes under a FIXED pruning threshold, and account
the cells/bytes those decisions skip.  This closes the instrument-alignment
gap named in the research plan (R2): the wide-block accounting kernel counts
the mechanism's upper envelope (per-boundary, token+document, breadth-first),
which does NOT predict what the doc-at-a-time checkpoint algorithm captures.
This simulator computes accounting numbers under the FUSED algorithm itself,
so its cells% predicts fused wall-clock savings.

Scope: fixed-tau policies only (oracle / seed — tau constant for the whole
scan).  Under a fixed tau the kernel's pruning decisions are per-document
independent (the shared rising threshold never exceeds a valid tau_seed, see
cpp/fused_panel_maxsim/common.hpp::atomic_max_tau), so the simulator is exact up to fp32
summation-order noise.  self_bound (tau rising from -inf in doc order) is
deliberately out of scope — its decisions depend on thread scheduling.

Bound math (Stage 1 §2.3 at document granularity, Stage 3b §5.5): after the
first c dims of `order`,
    UB_d = sum_i max_{j in d} (P_ij + resq_i * resd_j)
    resq_i = beta * sqrt(max(0, 1 - Qcum[i, c])),   beta = shrink + (1-shrink)*(D-c)/D
    resd_j = sqrt(max(0, 1 - sum_{z in scanned} x_jz^2))
and d is pruned at the first checkpoint where UB_d + eps < tau.  Duplicate-
token panel padding adds identical lanes (max-invariant), so simulating on
the UNPADDED corpus gives the same decisions; padded token counts are used
only for the kernel-comparable cells convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Panel width and the float32 guard on the UB < tau test — must mirror
# cpp/fused_panel_maxsim/common.hpp (PT, UB_EPSILON).
_PT = 16
_UB_EPSILON = 1e-4


@dataclass
class CheckpointSimResult:
    """Per-query fused-policy accounting under a fixed tau."""

    checkpoints: list[int]
    """Bound checkpoints actually evaluated (clamped/deduped, D excluded)."""
    docs_pruned_per_checkpoint: np.ndarray
    """int64 [len(checkpoints)] — docs first pruned at each checkpoint."""
    docs_pruned_total: int
    prune_stage: np.ndarray
    """int64 [n_docs] — index into `checkpoints` where the doc was pruned,
    or -1 if it survived to full scan."""
    cells_scanned_pct: float
    """Scanned cells as % of dense (dims x UNPADDED tokens x m) — the
    algorithmic-work convention comparable to live-set accounting."""
    cells_scanned_pct_padded: float
    """Same but with PADDED token counts — predicts the fused kernel's
    stats[0] wall-clock convention."""


def simulate_fused_doc_pruning(
    Q_eff: np.ndarray,
    flat_eff: np.ndarray,
    doc_starts: np.ndarray,
    order: np.ndarray,
    Qcum: np.ndarray,
    checkpoints: list[int] | tuple[int, ...] | np.ndarray,
    tau: float,
    shrink: float = 1.0,
) -> CheckpointSimResult:
    """Simulate the fused doc-level BOND policy for one query at fixed tau.

    Parameters
    ----------
    Q_eff      : float32 [m, D]  — query in the kernel's effective space
                 (rotated for order="pca"), as from dispatch_order_panel
    flat_eff   : float32 [T, D]  — corpus tokens in the SAME effective space
    doc_starts : int64 [n_docs]  — start offset of each doc in flat_eff
    order      : int [D]         — dimension scan order (permutation)
    Qcum       : float32 [m, D+1] — cumulative squared query norms along order
                 (bondmaxsim.data.packing.build_qcum)
    checkpoints: bound-checkpoint dims (kernel semantics: clamped to (0, D),
                 sorted, deduped; D itself never carries a bound)
    tau        : the fixed pruning threshold (kernel tau_seed; use the same
                 resolve_tau_seed value handed to the kernel)
    shrink     : residual scaling (1.0 = exact-safe)
    """
    m, D = Q_eff.shape
    T = flat_eff.shape[0]
    n_docs = len(doc_starts)
    order = np.asarray(order, dtype=np.int64)

    cps = sorted({int(c) for c in np.asarray(checkpoints).ravel() if 0 < int(c) < D})

    doc_ends = np.empty(n_docs, dtype=np.int64)
    doc_ends[:-1] = doc_starts[1:]
    doc_ends[-1] = T
    doc_len = doc_ends - np.asarray(doc_starts, dtype=np.int64)
    doc_len_padded = ((doc_len + _PT - 1) // _PT) * _PT

    prune_stage = np.full(n_docs, -1, dtype=np.int64)
    pruned_per_cp = np.zeros(len(cps), dtype=np.int64)
    alive = np.ones(n_docs, dtype=bool)

    # Incremental partials over dimension segments (float32, like the kernel).
    P = np.zeros((m, T), dtype=np.float32)
    ss = np.zeros(T, dtype=np.float32)
    prev = 0
    for ci, c in enumerate(cps):
        seg = order[prev:c]
        Xseg = flat_eff[:, seg]
        P += Q_eff[:, seg].astype(np.float32) @ Xseg.T.astype(np.float32)
        ss += (Xseg.astype(np.float32) ** 2).sum(axis=1)
        prev = c

        resd = np.sqrt(np.maximum(0.0, 1.0 - ss)).astype(np.float32)          # [T]
        beta = shrink + (1.0 - shrink) * (D - c) / D
        resq = beta * np.sqrt(np.maximum(0.0, 1.0 - Qcum[:, c])).astype(np.float32)  # [m]

        ub_tok = P + resq[:, None] * resd[None, :]                            # [m, T]
        # Per-doc max over the doc's tokens, then sum over query rows.
        ub_doc = np.add.reduce(
            np.maximum.reduceat(ub_tok, np.asarray(doc_starts, dtype=np.int64), axis=1),
            axis=0,
        )                                                                     # [n_docs]

        newly = alive & (ub_doc + _UB_EPSILON < tau)
        prune_stage[newly] = ci
        pruned_per_cp[ci] = int(newly.sum())
        alive &= ~newly

    # Cells accounting: pruned docs scanned cps[stage] dims, survivors D.
    dims_scanned = np.where(prune_stage >= 0,
                            np.asarray(cps, dtype=np.int64)[np.maximum(prune_stage, 0)],
                            D)
    cells = float((dims_scanned * doc_len).sum()) * m
    cells_padded = float((dims_scanned * doc_len_padded).sum()) * m
    dense = float(D) * float(doc_len.sum()) * m
    dense_padded = float(D) * float(doc_len_padded.sum()) * m

    return CheckpointSimResult(
        checkpoints=cps,
        docs_pruned_per_checkpoint=pruned_per_cp,
        docs_pruned_total=int(pruned_per_cp.sum()),
        prune_stage=prune_stage,
        cells_scanned_pct=100.0 * cells / dense if dense else 0.0,
        cells_scanned_pct_padded=100.0 * cells_padded / dense_padded if dense_padded else 0.0,
    )
