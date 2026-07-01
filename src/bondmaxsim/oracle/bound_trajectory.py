"""Bound trajectory analysis for BOND-MaxSim dimension pruning.

Single responsibility: compute, for one query vs a corpus under a given
dimension scan order, the document upper-bound (UB_d(k)) slack trajectory
over a prefix grid, and two-level survival curves (document-live fraction and
token-live fraction) under a threshold policy.  All functions are dataset-
agnostic: they take (flat_tokens, doc_starts, query, order, ...) directly.

Memory strategy: corpora can have O(1M) tokens at D=128.  Allocating full
[T, D] or [m, T] arrays (hundreds of MB) would swap on memory-limited machines.
``_compute_all`` therefore processes documents in batches so that only
[n_batch_toks, D] is resident at once (typically ~50–100 MB per batch).

Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §2.3 (Cauchy-Schwarz
  upper bound P_ij(k) ± resq_i(k)·resd_j(k)), §4.4 (threshold policies:
  oracle / self_bound), §5.2–5.3 (document pruning vs token pruning), §6
  (D/4 warmup / early-token-pruning rate).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Grid helper
# ---------------------------------------------------------------------------

def make_prefix_grid(D: int, n_points: int = 33) -> np.ndarray:
    """Build a prefix grid k in [0, D] with geometric spacing.

    Always includes 0 and D.  Geometric spacing provides extra resolution near
    small k where the Cauchy-Schwarz bound changes fastest.

    Parameters
    ----------
    D        : number of dimensions (e.g. 128)
    n_points : target number of grid points (deduplicated; at least 2)

    Returns
    -------
    grid : int64 ndarray, sorted, starts at 0, ends at D
    """
    if n_points < 2:
        return np.array([0, D], dtype=np.int64)
    inner = np.geomspace(1, D, n_points - 2, dtype=float).round().astype(np.int64)
    return np.unique(np.concatenate([[0], inner, [D]])).astype(np.int64)


# ---------------------------------------------------------------------------
# Core engine (batch-per-document to limit peak memory)
# ---------------------------------------------------------------------------

def _compute_all(
    query: np.ndarray,
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    order: np.ndarray,
    prefix_grid: np.ndarray,
    doc_batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core engine: compute UB, LB, token-live count, and exact scores.

    Processes documents in batches of ``doc_batch_size`` to cap peak memory at
    roughly ``n_batch_toks * D * 5 * 4`` bytes (typically 50–100 MB).

    For each prefix k in prefix_grid, computes over the first k dims of order:
      P_ij(k) = dot(q_i[:k_ord], d_j[:k_ord])              (partial dot product)
      resq_i(k) = sqrt(max(0, 1 - ||q_i[:k_ord]||^2))      (query residual norm)
      resd_j(k) = sqrt(max(0, 1 - ||d_j[:k_ord]||^2))      (doc residual norm)
      U_ij(k) = P_ij(k) + resq_i(k)*resd_j(k)              (upper bound)
      L_ij(k) = P_ij(k) - resq_i(k)*resd_j(k)              (lower bound)
      UB_d(k) = sum_i max_{j in d} U_ij(k)                 (document UB)
      LB_d(k) = sum_i max_{j in d} L_ij(k)                 (document LB)
      live_j = exists i: U_ij(k) >= max_{j'} L_ij'(k)      (token liveness)

    Parameters
    ----------
    query          : float32 [m, D]
    flat_tokens    : float32 [T, D]
    doc_starts     : int64  [num_docs] — start offset of each doc (doc_starts[0]=0)
    order          : int64  [D]        — permutation of 0..D-1
    prefix_grid    : int64  [S]        — sorted prefix lengths in [0, D]
    doc_batch_size : int               — number of docs per memory batch

    Returns
    -------
    ub_traj          : float32 [num_docs, S]
    lb_traj          : float32 [num_docs, S]
    token_live_count : int64  [S]   — # live tokens at each prefix step
    exact_scores     : float32 [num_docs]  — UB at k=D (residuals vanish)
    """
    m, D = query.shape
    T = flat_tokens.shape[0]
    num_docs = len(doc_starts)
    S = len(prefix_grid)

    # Reindex query to scan order (small: [m, D]).
    Q_ord = np.ascontiguousarray(query[:, order], dtype=np.float32)  # [m, D]

    # Cumulative squared norms for query along scan order.
    # Qcum2[i, k] = sum_{z=0}^{k-1} Q_ord[i, z]^2
    Qcum2 = np.zeros((m, D + 1), dtype=np.float32)
    Qcum2[:, 1:] = np.cumsum(Q_ord ** 2, axis=1)

    # Pre-cache resq for all prefix steps (tiny: [m] per step).
    resq_by_step: list[np.ndarray] = [
        np.sqrt(np.maximum(0.0, 1.0 - Qcum2[:, int(k)])).astype(np.float32)
        for k in prefix_grid
    ]

    # Output arrays.
    ub_traj = np.zeros((num_docs, S), dtype=np.float32)
    lb_traj = np.zeros((num_docs, S), dtype=np.float32)
    token_live_count = np.zeros(S, dtype=np.int64)

    # Process batches of documents.
    for batch_start in range(0, num_docs, doc_batch_size):
        batch_end = min(batch_start + doc_batch_size, num_docs)
        n_batch_docs = batch_end - batch_start

        # Token range for this batch.
        tok_start = int(doc_starts[batch_start])
        tok_end = int(doc_starts[batch_end]) if batch_end < num_docs else T
        n_b = tok_end - tok_start  # number of tokens in batch

        # Reindex batch tokens to scan order (contiguous float32 for BLAS).
        F_b = np.ascontiguousarray(
            flat_tokens[tok_start:tok_end][:, order], dtype=np.float32
        )  # [n_b, D]

        # Cumulative squared norms for batch tokens; pre-cache resd per step.
        Fcum2_b = np.zeros((n_b, D + 1), dtype=np.float32)
        Fcum2_b[:, 1:] = np.cumsum(F_b ** 2, axis=1)
        resd_by_step_b: list[np.ndarray] = [
            np.sqrt(np.maximum(0.0, 1.0 - Fcum2_b[:, int(k)])).astype(np.float32)
            for k in prefix_grid
        ]
        del Fcum2_b  # no longer needed

        # Batch-relative doc start offsets (for np.maximum.reduceat).
        batch_doc_starts = doc_starts[batch_start:batch_end]
        batch_seg = (batch_doc_starts - tok_start).astype(np.intp)

        # Batch tok→doc mapping (local doc indices 0..n_batch_docs-1).
        batch_tok2doc = np.empty(n_b, dtype=np.int64)
        for bi in range(n_batch_docs):
            d = batch_start + bi
            s_d = int(doc_starts[d]) - tok_start
            e_d = (int(doc_starts[d + 1]) if d + 1 < num_docs else T) - tok_start
            batch_tok2doc[s_d:e_d] = bi

        # Incremental partial dot products [m, n_b].
        P_b = np.zeros((m, n_b), dtype=np.float32)

        prev_k = 0
        for s, k in enumerate(prefix_grid):
            k = int(k)

            # Extend P_b by the dims from prev_k to k.
            if k > prev_k:
                P_b += Q_ord[:, prev_k:k] @ F_b[:, prev_k:k].T

            prev_k = k

            resq = resq_by_step[s]             # [m]
            resd = resd_by_step_b[s]           # [n_b]

            # Outer product of residuals.
            resid = resq[:, np.newaxis] * resd[np.newaxis, :]  # [m, n_b]

            # ---- Lower bounds ----
            L_b = P_b - resid                                              # [m, n_b]
            LB_qi_b = np.maximum.reduceat(L_b, batch_seg, axis=1)         # [m, n_batch_docs]
            lb_traj[batch_start:batch_end, s] = LB_qi_b.sum(axis=0)

            # Lbest per token: for token j, the best lower bound across all i.
            Lbest_per_tok = LB_qi_b[:, batch_tok2doc]                     # [m, n_b]
            del L_b, LB_qi_b

            # ---- Upper bounds ----
            U_b = P_b + resid                                              # [m, n_b]
            del resid
            UB_qi_b = np.maximum.reduceat(U_b, batch_seg, axis=1)         # [m, n_batch_docs]
            ub_traj[batch_start:batch_end, s] = UB_qi_b.sum(axis=0)
            del UB_qi_b

            # ---- Token liveness ----
            live_b = (U_b >= Lbest_per_tok).any(axis=0)                   # [n_b]
            token_live_count[s] += int(live_b.sum())

            del U_b, Lbest_per_tok, live_b

        del F_b, P_b, resd_by_step_b, batch_tok2doc

    # Exact scores: UB at k=D (residuals vanish for unit-norm tokens).
    d_idx = int(np.searchsorted(prefix_grid, D))
    exact_scores = ub_traj[:, d_idx].copy()

    return ub_traj, lb_traj, token_live_count, exact_scores


# ---------------------------------------------------------------------------
# Public API — document UB trajectory (e01)
# ---------------------------------------------------------------------------

def doc_ub_trajectory(
    query: np.ndarray,
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    order: np.ndarray,
    prefix_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute UB_d(k) trajectories and exact scores for all documents.

    Parameters
    ----------
    query        : float32 [m, D]
    flat_tokens  : float32 [T, D]
    doc_starts   : int64  [num_docs]
    order        : int64  [D]
    prefix_grid  : int64  [S]

    Returns
    -------
    ub_traj      : float32 [num_docs, S]
    exact_scores : float32 [num_docs]
    """
    ub, _lb, _tlc, scores = _compute_all(
        query, flat_tokens, doc_starts, order, prefix_grid
    )
    return ub, scores


# ---------------------------------------------------------------------------
# Public API — survival trajectories (e02)
# ---------------------------------------------------------------------------

def survival_trajectories(
    query: np.ndarray,
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    order: np.ndarray,
    tau: float,
    prefix_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute doc-live and token-live fractions under a fixed threshold tau.

    Document d is pruned at step k when UB_d(k) < tau (oracle policy: tau is
    the exact k-th best score, fixed for all prefix steps).

    Parameters
    ----------
    tau : float — fixed document pruning threshold (exact k-th score for oracle)

    Returns
    -------
    doc_live_frac   : float32 [S] — fraction of docs with UB_d(k) >= tau
    token_live_frac : float32 [S] — fraction of tokens live in >= 1 query token
    """
    ub, _lb, tlc, _scores = _compute_all(
        query, flat_tokens, doc_starts, order, prefix_grid
    )
    T = flat_tokens.shape[0]
    num_docs = len(doc_starts)

    doc_live_frac = (ub >= tau).sum(axis=0).astype(np.float32) / num_docs
    token_live_frac = tlc.astype(np.float32) / T

    return doc_live_frac, token_live_frac


def survival_trajectories_self_bound(
    query: np.ndarray,
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    order: np.ndarray,
    k_top: int,
    prefix_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute survival fractions under the self_bound threshold policy.

    At each prefix k, the threshold is the k_top-th largest document lower bound:
      tau_k = kth_largest(LB_d(k) for d in docs)
    Document d is live at step k if UB_d(k) >= tau_k.

    Parameters
    ----------
    k_top : int — number of results (top-k); threshold = k_top-th LB

    Returns
    -------
    doc_live_frac   : float32 [S]
    token_live_frac : float32 [S]
    """
    ub, lb, tlc, _scores = _compute_all(
        query, flat_tokens, doc_starts, order, prefix_grid
    )
    T = flat_tokens.shape[0]
    num_docs = len(doc_starts)
    S = len(prefix_grid)

    doc_live_frac = np.zeros(S, dtype=np.float32)
    for s in range(S):
        if k_top >= num_docs:
            tau_s = float(lb[:, s].min())
        else:
            # k_top-th largest lower bound.
            tau_s = float(np.partition(lb[:, s], -k_top)[-k_top])
        doc_live_frac[s] = float((ub[:, s] >= tau_s).sum()) / num_docs

    token_live_frac = tlc.astype(np.float32) / T

    return doc_live_frac, token_live_frac


# ---------------------------------------------------------------------------
# Derived scalars
# ---------------------------------------------------------------------------

def dims_to_prune_pct(
    live_frac: np.ndarray,
    prefix_grid: np.ndarray,
    pct: float,
) -> int:
    """Return the smallest k at which >= pct% of items have been pruned.

    Parameters
    ----------
    live_frac   : float [S] — fraction of items still live at each prefix k
    prefix_grid : int64 [S]
    pct         : float in (0, 100) — pruning percentage target (e.g. 90)

    Returns
    -------
    k : int — prefix length when live_frac <= (1 - pct/100) first holds;
        returns prefix_grid[-1] if never reached.
    """
    target_live = 1.0 - pct / 100.0
    for k, frac in zip(prefix_grid, live_frac):
        if frac <= target_live:
            return int(k)
    return int(prefix_grid[-1])


def early_token_pruning_rate(
    token_live_frac: np.ndarray,
    prefix_grid: np.ndarray,
    D: int,
) -> float:
    """Fraction of tokens pruned before D/4 dimensions (Stage 1 §6).

    Parameters
    ----------
    token_live_frac : float [S]
    prefix_grid     : int64 [S]
    D               : int — total number of dimensions

    Returns
    -------
    rate : float in [0, 1] — fraction of tokens pruned before D/4
    """
    quarter = D // 4
    idx = max(0, int(np.searchsorted(prefix_grid, quarter, side='right')) - 1)
    return 1.0 - float(token_live_frac[idx])
