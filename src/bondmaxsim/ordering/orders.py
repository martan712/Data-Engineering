"""Dimension-order signals for BOND-MaxSim.

Single responsibility: given query token matrix Q and optional corpus statistics
(mu, var), return a permutation of {0,...,D-1} that is used as the shared
per-query scan order across all m query tokens.  Different orders affect how
fast the BOND upper-bound tightens but never affect correctness (Stage 1 §4.5).

The per-query order is recomputed once per query (O(D log D)) and reused by
every document, as in PDX-sigmod's GetDimensionsAccessOrder.

Ported artifact: dimension_order() function from
  archive/reference/05_maxsim_bond_instrumentation.py (query_energy, natural,
  doc_var modes) and bond_order/pca_order patterns from
  research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §4.5 (order is
  query-dependent, recomputed per query; MaxSim aggregates token importance over
  m query tokens; pca rotation is orthogonal -> exact-safe at shrink=1),
  docs/sources/bond_maxsim_methodology.md M6 (three signals: q², |q-mu|,
  q²·(mu²+var); DISTANCE_TO_MEANS_IMPROVED top-25% partition).
"""

from __future__ import annotations

import numpy as np


def natural_order(query: np.ndarray) -> np.ndarray:
    """Identity permutation — scan dimensions 0, 1, ..., D-1.

    Parameters
    ----------
    query : float32 [m, D] — query token matrix (unused; signature uniform)

    Returns
    -------
    order : int64 [D]
    """
    D = query.shape[1]
    return np.arange(D, dtype=np.int64)


def bond_order(
    query: np.ndarray,
    mu: np.ndarray,
    top_frac: float = 0.25,
) -> np.ndarray:
    """BOND dimension order: sum_i (q_i - mu)^2, top top_frac% first.

    Mirrors PDX-sigmod BOND's DISTANCE_TO_MEANS_IMPROVED: select the top
    `top_frac` fraction of dimensions by per-dimension importance, then
    sort each partition by physical index (cache-friendly monotonic-with-gaps
    access pattern).  Importance is aggregated over all m query tokens:
    importance[d] = sum_i (q_i[d] - mu[d])^2.

    Parameters
    ----------
    query    : float32 [m, D] — query token matrix
    mu       : float32 [D]    — per-dimension mean over corpus tokens
    top_frac : float          — fraction of dimensions to put in the "hot" partition

    Returns
    -------
    order : int64 [D]
    """
    D = query.shape[1]
    imp = ((query - mu) ** 2).sum(axis=0)           # [D]
    idx = np.argsort(imp)[::-1]                      # descending importance
    tp = int(np.floor(D * top_frac))
    return np.concatenate([np.sort(idx[:tp]), np.sort(idx[tp:])]).astype(np.int64)


def pca_order(
    query: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA (orthogonal rotation) order: identity order on rotated token space.

    Applies a fixed orthogonal rotation R (precomputed over corpus tokens via
    PCA eigenvectors) to Q and doc tokens, then uses natural order.  The
    rotation is orthogonal so it preserves inner products (A2) and unit norm
    (A1) -> exact-safe at shrink=1 (Stage 1 §4.5).  This is NOT ADSampling:
    no ratio bound is used; only the rotation transfers to MaxSim.

    Parameters
    ----------
    query    : float32 [m, D] — original query token matrix
    rotation : float32 [D, D] — orthogonal rotation matrix

    Returns
    -------
    rotated_query : float32 [m, D] — query in rotated space
    order         : int64 [D]      — natural order (identity permutation)
    """
    D = query.shape[1]
    rotated_query = (query @ rotation).astype(np.float32)
    order = np.arange(D, dtype=np.int64)
    return rotated_query, order


def bond_q2_var_order(
    query: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    top_frac: float = 0.25,
) -> np.ndarray:
    """Extended BOND order: q²·(mu²+var) — combines query energy with expected doc energy.

    importance[d] = (sum_i q_i[d]^2) * (mu[d]^2 + var[d])
    This is the principled "BOND for MaxSim" signal (M6 in methodology): shrinks
    both the query and document residual factors.

    Parameters
    ----------
    query    : float32 [m, D]
    mu       : float32 [D]    — per-dimension mean over corpus tokens
    var      : float32 [D]    — per-dimension variance over corpus tokens
    top_frac : float          — fraction of dimensions in the hot partition

    Returns
    -------
    order : int64 [D]
    """
    D = query.shape[1]
    imp = (query ** 2).sum(axis=0) * (mu ** 2 + var)   # [D]
    idx = np.argsort(imp)[::-1]                          # descending importance
    tp = int(np.floor(D * top_frac))
    return np.concatenate([np.sort(idx[:tp]), np.sort(idx[tp:])]).astype(np.int64)
