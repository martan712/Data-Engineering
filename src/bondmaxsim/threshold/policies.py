"""Threshold policies for BOND document pruning.

Single responsibility: compute or maintain the pruning threshold tau_k used in
the document-pruning rule UB_d < tau_k.  Three policies are exposed:
  - self_bound : tau from the k-th best running lower bound (BOND classic)
  - oracle     : tau from the true k-th exact score (upper bound on potential)
  - seed       : tau from a cheap early partial-score seed (realistic policy for
                 synchronized wide-block scan where self_bound stays -inf)

All policies preserve exactness (Stage 1 §4.4: the theorem holds for any tau
that is a valid lower bound on the final k-th score, including -inf); the
difference is purely in how early pruning fires.

Ported artifact: run_bond_for_query / threshold_mode logic from
  archive/reference/05_maxsim_bond_instrumentation.py (brought on-branch from
  Mikel; this module reimplements that logic as clean typed functions/classes).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §4.4 (threshold
  dynamics in sequential vs synchronized scan; seeded threshold is first-class),
  §8 items 4 (approximate arm isolation) and 6 (seeded threshold first-class).
"""

from __future__ import annotations

import numpy as np


def self_bound_threshold(lower_bounds: np.ndarray, k: int) -> float:
    """Threshold from the k-th largest running lower bound on document scores.

    Effective in sequential per-document scans where documents finalize one at
    a time and tau_k can grow between documents.  In synchronized wide-block
    scans no document finalizes mid-pass, so this stays -inf for the full scan.

    Parameters
    ----------
    lower_bounds : float32 [num_live_docs] — current lower bound for each live doc
    k            : int                      — top-k target

    Returns
    -------
    tau : float — k-th largest lower bound, or -inf if fewer than k docs are live
    """
    raise NotImplementedError


def oracle_threshold(exact_scores: np.ndarray, k: int) -> float:
    """Threshold from the true k-th exact score (oracle; upper bound on potential).

    This is NOT a realistic policy — it requires knowing exact scores up front.
    Use only as the upper bound on pruning potential in ablation experiments.

    Parameters
    ----------
    exact_scores : float32 [num_docs] — exact MaxSim scores for all documents
    k            : int

    Returns
    -------
    tau : float — exact k-th best score
    """
    raise NotImplementedError


def seed_threshold(
    partial_scores: np.ndarray,
    exact_scores: np.ndarray,
    k: int,
    seed_fraction: float = 0.02,
) -> float:
    """Threshold seeded from exact scores of the top seed_fraction docs by partial score.

    Realistic cheap seed for the synchronized wide-block scan: at the first
    checkpoint, rank documents by partial score, fully evaluate the top
    seed_fraction fraction, and use their k-th best exact score as tau_k.
    This is never larger than the true k-th best, so it is always safe.

    Parameters
    ----------
    partial_scores : float32 [num_docs] — partial MaxSim score at first checkpoint
    exact_scores   : float32 [num_docs] — true exact MaxSim scores (for the seed docs)
    k              : int
    seed_fraction  : float — fraction of docs to fully evaluate for the seed

    Returns
    -------
    tau : float — seeded threshold, or -inf if fewer than k docs are in the seed
    """
    raise NotImplementedError
