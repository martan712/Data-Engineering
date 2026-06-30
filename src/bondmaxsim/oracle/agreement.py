"""Exact-agreement test between pruned and exact MaxSim top-k results.

Single responsibility: compute recall of a pruned kernel's top-k against exact
MaxSim top-k using set-equality semantics.  Must return 1.0 for shrink=1 under
all dimension orders (blocking Stage 2 regression test).

Ported artifact: validate() pattern from
  research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py (the
  correctness check that showed recall=1.0 for the exp-09 kernel).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §2.4–§2.5 (survival
  invariant proof: shrink=1 is exact-safe), §4.3 (set-equality semantics), §8
  items 2 and 5 (exact-agreement test is blocking; order must not change top-k
  set for shrink=1 — regression test).
"""

from __future__ import annotations

import numpy as np


def recall_at_k(
    retrieved: np.ndarray,
    relevant: np.ndarray,
    k: int | None = None,
) -> float:
    """Set-intersection recall of retrieved vs relevant document ids.

    For shrink=1 (exact-safe mode), this must equal 1.0 for all queries.

    Parameters
    ----------
    retrieved : int array — document ids returned by the pruned kernel
    relevant  : int array — document ids from the exact oracle (same k)
    k         : if given, truncate both arrays to their first k elements

    Returns
    -------
    Fraction of relevant ids present in retrieved (in [0, 1]).
    """
    raise NotImplementedError


def exact_agreement(
    pruned_ids: np.ndarray,
    exact_ids: np.ndarray,
) -> float:
    """Return set-overlap fraction between pruned and exact top-k.

    This is recall_at_k with k inferred from len(exact_ids).
    For shrink=1, the return value must be 1.0 (Stage 1 §2.5).
    """
    raise NotImplementedError


def assert_exact_agreement(
    pruned_ids: np.ndarray,
    exact_ids: np.ndarray,
    tol: float = 0.0,
) -> None:
    """Raise AssertionError if exact_agreement < 1.0 - tol.

    Use tol=0.0 for a hard gate (blocking check for shrink=1 mode).
    """
    raise NotImplementedError
