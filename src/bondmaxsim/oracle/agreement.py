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
    if k is not None:
        retrieved = retrieved[:k]
        relevant = relevant[:k]
    rel_set = set(int(x) for x in relevant)
    if len(rel_set) == 0:
        return 1.0
    ret_set = set(int(x) for x in retrieved)
    return len(ret_set & rel_set) / len(rel_set)


def exact_agreement(
    pruned_ids: np.ndarray,
    exact_ids: np.ndarray,
    exact_scores: np.ndarray | None = None,
) -> float:
    """Return set-overlap fraction between pruned and exact top-k.

    When exact_scores (aligned with exact_ids) is provided, a miss is
    accepted as a valid tie-break when the missing doc's oracle score equals
    the minimum oracle score in the top-K (i.e. it shares the K-th rank with
    another doc and the kernel chose the alternative).  This is correct
    behaviour for shrink=1: the survival invariant guarantees all docs with
    score > tau are retained; boundary ties may resolve differently across
    kernels and BLAS implementations.
    """
    basic = recall_at_k(pruned_ids, exact_ids, k=len(exact_ids))
    if basic == 1.0 or exact_scores is None:
        return basic

    tau = float(exact_scores.min())  # K-th oracle score
    missing = np.setdiff1d(exact_ids, pruned_ids)
    for d in missing:
        idx = np.where(exact_ids == d)[0]
        if len(idx) == 0 or float(exact_scores[idx[0]]) > tau:
            return basic  # genuine miss (score strictly above the K-th boundary)
    return 1.0  # every miss is a boundary tie — kernel's choice is equally valid


def assert_exact_agreement(
    pruned_ids: np.ndarray,
    exact_ids: np.ndarray,
    tol: float = 0.0,
) -> None:
    """Raise AssertionError if exact_agreement < 1.0 - tol.

    Use tol=0.0 for a hard gate (blocking check for shrink=1 mode).
    """
    agreement = exact_agreement(pruned_ids, exact_ids)
    threshold = 1.0 - tol
    if agreement < threshold:
        raise AssertionError(
            f"Exact-agreement check failed: agreement={agreement:.6f} < "
            f"threshold={threshold:.6f} (tol={tol}).  "
            f"pruned={list(pruned_ids)}, exact={list(exact_ids)}"
        )
