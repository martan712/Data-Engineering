"""Unit-norm verification for token embeddings (blocking Stage 2 check).

Single responsibility: verify that all token embeddings are L2-unit-normalised
to fp32 tolerance.  Failing this check means the shrink=1 kernel is NOT
guaranteed exact-safe (Stage 1 §4.1 is the highest-risk precondition).

Ported artifact: normalization check inferred from
  research/colbert/02b_corect_bruteforce.py (l2_normalize utility) and
  archive/reference/05_maxsim_bond_instrumentation.py (l2_normalize call).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §1.2 (A1: unit
  normalization), §4.1 (highest-risk precondition: ||x|| > 1 silently breaks
  exactness), §8 item 1 (normalization guard is a BLOCKING Stage 2 check).
"""

from __future__ import annotations

import numpy as np


def assert_unit_norm(tokens: np.ndarray, tol: float = 1e-4) -> None:
    """Assert that every row of *tokens* has L2 norm in [1-tol, 1+tol].

    Parameters
    ----------
    tokens : float32 [N, D] — token embedding matrix
    tol    : float          — tolerance for deviation from unit norm

    Raises
    ------
    AssertionError if any token deviates from unit norm by more than tol.
    """
    norms = np.linalg.norm(tokens, axis=1)
    violating = np.where((norms < 1.0 - tol) | (norms > 1.0 + tol))[0]
    if len(violating) > 0:
        raise AssertionError(
            f"Unit-norm check failed: {len(violating)} token(s) outside "
            f"[{1.0 - tol:.6f}, {1.0 + tol:.6f}].  "
            f"min_norm={norms.min():.6f}, max_norm={norms.max():.6f}, "
            f"n_violating={len(violating)}"
        )


def check_unit_norm(tokens: np.ndarray, tol: float = 1e-4) -> dict[str, float]:
    """Return a dict with norm statistics without raising.

    Returns keys: min_norm, max_norm, mean_norm, n_violating (count outside tol).
    """
    norms = np.linalg.norm(tokens, axis=1)
    n_violating = int(np.sum((norms < 1.0 - tol) | (norms > 1.0 + tol)))
    return {
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "n_violating": n_violating,
    }
