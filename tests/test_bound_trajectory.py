"""Tests for bondmaxsim.oracle.bound_trajectory.

Tiny synthetic unit-norm data verifies the four invariants in the spec:
  (a) UB_d(D) ≈ exact MaxSim score (residuals vanish at full rank).
  (b) UB_d(k) >= exact score for all k (valid upper bound).
  (c) UB_d(k) is non-increasing in k.
  (d) Token-live fraction is non-increasing in k.
  (e) A document with exact score < tau is eventually non-live.
"""

from __future__ import annotations

import numpy as np
import pytest

from bondmaxsim.oracle.bound_trajectory import (
    _compute_all,
    doc_ub_trajectory,
    early_token_pruning_rate,
    dims_to_prune_pct,
    make_prefix_grid,
    survival_trajectories,
    survival_trajectories_self_bound,
)
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_tiny_corpus(
    n_docs: int = 5,
    D: int = 8,
    seed: int = 0,
    doc_sizes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (query, flat_tokens, doc_starts, order) with unit-norm rows."""
    rng = np.random.default_rng(seed)

    if doc_sizes is None:
        doc_sizes = rng.integers(2, 5, size=n_docs).tolist()

    docs = []
    for nd in doc_sizes:
        x = rng.standard_normal((nd, D)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        docs.append(x)

    flat_tokens = np.concatenate(docs, axis=0)
    doc_starts = np.concatenate([[0], np.cumsum(doc_sizes)[:-1]]).astype(np.int64)

    m = 4  # query tokens
    q = rng.standard_normal((m, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    order = np.arange(D, dtype=np.int64)  # natural order for tests

    return q, flat_tokens, doc_starts, order


# ---------------------------------------------------------------------------
# (a) UB_d(D) ≈ exact MaxSim score
# ---------------------------------------------------------------------------

def test_ub_at_full_rank_equals_exact_score():
    q, flat, starts, order = make_tiny_corpus(D=8)
    grid = make_prefix_grid(8, n_points=9)

    ub_traj, exact_scores = doc_ub_trajectory(q, flat, starts, order, grid)

    # UB at k=D (last grid point) should equal exact MaxSim score.
    ref = exact_maxsim_scores(q, flat, starts)

    np.testing.assert_allclose(
        ub_traj[:, -1], ref, rtol=1e-4, atol=1e-4,
        err_msg="UB_d(D) should match exact MaxSim score"
    )


# ---------------------------------------------------------------------------
# (b) UB_d(k) >= exact score for all k (valid upper bound)
# ---------------------------------------------------------------------------

def test_ub_is_upper_bound_at_all_prefixes():
    q, flat, starts, order = make_tiny_corpus(D=8)
    grid = make_prefix_grid(8)

    ub_traj, exact_scores = doc_ub_trajectory(q, flat, starts, order, grid)

    # Every UB entry must be >= the exact score (with small float tolerance).
    for s in range(len(grid)):
        diff = exact_scores - ub_traj[:, s]
        assert np.all(diff <= 1e-4), (
            f"UB_d({grid[s]}) < exact_score by up to {diff.max():.6f} "
            f"for {(diff > 1e-4).sum()} docs"
        )


# ---------------------------------------------------------------------------
# (c) UB_d(k) non-increasing in k
# ---------------------------------------------------------------------------

def test_ub_trajectory_non_increasing():
    q, flat, starts, order = make_tiny_corpus(D=8)
    grid = make_prefix_grid(8)

    ub_traj, _ = doc_ub_trajectory(q, flat, starts, order, grid)

    # Consecutive steps should be non-increasing (allow small float noise).
    violations = 0
    for s in range(1, len(grid)):
        diff = ub_traj[:, s] - ub_traj[:, s - 1]
        if np.any(diff > 1e-4):
            violations += np.sum(diff > 1e-4)
    assert violations == 0, (
        f"UB_d(k) increased at {violations} (doc, step) pairs — "
        "should be non-increasing"
    )


# ---------------------------------------------------------------------------
# (d) Token-live fraction non-increasing in k
# ---------------------------------------------------------------------------

def test_token_live_frac_non_increasing():
    q, flat, starts, order = make_tiny_corpus(D=8)
    grid = make_prefix_grid(8)

    # Use oracle tau = 0 (everything potentially live — tests pure token pruning).
    _, token_live_frac = survival_trajectories(q, flat, starts, order, tau=0.0, prefix_grid=grid)

    # Should be non-increasing (allow tiny float noise).
    diffs = np.diff(token_live_frac)
    assert np.all(diffs <= 1e-6), (
        f"Token-live fraction increased at steps {np.where(diffs > 1e-6)[0].tolist()}: "
        f"{diffs[diffs > 1e-6]}"
    )


# ---------------------------------------------------------------------------
# (e) A doc with score < tau is eventually non-live
# ---------------------------------------------------------------------------

def test_low_score_doc_eventually_pruned():
    """The lowest-scoring document should be pruned under oracle tau = 2nd score."""
    q, flat, starts, order = make_tiny_corpus(n_docs=6, D=8, seed=7)
    grid = make_prefix_grid(8)

    _, exact_scores = doc_ub_trajectory(q, flat, starts, order, grid)

    # Set tau just above the minimum score — worst doc should be pruned by k=D.
    tau = float(np.sort(exact_scores)[1])  # 2nd lowest: at least 1 doc < tau
    doc_live_frac, _ = survival_trajectories(q, flat, starts, order, tau=tau, prefix_grid=grid)

    # At k=D, at least one doc should be pruned (live fraction < 1).
    assert doc_live_frac[-1] < 1.0, (
        f"Expected at least one pruned doc at k=D with tau={tau:.4f}, "
        f"but doc_live_frac[-1]={doc_live_frac[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Additional sanity checks
# ---------------------------------------------------------------------------

def test_make_prefix_grid_includes_0_and_D():
    for D in [8, 16, 128]:
        g = make_prefix_grid(D)
        assert g[0] == 0, f"Grid should start at 0 (D={D})"
        assert g[-1] == D, f"Grid should end at D={D}"
        assert np.all(np.diff(g) > 0), "Grid should be strictly increasing"


def test_dims_to_prune_pct_basic():
    # live_frac goes 1.0, 0.8, 0.6, 0.4, 0.2 at k=0,2,4,6,8
    grid = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    live_frac = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    # 50% pruned means <= 0.5 alive → first at k=6
    assert dims_to_prune_pct(live_frac, grid, 50) == 6
    # 80% pruned means <= 0.2 alive → first at k=8
    assert dims_to_prune_pct(live_frac, grid, 80) == 8


def test_self_bound_survival_non_increasing():
    q, flat, starts, order = make_tiny_corpus(D=8)
    grid = make_prefix_grid(8)

    doc_frac, tok_frac = survival_trajectories_self_bound(
        q, flat, starts, order, k_top=3, prefix_grid=grid
    )

    # Both curves should be non-increasing (allow small noise).
    assert np.all(np.diff(doc_frac) <= 1e-6), "doc_live_frac (self_bound) should be non-increasing"
    assert np.all(np.diff(tok_frac) <= 1e-6), "token_live_frac (self_bound) should be non-increasing"


def test_early_token_pruning_rate_range():
    grid = make_prefix_grid(8)
    tok_frac = np.linspace(1.0, 0.3, len(grid))  # monotone decreasing
    rate = early_token_pruning_rate(tok_frac, grid, D=8)
    assert 0.0 <= rate <= 1.0, f"early_token_pruning_rate out of [0,1]: {rate}"
