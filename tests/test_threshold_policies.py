"""Tests for bondmaxsim.threshold.policies (self_bound / oracle / seed).

Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §4.4 (threshold
policies preserve exactness for any tau that is a valid lower bound on the
true k-th best score, including -inf).
"""

from __future__ import annotations

import numpy as np
import pytest

from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores
from bondmaxsim.threshold.policies import (
    oracle_threshold,
    seed_threshold,
    self_bound_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_tiny_corpus(n_docs: int = 40, D: int = 8, seed: int = 0):
    """Return (query, flat_tokens, doc_starts) with unit-norm rows."""
    rng = np.random.default_rng(seed)
    doc_sizes = rng.integers(2, 6, size=n_docs).tolist()

    docs = []
    for nd in doc_sizes:
        x = rng.standard_normal((nd, D)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        docs.append(x)

    flat_tokens = np.concatenate(docs, axis=0)
    doc_starts = np.concatenate([[0], np.cumsum(doc_sizes)[:-1]]).astype(np.int64)

    m = 4
    q = rng.standard_normal((m, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    return q, flat_tokens, doc_starts


# ---------------------------------------------------------------------------
# oracle_threshold
# ---------------------------------------------------------------------------

def test_oracle_threshold_equals_exact_kth_score():
    q, flat, starts = make_tiny_corpus()
    scores = exact_maxsim_scores(q, flat, starts)
    k = 5

    tau = oracle_threshold(scores, k)
    expected = float(np.sort(scores)[::-1][k - 1])

    assert tau == pytest.approx(expected, abs=1e-5)


def test_oracle_threshold_k_covers_all_docs():
    scores = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    tau = oracle_threshold(scores, k=len(scores))
    assert tau == pytest.approx(1.0)

    tau_over = oracle_threshold(scores, k=len(scores) + 5)
    assert tau_over == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# self_bound_threshold
# ---------------------------------------------------------------------------

def test_self_bound_starts_at_minus_inf():
    k = 5
    tau = self_bound_threshold(np.empty(0, dtype=np.float32), k)
    assert tau == float("-inf")

    # Fewer than k finalized docs: still -inf.
    tau = self_bound_threshold(np.array([1.0, 2.0], dtype=np.float32), k)
    assert tau == float("-inf")


def test_self_bound_monotonically_nondecreasing_as_scores_finalize():
    rng = np.random.default_rng(3)
    all_scores = rng.standard_normal(50).astype(np.float32)
    k = 8

    taus = []
    for n in range(0, len(all_scores) + 1):
        finalized = all_scores[:n]
        taus.append(self_bound_threshold(finalized, k))

    for prev, cur in zip(taus, taus[1:]):
        assert cur >= prev - 1e-6, f"self_bound tau decreased: {prev} -> {cur}"

    # After all docs finalized, must be finite (n >= k).
    assert taus[-1] != float("-inf")


# ---------------------------------------------------------------------------
# seed_threshold
# ---------------------------------------------------------------------------

def test_seed_threshold_at_least_self_bound_at_pass_start():
    """At pass start (no docs finalized yet) self_bound is -inf; seed_threshold
    (built from a cheap partial-score seed) must be >= that."""
    rng = np.random.default_rng(5)
    n_docs = 200
    partial_scores = rng.standard_normal(n_docs).astype(np.float32)
    exact_scores = partial_scores + rng.normal(0, 0.1, n_docs).astype(np.float32)
    k = 10

    tau_self_bound_start = self_bound_threshold(np.empty(0, dtype=np.float32), k)
    tau_seed = seed_threshold(partial_scores, exact_scores, k, seed_fraction=0.1)

    assert tau_seed >= tau_self_bound_start


def test_seed_threshold_never_exceeds_true_kth_best():
    rng = np.random.default_rng(9)
    n_docs = 300
    exact_scores = rng.standard_normal(n_docs).astype(np.float32)
    # Partial scores correlated with, but noisier than, exact scores.
    partial_scores = exact_scores + rng.normal(0, 0.5, n_docs).astype(np.float32)
    k = 10

    true_kth = oracle_threshold(exact_scores, k)
    tau_seed = seed_threshold(partial_scores, exact_scores, k, seed_fraction=0.2)

    assert tau_seed <= true_kth + 1e-6


def test_seed_threshold_minus_inf_when_seed_too_small():
    rng = np.random.default_rng(11)
    n_docs = 20
    partial_scores = rng.standard_normal(n_docs).astype(np.float32)
    exact_scores = partial_scores.copy()
    k = 10

    # seed_fraction so small that n_seed < k.
    tau = seed_threshold(partial_scores, exact_scores, k, seed_fraction=0.01)
    assert tau == float("-inf")


# ---------------------------------------------------------------------------
# Pruning-safety: shrink=1, no true top-k document ever excluded
# ---------------------------------------------------------------------------

def test_pruning_never_excludes_true_topk_doc_any_policy():
    """For each policy, applying tau as `exclude if exact_score < tau` on a
    small random unit-normalized dataset must never exclude a document that
    truly belongs to the top-k (exact oracle as reference, shrink=1 regime)."""
    q, flat, starts = make_tiny_corpus(n_docs=60, D=8, seed=21)
    exact_scores = exact_maxsim_scores(q, flat, starts)
    k = 6

    true_topk_ids = np.argsort(exact_scores)[::-1][:k]
    true_topk_scores = exact_scores[true_topk_ids]

    # oracle policy: tau = true k-th best score exactly.
    tau_oracle = oracle_threshold(exact_scores, k)
    assert np.all(true_topk_scores >= tau_oracle - 1e-6)

    # self_bound policy at pass start (no docs finalized): tau = -inf, always safe.
    tau_self_bound = self_bound_threshold(np.empty(0, dtype=np.float32), k)
    assert np.all(true_topk_scores >= tau_self_bound)

    # seed policy: cheap seed ranked by a partial (rank-1 dim) score.
    rng = np.random.default_rng(23)
    partial_scores = exact_scores + rng.normal(0, 0.05, len(exact_scores)).astype(np.float32)
    tau_seed = seed_threshold(partial_scores, exact_scores, k, seed_fraction=0.3)
    assert np.all(true_topk_scores >= tau_seed - 1e-6)
