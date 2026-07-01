"""Tests for bondmaxsim.oracle.exact_maxsim."""

import numpy as np
import pytest

from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, exact_maxsim_topk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_corpus(n_docs: int = 6, D: int = 8, seed: int = 42):
    """Return (query, flat_tokens, doc_starts) for a tiny synthetic corpus."""
    rng = np.random.default_rng(seed)
    # Each doc has a random number of tokens (3..8)
    doc_sizes = rng.integers(3, 9, size=n_docs)
    # Build per-doc token matrices (unit-normalised)
    docs = []
    for nd in doc_sizes:
        x = rng.standard_normal((nd, D)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        docs.append(x)
    flat_tokens = np.concatenate(docs, axis=0)  # [T, D]
    doc_starts = np.concatenate([[0], np.cumsum(doc_sizes)[:-1]]).astype(np.int64)
    # Query: 5 unit-norm tokens
    q = rng.standard_normal((5, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q, flat_tokens, doc_starts, docs


def naive_maxsim_scores(query, docs):
    """Independent triple-loop reference: for i: max_j q_i·d_j, then sum."""
    scores = np.zeros(len(docs), dtype=np.float64)
    for d_idx, doc in enumerate(docs):
        s = 0.0
        for i in range(query.shape[0]):
            best = float("-inf")
            for j in range(doc.shape[0]):
                val = float(query[i] @ doc[j])
                if val > best:
                    best = val
            s += best
        scores[d_idx] = s
    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_exact_maxsim_scores_matches_naive():
    q, flat, starts, docs = make_corpus()
    got = exact_maxsim_scores(q, flat, starts)
    ref = naive_maxsim_scores(q, docs)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5,
                               err_msg="exact_maxsim_scores != naive reference")


def test_exact_maxsim_scores_shape_and_dtype():
    q, flat, starts, docs = make_corpus()
    scores = exact_maxsim_scores(q, flat, starts)
    assert scores.shape == (len(docs),)
    assert scores.dtype == np.float32


def test_exact_maxsim_topk_ids_match_argsort():
    q, flat, starts, docs = make_corpus(n_docs=6)
    k = 3
    ids, sc = exact_maxsim_topk(q, flat, starts, k)
    # Reference: full scores sorted descending
    all_scores = exact_maxsim_scores(q, flat, starts)
    ref_ids = np.argsort(all_scores)[::-1][:k]
    # The top-k sets must be equal (set semantics, Stage 1 §4.3)
    assert set(ids.tolist()) == set(ref_ids.tolist()), (
        f"top-{k} set mismatch: got {ids.tolist()}, expected {ref_ids.tolist()}"
    )


def test_exact_maxsim_topk_scores_correct():
    q, flat, starts, docs = make_corpus(n_docs=6)
    k = 3
    ids, sc = exact_maxsim_topk(q, flat, starts, k)
    all_scores = exact_maxsim_scores(q, flat, starts)
    np.testing.assert_allclose(sc, all_scores[ids], rtol=1e-5)


def test_exact_maxsim_topk_descending_order():
    q, flat, starts, docs = make_corpus(n_docs=6)
    k = 4
    ids, sc = exact_maxsim_topk(q, flat, starts, k)
    # Scores must be non-increasing
    assert np.all(np.diff(sc) <= 1e-6), f"Scores not descending: {sc}"


def test_exact_maxsim_topk_k_larger_than_docs():
    """When k > num_docs, return all docs sorted."""
    q, flat, starts, docs = make_corpus(n_docs=4)
    ids, sc = exact_maxsim_topk(q, flat, starts, k=20)
    assert len(ids) == 4
    assert set(ids.tolist()) == {0, 1, 2, 3}


def test_exact_maxsim_topk_dtype():
    q, flat, starts, docs = make_corpus()
    ids, sc = exact_maxsim_topk(q, flat, starts, k=3)
    assert ids.dtype == np.int64
    assert sc.dtype == np.float32


def test_exact_maxsim_empty_doc_safe():
    """A document with zero tokens should score 0 without crashing."""
    D = 8
    rng = np.random.default_rng(7)
    # Create one real doc and one empty doc (starts[1] == starts[2])
    x = rng.standard_normal((4, D)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    flat = x
    # doc_starts: doc 0 -> rows 0:4, doc 1 -> rows 4:4 (empty), doc 2 -> rows 4:... but T=4
    # We'll just test two docs where the second is empty
    starts = np.array([0, 4], dtype=np.int64)
    q = rng.standard_normal((2, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scores = exact_maxsim_scores(q, flat, starts)
    assert scores[1] == 0.0
    assert scores[0] > 0.0
