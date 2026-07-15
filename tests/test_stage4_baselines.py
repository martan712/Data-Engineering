"""Stage 4 baseline gates: FAISS-IVF rerank correctness, fixed-budget control,
and candidate-seeded threshold safety (R8).

The FAISS-IVF baseline must (a) never exceed the fixed candidate budget
(Convention 7 — candidate-set size held fixed), (b) rerank with EXACT MaxSim so
that with the budget = corpus size it reproduces the exact oracle top-k, and
(c) the candidate-seeded threshold must always be a safe lower bound on the
true k-th best score (Stage 1 §4.4 option b).

The PLAID wrapper gets a build/search smoke gate on a synthetic corpus
(skips cleanly when pylate is not installed).
"""

from __future__ import annotations

import numpy as np
import pytest

from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, exact_maxsim_topk
from bondmaxsim.threshold.policies import candidate_seed_threshold, oracle_threshold


def _synthetic_corpus(seed: int = 0, n_docs: int = 300, D: int = 32):
    """Unit-norm token corpus with variable doc lengths + a few queries."""
    rng = np.random.default_rng(seed)
    lens = rng.integers(4, 16, size=n_docs)
    starts = np.zeros(n_docs, dtype=np.int64)
    np.cumsum(lens[:-1], out=starts[1:])
    T = int(lens.sum())
    tokens = rng.normal(size=(T, D)).astype(np.float32)
    tokens /= np.linalg.norm(tokens, axis=1, keepdims=True)
    queries = []
    for _ in range(5):
        q = rng.normal(size=(8, D)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        queries.append(q)
    return tokens, starts, queries


# ---------------------------------------------------------------------------
# candidate_seed_threshold (no external deps)
# ---------------------------------------------------------------------------

def test_candidate_seed_threshold_is_safe_lower_bound():
    tokens, starts, queries = _synthetic_corpus()
    k = 10
    rng = np.random.default_rng(1)
    for q in queries:
        scores = exact_maxsim_scores(q, tokens, starts)
        tau_true = oracle_threshold(scores, k)
        for n_cand in (10, 25, 100):
            cand = rng.choice(len(starts), size=n_cand, replace=False)
            tau_seed = candidate_seed_threshold(scores[cand], k)
            assert tau_seed <= tau_true + 1e-6, (
                "subset k-th best exceeded the true k-th best")


def test_candidate_seed_threshold_underfilled_is_neg_inf():
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert candidate_seed_threshold(scores, k=5) == float("-inf")


# ---------------------------------------------------------------------------
# gather_candidate_tokens + FaissIVFBaseline
# ---------------------------------------------------------------------------

def test_gather_candidate_tokens_matches_naive():
    from bondmaxsim.baselines.faiss_ivf import gather_candidate_tokens

    tokens, starts, _ = _synthetic_corpus()
    ends = np.append(starts[1:], len(tokens))
    doc_ids = np.array([5, 0, 42, 299, 7], dtype=np.int64)
    cand_tokens, cand_starts = gather_candidate_tokens(tokens, starts, doc_ids)
    naive = np.concatenate([tokens[starts[d]:ends[d]] for d in doc_ids])
    np.testing.assert_array_equal(cand_tokens, naive)
    lens = ends[doc_ids] - starts[doc_ids]
    np.testing.assert_array_equal(np.diff(cand_starts), lens[:-1])


faiss = pytest.importorskip("faiss")


def _built_baseline(tokens, starts, **kw):
    from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline

    b = FaissIVFBaseline(tokens, starts, n_lists=16, **kw)
    b.build()
    return b


def test_faiss_ivf_budget_is_respected():
    tokens, starts, queries = _synthetic_corpus()
    b = _built_baseline(tokens, starts)
    for budget in (10, 50, 100):
        for q in queries:
            cand = b.candidates(q, candidate_budget=budget)
            assert len(cand) <= budget
            assert len(np.unique(cand)) == len(cand)


def test_faiss_ivf_full_budget_reproduces_exact_oracle():
    """With budget = corpus size and all lists probed, the pipeline IS exact."""
    tokens, starts, queries = _synthetic_corpus()
    b = _built_baseline(tokens, starts, nprobe=16)
    k = 10
    for q in queries:
        ids, scores, n_cand = b.topk(q, k=k, candidate_budget=len(starts),
                                     k_token=len(tokens))
        e_ids, e_scores = exact_maxsim_topk(q, tokens, starts, k)
        assert set(ids.tolist()) == set(e_ids.tolist())
        np.testing.assert_allclose(np.sort(scores), np.sort(e_scores), rtol=1e-5)


def test_faiss_ivf_rerank_is_exact_on_candidates():
    """Rerank scores must equal exact MaxSim for the candidate docs."""
    tokens, starts, queries = _synthetic_corpus()
    b = _built_baseline(tokens, starts)
    q = queries[0]
    cand = b.candidates(q, candidate_budget=50)
    ids, scores = b.rerank(q, cand, k=10)
    full = exact_maxsim_scores(q, tokens, starts)
    np.testing.assert_allclose(scores, full[ids], rtol=1e-5)
    assert set(ids.tolist()) <= set(cand.tolist())


# ---------------------------------------------------------------------------
# PDXIVFBaseline (Mikel-branch flat PDX-IVF; needs the pdxearch build)
# ---------------------------------------------------------------------------

pdxearch = pytest.importorskip("pdxearch")


def test_pdx_ivf_budget_and_exactness():
    from bondmaxsim.baselines.pdx_ivf import PDXIVFBaseline

    tokens, starts, queries = _synthetic_corpus()
    b = PDXIVFBaseline(tokens, starts, n_buckets=16, nprobe=16)
    b.build()
    # Budget is respected and candidates unique.
    for q in queries:
        cand = b.candidates(q, candidate_budget=50)
        assert len(cand) <= 50
        assert len(np.unique(cand)) == len(cand)
    # Full budget + all buckets probed reproduces the exact oracle.
    q = queries[0]
    ids, scores, _ = b.topk(q, k=10, candidate_budget=len(starts),
                            k_token=len(tokens))
    e_ids, e_scores = exact_maxsim_topk(q, tokens, starts, 10)
    assert set(ids.tolist()) == set(e_ids.tolist())
    np.testing.assert_allclose(np.sort(scores), np.sort(e_scores), rtol=1e-5)


# ---------------------------------------------------------------------------
# PartitionedFusedScan (IVF-partitioned fused BOND scan)
# ---------------------------------------------------------------------------

def _partitioned_index(tokens, starts, n_partitions=8):
    from bondmaxsim.partitioned_scan import PartitionedFusedScan

    idx = PartitionedFusedScan(tokens, starts, n_partitions=n_partitions)
    stats = idx.build()
    assert stats["n_partitions_nonempty"] >= 2
    return idx


def _fused_lib_or_skip():
    from bondmaxsim.kernels.fused_panel import load_fused_panel_kernel

    try:
        return load_fused_panel_kernel()
    except FileNotFoundError:
        pytest.skip("fused_panel_maxsim.so not built")


def test_partitioned_scan_full_probe_is_exact():
    """nprobe = all partitions degenerates to the exact exhaustive scan."""
    from bondmaxsim.oracle.agreement import validate_boundary_tie_equivalence

    lib = _fused_lib_or_skip()
    tokens, starts, queries = _synthetic_corpus(seed=3)
    idx = _partitioned_index(tokens, starts)
    k = 10
    for q in queries:
        ids, scores, stats = idx.search(lib, q, k=k, nprobe=len(idx.partitions),
                                        checkpoints=(16,), n_threads=1)
        full_scores = exact_maxsim_scores(q, tokens, starts)
        e_ids, e_scores = exact_maxsim_topk(q, tokens, starts, k)
        assert stats["docs_probed_pct"] == pytest.approx(100.0)
        agreement = validate_boundary_tie_equivalence(
            ids, e_ids, e_scores, k=k, num_documents=len(starts),
            exact_scores_by_id=full_scores,
        )
        assert agreement.exact_gate_passed
        np.testing.assert_allclose(np.sort(scores), np.sort(e_scores),
                                   rtol=1e-4, atol=1e-4)


def test_partitioned_scan_partial_probe_is_sane():
    lib = _fused_lib_or_skip()
    tokens, starts, queries = _synthetic_corpus(seed=4)
    idx = _partitioned_index(tokens, starts)
    for q in queries:
        ids, scores, stats = idx.search(lib, q, k=10, nprobe=2,
                                        checkpoints=(16,), n_threads=1)
        assert stats["docs_probed_pct"] < 100.0
        assert len(ids) == len(np.unique(ids))
        # Scores must be exact MaxSim scores of the returned docs.
        full = exact_maxsim_scores(q, tokens, starts)
        np.testing.assert_allclose(scores, full[ids], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# PLAID wrapper smoke gate
# ---------------------------------------------------------------------------

pylate = pytest.importorskip("pylate")


def test_plaid_build_and_search_smoke(tmp_path):
    from bondmaxsim.baselines.plaid import PLAIDBaseline
    from bondmaxsim.data.loader import unpack_embeddings

    tokens, starts, queries = _synthetic_corpus(n_docs=200)
    docs = unpack_embeddings(tokens, starts)
    p = PLAIDBaseline("synthetic", index_root=tmp_path, n_full_scores=200)
    p.build(docs)
    res = p.search(queries, k=10)
    assert len(res) == len(queries)
    hits = 0
    for (ids, scores), q in zip(res, queries):
        assert 1 <= len(ids) <= 10
        assert np.all(np.diff(scores) <= 1e-6)  # descending
        e_ids, _ = exact_maxsim_topk(q, tokens, starts, 10)
        hits += len(set(ids.tolist()) & set(e_ids.tolist()))
    # Quantized PLAID on random data is approximate; require non-trivial overlap.
    assert hits >= len(queries) * 3
