"""Stage 3b K2 blocking gate: fused panel kernel exact agreement.

The fused panel kernel is a dense (no-pruning) scan, so it must agree with
the NumPy exact-MaxSim oracle EXACTLY: same top-k document set and matching
scores (float32 accumulation order differs from BLAS, so scores use a tight
tolerance; the id set uses set-equality with tie tolerance, Stage 1 §4.3).

Covers: multi-group corpora, documents needing 0..15 padding tokens, queries
with m > 24 (multi-tile path), K larger/smaller than group doc counts, and
1-thread vs multi-thread invariance.

Requires the compiled kernel (cpp/fused_panel_maxsim/fused_panel_maxsim.so).
Build with `make -C cpp/fused_panel_maxsim`; the test skips cleanly if absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from bondmaxsim.data.packing import pack_corpus_panels
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, exact_maxsim_topk

pytest.importorskip("bondmaxsim.kernels.fused_panel")
from bondmaxsim.kernels.fused_panel import (  # noqa: E402
    load_fused_panel_kernel,
    run_fused_panel_bond,
    run_fused_panel_brute,
)

try:
    LIB = load_fused_panel_kernel()
except FileNotFoundError:
    pytest.skip(
        "fused_panel_maxsim.so not built; run `make -C cpp/fused_panel_maxsim`",
        allow_module_level=True,
    )

SCORE_ATOL = 1e-4  # float32 summation-order tolerance vs BLAS


def _make_corpus(seed: int, n_docs: int, D: int, min_len=1, max_len=40, n_queries=6,
                 mq_range=(2, 34)):
    rng = np.random.default_rng(seed)
    docs, starts, cur = [], [], 0
    for _ in range(n_docs):
        n = int(rng.integers(min_len, max_len + 1))
        starts.append(cur)
        v = rng.standard_normal((n, D)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        docs.append(v)
        cur += n
    flat = np.concatenate(docs, axis=0).astype(np.float32)
    doc_starts = np.array(starts, dtype=np.int64)
    queries = []
    for _ in range(n_queries):
        mq = int(rng.integers(mq_range[0], mq_range[1] + 1))
        q = rng.standard_normal((mq, D)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        queries.append(q)
    return flat, doc_starts, queries


def _assert_agrees(flat, doc_starts, q, packed, k, n_threads=1):
    panel_data, group_offsets, doc_offsets, group_doc_starts, _ = packed
    ids, scores = run_fused_panel_brute(
        LIB, panel_data, group_offsets, doc_offsets, group_doc_starts,
        q, K=k, n_threads=n_threads,
    )
    ref_scores = exact_maxsim_scores(q, flat, doc_starts)
    ref_ids, _ = exact_maxsim_topk(q, flat, doc_starts, k)

    # Scores of the returned docs match the oracle's score of the SAME doc.
    for i in range(min(k, len(doc_starts))):
        assert abs(scores[i] - ref_scores[ids[i]]) <= SCORE_ATOL, (
            f"score mismatch for doc {ids[i]}: kernel {scores[i]} vs "
            f"oracle {ref_scores[ids[i]]}"
        )
    # Set equality with tie tolerance: every returned doc scores at least the
    # oracle's k-th score (within tolerance).
    kth = np.sort(ref_scores)[-min(k, len(doc_starts))]
    assert (ref_scores[ids[:min(k, len(doc_starts))]] >= kth - SCORE_ATOL).all(), (
        f"top-k set mismatch: kernel {sorted(ids.tolist())} vs "
        f"oracle {sorted(ref_ids.tolist())}"
    )


@pytest.mark.parametrize("seed,n_docs,D,target", [
    (7, 20, 32, 4096),      # single group
    (17, 120, 16, 256),     # many groups, small target
    (23, 60, 128, 512),     # D=128 like the real datasets
])
def test_exact_agreement(seed, n_docs, D, target):
    flat, doc_starts, queries = _make_corpus(seed, n_docs, D)
    packed = pack_corpus_panels(flat, doc_starts, target_group_tokens=target)
    for q in queries:
        _assert_agrees(flat, doc_starts, q, packed, k=10)


def test_multi_tile_query_m_over_24():
    """m in (24, 48] exercises the two-tile path (24 + remainder)."""
    flat, doc_starts, _ = _make_corpus(31, 50, 48)
    packed = pack_corpus_panels(flat, doc_starts, target_group_tokens=300)
    rng = np.random.default_rng(9)
    for mq in (25, 32, 34, 48):
        q = rng.standard_normal((mq, 48)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        _assert_agrees(flat, doc_starts, q, packed, k=5)


def test_negative_similarity_docs_not_clamped():
    """Documents whose best similarity to a query token is NEGATIVE must keep
    their true (negative) max — this fails if padding used zero tokens."""
    D = 8
    rng = np.random.default_rng(2)
    q = rng.standard_normal((3, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Docs built as negated/perturbed query tokens -> mostly negative sims;
    # lengths chosen to force 13..15 padding tokens per doc.
    docs, starts, cur = [], [], 0
    for d in range(12):
        n = 3 if d % 2 == 0 else 1
        starts.append(cur)
        v = -q[rng.integers(0, 3, size=n)] + 0.05 * rng.standard_normal((n, D)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        docs.append(v.astype(np.float32))
        cur += n
    flat = np.concatenate(docs, axis=0).astype(np.float32)
    doc_starts = np.array(starts, dtype=np.int64)
    ref = exact_maxsim_scores(q, flat, doc_starts)
    assert (ref < 0).any(), "fixture should produce negative MaxSim scores"
    packed = pack_corpus_panels(flat, doc_starts, target_group_tokens=64)
    _assert_agrees(flat, doc_starts, q, packed, k=len(doc_starts))


ORDERS = ["natural", "bond", "pca"]
POLICIES = ["self_bound", "oracle"]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("policy", POLICIES)
def test_bond_exact_agreement_shrink1(order, policy):
    """Stage 3b K4 blocking gate: the fused BOND kernel at shrink=1 must be
    exact-safe for every dimension order and threshold policy (the Stage 2
    gate re-run on the new kernel), single- and multi-threaded."""
    from bondmaxsim.data.packing import build_qcum
    from bondmaxsim.testbed.packing_cache import PackingCache
    from bondmaxsim.testbed.config import RunConfig
    from bondmaxsim.testbed.thresholds import resolve_tau_seed

    flat, doc_starts, queries = _make_corpus(53, 120, 32, n_queries=5)
    packing = PackingCache(flat, doc_starts)
    cfg = RunConfig(dataset="", method="fused_panel_maxsim_bond",
                    dimension_order=order, threshold_policy=policy,
                    k=10, shrink=1.0)
    for q in queries:
        panel_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, ord_ = (
            packing.dispatch_order_panel(q, order)
        )
        Qcum = build_qcum(Q_eff, ord_)
        tau = resolve_tau_seed(cfg, q, ord_, order, packing)
        ref_scores = exact_maxsim_scores(q, flat, doc_starts)
        kth = np.sort(ref_scores)[-10]
        for n_threads in (1, 4):
            ids, scores, stats = run_fused_panel_bond(
                LIB, panel_data, group_offsets, doc_offsets, group_doc_starts,
                Q_eff, ord_, Qcum, shrink=1.0, tau_seed=tau, K=10,
                n_threads=n_threads,
            )
            for i in range(10):
                assert abs(scores[i] - ref_scores[ids[i]]) <= SCORE_ATOL
            assert (ref_scores[ids] >= kth - SCORE_ATOL).all(), (
                f"order={order} policy={policy} nt={n_threads}: "
                f"top-k set mismatch (recall < 1 at shrink=1)"
            )


def test_bond_prunes_documents():
    """With an oracle threshold the bond kernel must actually prune (stats[1]
    > 0) — otherwise it is just the brute kernel with extra steps.

    Uniform random vectors spread energy evenly across dimensions, which
    leaves the Cauchy-Schwarz residuals large at the mid-scan checkpoints and
    (correctly) prunes nothing.  Real embeddings under bond/pca ordering
    concentrate energy in early dimensions, so the fixture mimics that:
    dims >= 32 are scaled down and vectors renormalized."""
    from bondmaxsim.data.packing import build_qcum
    from bondmaxsim.testbed.packing_cache import PackingCache
    from bondmaxsim.testbed.config import RunConfig
    from bondmaxsim.testbed.thresholds import resolve_tau_seed

    flat, doc_starts, queries = _make_corpus(61, 300, 64, n_queries=2)
    # Concentrate ~99.8% of every vector's energy in dims [0, 32).
    flat = flat.copy(); flat[:, 32:] *= 0.05
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    queries = [q.copy() for q in queries]
    for q in queries:
        q[:, 32:] *= 0.05
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    packing = PackingCache(flat, doc_starts)
    cfg = RunConfig(dataset="", method="fused_panel_maxsim_bond",
                    dimension_order="natural", threshold_policy="oracle",
                    k=5, shrink=1.0)
    q = queries[0]
    panel_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, ord_ = (
        packing.dispatch_order_panel(q, "natural")
    )
    Qcum = build_qcum(Q_eff, ord_)
    tau = resolve_tau_seed(cfg, q, ord_, "natural", packing)
    _, _, stats = run_fused_panel_bond(
        LIB, panel_data, group_offsets, doc_offsets, group_doc_starts,
        Q_eff, ord_, Qcum, shrink=1.0, tau_seed=tau, K=5, n_threads=1,
    )
    assert int(stats[1]) > 0, "oracle-seeded bond kernel pruned zero documents"


def test_thread_invariance():
    flat, doc_starts, queries = _make_corpus(43, 200, 32, n_queries=3)
    packed = pack_corpus_panels(flat, doc_starts, target_group_tokens=256)
    panel_data, group_offsets, doc_offsets, group_doc_starts, _ = packed
    for q in queries:
        ids1, sc1 = run_fused_panel_brute(
            LIB, panel_data, group_offsets, doc_offsets, group_doc_starts,
            q, K=10, n_threads=1)
        ids8, sc8 = run_fused_panel_brute(
            LIB, panel_data, group_offsets, doc_offsets, group_doc_starts,
            q, K=10, n_threads=8)
        # Per-doc scores are computed identically regardless of threading
        # (only the merge order differs), so the sorted output must match.
        np.testing.assert_array_equal(np.sort(ids1), np.sort(ids8))
        np.testing.assert_allclose(np.sort(sc1), np.sort(sc8), rtol=0, atol=0)
