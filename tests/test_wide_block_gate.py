"""Blocking Stage 2 gate: wide-block kernel shrink=1 exact-agreement.

Mirrors tests/test_runner_gate.py for the wide-block MaxSim BOND kernel
(cpp/wide_block_maxsim_bond/).  Stage 1 reference:
docs/stage1_bond_maxsim_formalization.md §2.5, §5.3, §8 items 2-3, 6-8.  At
shrink=1 the wide-block kernel must be exact-safe: its top-k document set
must agree perfectly with the NumPy exact-MaxSim oracle, for every dimension
order (natural, bond, pca), in both accounting and throughput modes, and
under both the self_bound and oracle threshold policies (Stage 1 §4.4).

Requires the compiled kernel (cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.so).
Build it with `make -C cpp/wide_block_maxsim_bond` or `./setup.sh`; the test
skips cleanly if the library is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from bondmaxsim.data.packing import pack_corpus_wide
from bondmaxsim.testbed.runner import Runner, RunConfig

# Skip the whole module if the wide-block kernel library has not been built.
pytest.importorskip("bondmaxsim.kernels.wide_block")
from bondmaxsim.kernels.wide_block import load_wide_block_kernel  # noqa: E402

try:
    load_wide_block_kernel()
except FileNotFoundError:
    pytest.skip(
        "wide_block_maxsim_bond.so not built; run `make -C cpp/wide_block_maxsim_bond`",
        allow_module_level=True,
    )

ORDERS = ["natural", "bond", "pca"]
POLICIES = ["self_bound", "oracle"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_corpus(seed: int, n_docs: int, D: int):
    """Build a unit-norm synthetic corpus and query set (same pattern as
    tests/test_runner_gate.py's _make_corpus)."""
    rng = np.random.default_rng(seed)
    docs, starts, cur = [], [], 0
    for _ in range(n_docs):
        m = int(rng.integers(3, 12))
        starts.append(cur)
        v = rng.standard_normal((m, D)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        docs.append(v)
        cur += m
    flat = np.concatenate(docs, axis=0).astype(np.float32)
    doc_starts = np.array(starts, dtype=np.int64)

    queries = []
    for _ in range(8):
        mq = int(rng.integers(2, 6))
        q = rng.standard_normal((mq, D)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        queries.append(q)
    return flat, doc_starts, queries


@pytest.fixture(scope="module")
def runner():
    flat, doc_starts, queries = _make_corpus(seed=7, n_docs=20, D=32)
    return Runner(flat, doc_starts, queries)


@pytest.fixture(scope="module")
def runner_multigroup():
    """A larger, longer-document corpus that spans multiple wide vectorgroups
    (target ~4096 tokens/group), so staged finalization (Stage 1 §4.4) is
    actually exercised, not just the single-group degenerate case."""
    flat, doc_starts, queries = _make_corpus(seed=17, n_docs=120, D=16)
    return Runner(flat, doc_starts, queries)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("policy", POLICIES)
def test_accounting_exact_agreement(runner, order, policy):
    """shrink=1 accounting mode must reproduce the exact top-k for every
    order and threshold policy."""
    cfg = RunConfig(
        dataset="synthetic",
        method="wide_block_maxsim_bond",
        dimension_order=order,
        threshold_policy=policy,
        k=10,
        shrink=1.0,
    )
    rec = runner.accounting_mode(cfg)
    assert rec.recall_vs_exact_at_10 == 1.0, (
        f"accounting/{order}/{policy}: recall={rec.recall_vs_exact_at_10} != 1.0"
    )
    assert rec.boundary_tie_equivalent


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("policy", POLICIES)
def test_throughput_exact_agreement(runner, order, policy):
    """shrink=1 throughput mode must also reproduce the exact top-k."""
    cfg = RunConfig(
        dataset="synthetic",
        method="wide_block_maxsim_bond",
        dimension_order=order,
        threshold_policy=policy,
        k=10,
        shrink=1.0,
    )
    rec = runner.throughput_mode(cfg, n_repeats=2)
    assert rec.recall_vs_exact_at_10 == 1.0, (
        f"throughput/{order}/{policy}: recall={rec.recall_vs_exact_at_10} != 1.0"
    )
    assert rec.boundary_tie_equivalent


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("policy", POLICIES)
def test_accounting_exact_agreement_multigroup(runner_multigroup, order, policy):
    """Same gate on a corpus spanning multiple wide vectorgroups, so staged
    finalization across groups (Stage 1 §4.4) is exercised."""
    cfg = RunConfig(
        dataset="synthetic",
        method="wide_block_maxsim_bond",
        dimension_order=order,
        threshold_policy=policy,
        k=10,
        shrink=1.0,
    )
    rec = runner_multigroup.accounting_mode(cfg)
    assert rec.recall_vs_exact_at_10 == 1.0, (
        f"multigroup accounting/{order}/{policy}: recall={rec.recall_vs_exact_at_10} != 1.0"
    )
    assert rec.boundary_tie_equivalent


def test_accounting_reports_cost_but_not_time(runner):
    """Accounting mode reports cells/pruning, not wall-clock (Stage 1 §6)."""
    cfg = RunConfig(
        dataset="synthetic", method="wide_block_maxsim_bond", dimension_order="natural",
        threshold_policy="self_bound", k=10, shrink=1.0,
    )
    rec = runner.accounting_mode(cfg)
    assert rec.cells_scanned_pct is not None
    assert rec.pruned_docs_pct is not None
    assert rec.ms_per_query is None
    assert rec.qps is None


def test_throughput_reports_time_but_not_cost(runner):
    """Throughput mode reports wall-clock, not accounting cells (Stage 1 §6)."""
    cfg = RunConfig(
        dataset="synthetic", method="wide_block_maxsim_bond", dimension_order="natural",
        threshold_policy="self_bound", k=10, shrink=1.0,
    )
    rec = runner.throughput_mode(cfg, n_repeats=2)
    assert rec.ms_per_query is not None
    assert rec.qps is not None
    assert rec.cells_scanned_pct is None


def test_accounting_reports_tokens_pruned_pct(runner):
    """Accounting mode surfaces stats[2] (tokens_pruned) as tokens_pruned_pct
    (Stage 1 §5.3); throughput mode leaves it None (accounting-only metric)."""
    cfg = RunConfig(
        dataset="synthetic", method="wide_block_maxsim_bond", dimension_order="natural",
        threshold_policy="self_bound", k=10, shrink=1.0,
    )
    rec = runner.accounting_mode(cfg)
    assert rec.tokens_pruned_pct is not None
    assert 0.0 <= rec.tokens_pruned_pct <= 100.0
    assert rec.shrink == 1.0

    rec_t = runner.throughput_mode(cfg, n_repeats=2)
    assert rec_t.tokens_pruned_pct is None
    assert rec_t.shrink == 1.0


def test_accounting_populates_block_live_curves(runner):
    """Stage 2 e02 hooks: accounting_mode() on the wide-block path populates
    the per-fetch-boundary live-doc/live-token side channel."""
    cfg = RunConfig(
        dataset="synthetic", method="wide_block_maxsim_bond", dimension_order="natural",
        threshold_policy="self_bound", k=10, shrink=1.0,
    )
    runner.last_block_doc_live = None
    runner.last_block_token_live = None
    runner.accounting_mode(cfg)
    assert runner.last_block_doc_live is not None
    assert runner.last_block_token_live is not None
    assert runner.last_block_doc_live.shape == runner.last_block_token_live.shape
    assert np.all(runner.last_block_doc_live >= 0.0)
    assert np.all(runner.last_block_token_live >= 0.0)


def test_oracle_path_does_not_populate_block_live_curves(runner):
    """The unmodified per-document oracle path must not touch the wide-block
    side channel (dispatch separation)."""
    runner.last_block_doc_live = None
    runner.last_block_token_live = None
    cfg = RunConfig(
        dataset="synthetic", method="bond_pdx_maxsim_exact_safe", dimension_order="natural",
        threshold_policy="exact_safe_topk", k=10, shrink=1.0,
    )
    runner.accounting_mode(cfg)
    assert runner.last_block_doc_live is None
    assert runner.last_block_token_live is None


# ---------------------------------------------------------------------------
# pack_corpus_wide unit tests
# ---------------------------------------------------------------------------

def _random_corpus(seed: int, n_docs: int, D: int, doc_len_range=(3, 12)):
    rng = np.random.default_rng(seed)
    docs, starts, cur = [], [], 0
    for _ in range(n_docs):
        nd = int(rng.integers(*doc_len_range))
        starts.append(cur)
        v = rng.standard_normal((nd, D)).astype(np.float32)
        docs.append(v)
        cur += nd
    flat = np.concatenate(docs, axis=0).astype(np.float32)
    doc_starts = np.array(starts, dtype=np.int64)
    return flat, doc_starts


def test_pack_corpus_wide_round_trip():
    """Reconstructing token vectors from the wide dim-major layout must
    exactly reproduce the original token-major corpus."""
    flat, doc_starts = _random_corpus(seed=1, n_docs=25, D=12, doc_len_range=(2, 15))
    group_data, group_offsets, doc_offsets, group_doc_starts = pack_corpus_wide(
        flat, doc_starts, target_group_tokens=20
    )
    T, D = flat.shape

    assert doc_offsets[-1] == T
    assert group_offsets[-1] == T
    assert group_doc_starts[-1] == len(doc_starts)
    assert group_offsets[0] == 0
    assert group_doc_starts[0] == 0

    n_groups = len(group_offsets) - 1
    for g in range(n_groups):
        g0, g1 = int(group_offsets[g]), int(group_offsets[g + 1])
        G = g1 - g0
        chunk = group_data[g0 * D:(g0 + G) * D].reshape(D, G).T
        assert np.array_equal(chunk, flat[g0:g1]), f"group {g} round-trip mismatch"


def test_pack_corpus_wide_doc_boundary_preservation():
    """Every document's tokens land entirely inside exactly one group (never
    split across a group boundary)."""
    flat, doc_starts = _random_corpus(seed=2, n_docs=40, D=8, doc_len_range=(2, 10))
    _, group_offsets, doc_offsets, group_doc_starts = pack_corpus_wide(
        flat, doc_starts, target_group_tokens=15
    )
    n_docs = len(doc_starts)
    n_groups = len(group_offsets) - 1

    for g in range(n_groups):
        d0, d1 = int(group_doc_starts[g]), int(group_doc_starts[g + 1])
        if d0 == d1:
            continue
        assert doc_offsets[d0] >= group_offsets[g]
        assert doc_offsets[d1] <= group_offsets[g + 1]

    # Every document id appears in exactly one group's [d0, d1) range.
    covered = np.zeros(n_docs, dtype=bool)
    for g in range(n_groups):
        d0, d1 = int(group_doc_starts[g]), int(group_doc_starts[g + 1])
        assert not covered[d0:d1].any(), f"doc range re-covered by group {g}"
        covered[d0:d1] = True
    assert covered.all()


def test_pack_corpus_wide_oversized_doc_gets_own_group():
    """A document larger than target_group_tokens must get its own group."""
    rng = np.random.default_rng(3)
    small_docs = [rng.standard_normal((4, 6)).astype(np.float32) for _ in range(3)]
    big_doc = rng.standard_normal((50, 6)).astype(np.float32)
    more_small_docs = [rng.standard_normal((4, 6)).astype(np.float32) for _ in range(3)]

    docs = small_docs + [big_doc] + more_small_docs
    starts, cur = [], 0
    for d in docs:
        starts.append(cur)
        cur += d.shape[0]
    flat = np.concatenate(docs, axis=0).astype(np.float32)
    doc_starts = np.array(starts, dtype=np.int64)

    big_doc_id = len(small_docs)   # index of the oversized document

    _, group_offsets, doc_offsets, group_doc_starts = pack_corpus_wide(
        flat, doc_starts, target_group_tokens=10
    )
    n_groups = len(group_offsets) - 1

    found_own_group = False
    for g in range(n_groups):
        d0, d1 = int(group_doc_starts[g]), int(group_doc_starts[g + 1])
        if d1 - d0 == 1 and d0 == big_doc_id:
            found_own_group = True
            G = int(group_offsets[g + 1] - group_offsets[g])
            assert G == big_doc.shape[0]
    assert found_own_group, "oversized document did not get an isolated group"
