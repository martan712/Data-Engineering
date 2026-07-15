"""Blocking Stage 2 gate: shrink=1 exact-agreement (recall==1.0).

Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §2.5, §8 items 2-3.
At shrink=1 the pruning kernel must be exact-safe: its top-k document set must
agree perfectly with the NumPy exact-MaxSim oracle, for every dimension order
(natural, bond, pca) and in both accounting and throughput modes.  If this test
fails, no downstream mechanism/integration result is trustworthy.

Requires the compiled kernel (cpp/per_document_oracle/per_document_oracle.so).
Build it with `make -C cpp/per_document_oracle` or `./setup.sh`; the test skips
cleanly if the library is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from bondmaxsim.testbed.runner import Runner, RunConfig

# Skip the whole module if the kernel library has not been built.
pytest.importorskip("bondmaxsim.kernels.per_document")
from bondmaxsim.kernels.per_document import load_per_document_oracle  # noqa: E402

try:
    load_per_document_oracle()
except FileNotFoundError:
    pytest.skip(
        "per_document_oracle.so not built; run `make -C cpp/per_document_oracle`",
        allow_module_level=True,
    )

ORDERS = ["natural", "bond", "pca"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_corpus(seed: int, n_docs: int, D: int):
    """Build a unit-norm synthetic corpus and query set."""
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


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", ORDERS)
def test_accounting_exact_agreement(runner, order):
    """shrink=1 accounting mode must reproduce the exact top-k for every order."""
    cfg = RunConfig(
        dataset="synthetic",
        method="bond_pdx_maxsim_exact_safe",
        dimension_order=order,
        threshold_policy="exact_safe_topk",
        k=10,
        shrink=1.0,
    )
    rec = runner.accounting_mode(cfg)
    assert rec.recall_vs_exact_at_10 == 1.0, (
        f"accounting/{order}: recall={rec.recall_vs_exact_at_10} != 1.0"
    )
    assert rec.boundary_tie_equivalent
    assert rec.agreement_failure_codes == []


@pytest.mark.parametrize("order", ORDERS)
def test_throughput_exact_agreement(runner, order):
    """shrink=1 throughput mode must also reproduce the exact top-k."""
    cfg = RunConfig(
        dataset="synthetic",
        method="bond_pdx_maxsim_exact_safe",
        dimension_order=order,
        threshold_policy="exact_safe_topk",
        k=10,
        shrink=1.0,
    )
    rec = runner.throughput_mode(cfg, n_repeats=2)
    assert rec.recall_vs_exact_at_10 == 1.0, (
        f"throughput/{order}: recall={rec.recall_vs_exact_at_10} != 1.0"
    )
    assert rec.boundary_tie_equivalent
    assert rec.agreement_failure_codes == []


def test_accounting_reports_cost_but_not_time(runner):
    """Accounting mode reports cells/pruning, not wall-clock (Stage 1 §6)."""
    cfg = RunConfig(
        dataset="synthetic", method="m", dimension_order="natural",
        threshold_policy="exact_safe_topk", k=10, shrink=1.0,
    )
    rec = runner.accounting_mode(cfg)
    assert rec.cells_scanned_pct is not None
    assert rec.pruned_docs_pct is not None
    assert rec.ms_per_query is None
    assert rec.qps is None


def test_throughput_reports_time_but_not_cost(runner):
    """Throughput mode reports wall-clock, not accounting cells (Stage 1 §6)."""
    cfg = RunConfig(
        dataset="synthetic", method="m", dimension_order="natural",
        threshold_policy="exact_safe_topk", k=10, shrink=1.0,
    )
    rec = runner.throughput_mode(cfg, n_repeats=2)
    assert rec.ms_per_query is not None
    assert rec.qps is not None
    assert rec.cells_scanned_pct is None


def test_accounting_reports_tokens_pruned_pct(runner):
    """Accounting mode surfaces stats[2] (tokens_pruned) as tokens_pruned_pct
    (Stage 1 §5.3); throughput mode leaves it None (accounting-only metric)."""
    cfg = RunConfig(
        dataset="synthetic", method="m", dimension_order="natural",
        threshold_policy="exact_safe_topk", k=10, shrink=1.0,
    )
    rec = runner.accounting_mode(cfg)
    assert rec.tokens_pruned_pct is not None
    assert 0.0 <= rec.tokens_pruned_pct <= 100.0
    assert rec.shrink == 1.0

    rec_t = runner.throughput_mode(cfg, n_repeats=2)
    assert rec_t.tokens_pruned_pct is None
    assert rec_t.shrink == 1.0
