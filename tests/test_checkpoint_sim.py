"""R2 instrument-alignment gate: NumPy checkpoint simulator vs fused kernel.

The simulator (bondmaxsim.oracle.checkpoint_sim) must reproduce the fused
doc-level BOND kernel's pruning decisions under a fixed (oracle) tau — that
is the validation the research plan requires before the simulator's cells%
can be used to PREDICT fused wall-clock savings (e08).

Exactness caveat: the kernel accumulates fp32 per dimension segment while
NumPy matmuls use a different summation order, so documents whose UB sits
within fp32 noise of tau can flip.  The gate therefore allows a small count
tolerance but requires agreement on the overwhelming majority of decisions.

Requires the compiled kernel; skips cleanly if absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from bondmaxsim.data.packing import build_qcum
from bondmaxsim.oracle.checkpoint_sim import simulate_fused_doc_pruning
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores

pytest.importorskip("bondmaxsim.kernels.fused_panel")
from bondmaxsim.kernels.fused_panel import (  # noqa: E402
    load_fused_panel_kernel,
    run_fused_panel_bond,
)

try:
    LIB = load_fused_panel_kernel()
except FileNotFoundError:
    pytest.skip(
        "fused_panel_maxsim.so not built; run `make -C cpp/fused_panel_maxsim`",
        allow_module_level=True,
    )


def _prunable_corpus(seed: int, n_docs: int, D: int):
    """Corpus with energy concentrated in early dims so mid-scan checkpoints
    actually prune (same construction as the fused-panel gate fixture)."""
    rng = np.random.default_rng(seed)
    docs, starts, cur = [], [], 0
    for _ in range(n_docs):
        n = int(rng.integers(1, 40))
        starts.append(cur)
        v = rng.standard_normal((n, D)).astype(np.float32)
        docs.append(v)
        cur += n
    flat = np.concatenate(docs, axis=0)
    flat[:, D // 2:] *= 0.05
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    q = rng.standard_normal((7, D)).astype(np.float32)
    q[:, D // 2:] *= 0.05
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return flat.astype(np.float32), np.array(starts, dtype=np.int64), q


@pytest.mark.parametrize("checkpoints", [(32,), (32, 48), (16, 32, 48)])
def test_simulator_matches_kernel_doc_prune_counts(checkpoints):
    from bondmaxsim.testbed.packing_cache import PackingCache

    flat, doc_starts, q = _prunable_corpus(97, 400, 64)
    packing = PackingCache(flat, doc_starts)
    panel_data, group_offsets, doc_offsets, group_doc_starts, Q_eff, order = (
        packing.dispatch_order_panel(q, "natural")
    )
    Qcum = build_qcum(Q_eff, order)
    exact = exact_maxsim_scores(q, flat, doc_starts)
    k = 5
    tau = float(np.sort(exact)[-k]) - 1e-3   # oracle policy incl. its eps margin

    _, _, stats = run_fused_panel_bond(
        LIB, panel_data, group_offsets, doc_offsets, group_doc_starts,
        Q_eff, order, Qcum, shrink=1.0, tau_seed=tau, K=k, n_threads=1,
        checkpoints=np.asarray(checkpoints, dtype=np.uint32),
    )
    kernel_pruned = int(stats[1])

    sim = simulate_fused_doc_pruning(
        Q_eff, flat, doc_starts, order, Qcum, checkpoints, tau, shrink=1.0,
    )

    assert kernel_pruned > 0, "fixture must actually prune for the gate to bite"
    # fp32 summation-order noise may flip near-tie docs; allow 1% of the corpus.
    assert abs(sim.docs_pruned_total - kernel_pruned) <= max(4, 0.01 * len(doc_starts)), (
        f"simulator {sim.docs_pruned_total} vs kernel {kernel_pruned} docs pruned "
        f"(checkpoints={checkpoints})"
    )


def test_simulator_no_tau_no_pruning():
    flat, doc_starts, q = _prunable_corpus(11, 60, 32)
    order = np.arange(32)
    Qcum = build_qcum(q, order)
    sim = simulate_fused_doc_pruning(
        q, flat, doc_starts, order, Qcum, (16,), tau=float("-inf"),
    )
    assert sim.docs_pruned_total == 0
    assert sim.cells_scanned_pct == pytest.approx(100.0)
    assert sim.cells_scanned_pct_padded == pytest.approx(100.0)


def test_simulator_cells_account_for_skipped_dims():
    """A doc pruned at checkpoint c must contribute c/D of its dense cells."""
    flat, doc_starts, q = _prunable_corpus(13, 200, 64)
    from bondmaxsim.testbed.packing_cache import PackingCache
    packing = PackingCache(flat, doc_starts)
    _, _, _, _, Q_eff, order = packing.dispatch_order_panel(q, "natural")
    Qcum = build_qcum(Q_eff, order)
    exact = exact_maxsim_scores(q, flat, doc_starts)
    tau = float(np.sort(exact)[-5]) - 1e-3
    sim = simulate_fused_doc_pruning(Q_eff, flat, doc_starts, order, Qcum, (32,), tau)
    if sim.docs_pruned_total == 0:
        pytest.skip("fixture did not prune; covered by the count-match gate")
    doc_len = np.diff(np.append(doc_starts, flat.shape[0]))
    T = doc_len.sum()
    expected = 100.0 * (
        (np.where(sim.prune_stage >= 0, 32, 64) * doc_len).sum() / (64.0 * T)
    )
    assert sim.cells_scanned_pct == pytest.approx(expected)
