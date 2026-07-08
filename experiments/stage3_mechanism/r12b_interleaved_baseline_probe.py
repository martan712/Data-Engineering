"""R12b probe: interleaved dense-vs-BOND timing (the standalone-baseline check).

Single responsibility: settle whether e08/e09's exact-safe wall-clock "wins"
are real or an artifact of timing the dense baseline STANDALONE (once, in
isolation) while the pruning arms are timed in separate windows.  A thermally
unlucky window then inflates (or deflates) every arm's saving by a constant.

Method: time dense-brute and the BOND doc-kernel (natural order, tight bound)
ROUND-ROBIN, one rep each per loop, best-of-N per arm — so slow thermal drift
hits both equally and cancels in the ratio.  Timing only; recall is already
known 1.0 from e08/e09, so we call the raw kernels directly (no per-rep exact
top-k recompute, which is what makes the runner modes too slow to interleave).

Arms per checkpoint set: C={32} (or first set) = the ZERO-PRUNE control (its
offset vs dense is the pure checkpoint overhead / measurement artifact), plus
the real-pruning late sets.

Finding (2026-07-04, arguana): standalone e08 reported −29% (1T) / −19.6% (MT)
at C={112}.  Interleaved, the 1T zero-prune offset collapses +19% -> +0.2%
(standalone dense 179.6 ms/q was throttled; interleaved 149) and the MT saving
corrects to ~+8%, matching the e09 run's independent +8.0%.  Conclusion:
wall-clock tracks the ~-12% cells prediction once the baseline is fair; there
is no disproportionately-expensive final segment.  See the plan doc R12(b) and
the `interleaved-baseline-measurement` memory.

Usage:
    uv run python -m experiments.stage3_mechanism.r12b_interleaved_baseline_probe \
        [dataset] [n_queries] [n_threads]   # n_threads: 0 = all cores, 1 = single
"""
from __future__ import annotations

import sys
import time

import numpy as np

from bondmaxsim.data.loader import load_dataset
from bondmaxsim.data.packing import build_qcum
from bondmaxsim.kernels.fused_panel import run_fused_panel_bond, run_fused_panel_brute
from bondmaxsim.testbed.runner import Runner, RunConfig
from bondmaxsim.testbed.thresholds import resolve_tau_seed

K_TOP = 10
REP = 8
CHECKPOINT_SETS = [(32,), (112,), (64, 112), (32, 64, 96, 112)]


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "arguana"
    nq = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = all cores

    flat, starts, queries = load_dataset(dataset)
    queries = queries[:nq]
    runner = Runner(flat, starts, queries)
    pk = runner._packing
    lib = runner._get_fused_lib()
    m_mean = float(np.mean([q.shape[0] for q in queries]))
    print(f"{dataset}: {len(starts)} docs, nq={len(queries)}, m(query) mean={m_mean:.1f}, "
          f"threads={'all' if nt == 0 else nt}")

    pdat, goff, doff, gds, _ = pk._get_panel_packing()
    Qs = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]

    def run_dense() -> None:
        for Q in Qs:
            run_fused_panel_brute(lib, pdat, goff, doff, gds, Q, K_TOP, n_threads=nt)

    def make_bond(cps: tuple[int, ...]):
        cfg = RunConfig(dataset=dataset, method="fused_panel_maxsim_bond",
                        dimension_order="natural", threshold_policy="oracle",
                        k=K_TOP, shrink=1.0, checkpoints=cps)
        prepared = []
        for q in queries:
            pd, go, do, gd, Qe, order = pk.dispatch_order_panel(q, "natural")
            Qc = build_qcum(Qe, order)
            tau = resolve_tau_seed(cfg, q, order, "natural", pk)
            prepared.append((pd, go, do, gd, Qe, order, Qc, tau))
        cps_arr = np.asarray(cps, dtype=np.uint32)

        def _run() -> None:
            for pd, go, do, gd, Qe, order, Qc, tau in prepared:
                run_fused_panel_bond(lib, pd, go, do, gd, Qe, order, Qc,
                                     shrink=1.0, tau_seed=tau, K=K_TOP, n_threads=nt,
                                     level="doc", checkpoints=cps_arr, bound="tight")
        return _run

    bond_runs = {cps: make_bond(cps) for cps in CHECKPOINT_SETS}

    # Warm everything.
    run_dense()
    for r in bond_runs.values():
        r()

    def ms(fn) -> float:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) / len(Qs) * 1e3

    best_dense = float("inf")
    best_bond = {cps: float("inf") for cps in CHECKPOINT_SETS}
    for _ in range(REP):
        best_dense = min(best_dense, ms(run_dense))
        for cps in CHECKPOINT_SETS:
            best_bond[cps] = min(best_bond[cps], ms(bond_runs[cps]))

    print(f"\n  dense-brute = {best_dense:.3f} ms/q  (interleaved baseline)")
    for cps in CHECKPOINT_SETS:
        b = best_bond[cps]
        print(f"  C={str(list(cps)):16} bond tight = {b:7.3f} ms/q   "
              f"saving vs dense = {(best_dense - b) / best_dense * 100:+.1f}%")


if __name__ == "__main__":
    main()
