"""Stage 3 R12c: corrected exact-safe margins with an INTERLEAVED dense baseline.

Single responsibility: re-measure the e08/e09 exact-safe wall-clock comparison
with the dense baseline timed INTERLEAVED with the pruning arms (round-robin,
best-of-N per arm) instead of standalone.  R12b showed the standalone baseline
inflated arguana's margin (−19% MT reported vs ~+8% interleaved) because a
thermally unlucky baseline window shifts every arm by a constant.  This driver
produces the honest G1 numbers on all four datasets.

Same 50-query subsample (seed 42) as e08, so the ONLY difference from e08 is the
interleaving — the margins here supersede e08's for the G1 gate.  Tight bound
(the R12a-adopted default), natural order (the e08 winner).  Checkpoint sets:
{32} = zero-prune overhead control, plus the winners {112} and {64,112}.

Timing only (recall is 1.0 on every arm by e08/e09; the raw kernels are called
directly so the exact-top-k recompute does not dominate and interleaving stays
tight).

Datasets  : scifact, nfcorpus, arguana, scidocs
Threads   : one setting per run (arg): 0 = all cores (the G1 baseline), 1 = 1T
Queries   : 50 (seed 42, == e08)
k         : 10
REP       : best-of-8, interleaved

Output:
  results/json/stage3_mechanism_r12c_interleaved_exact_safe_<mt|1t>.json

Usage:
    uv run python -m experiments.stage3_mechanism.r12c_interleaved_exact_safe \
        <threads:0|1> [dataset ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.data.packing import build_qcum
from bondmaxsim.kernels.fused_panel import run_fused_panel_bond, run_fused_panel_brute
from bondmaxsim.testbed.runner import Runner, RunConfig
from bondmaxsim.testbed.thresholds import resolve_tau_seed

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
CHECKPOINT_SETS = [(32,), (112,), (64, 112)]
N_QUERIES = 50
QUERY_SEED = 42
K_TOP = 10
REP = 8

RESULTS_JSON = REPO_ROOT / "results" / "json"


def _subsample_queries(queries, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


def _e08_reference(dataset, cps):
    """Standalone e08 margin + simulator cells-saved for (natural, cps), for cross-check."""
    p = RESULTS_JSON / f"stage3_mechanism_e08_checkpoint_ablation_{dataset}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    b = d["dense_fused_baseline"]
    for a in d["arms"]:
        if a["dimension_order"] == "natural" and a["checkpoints"] == list(cps):
            return {
                "e08_cells_saved_pct": 100.0 - a["sim_cells_scanned_pct_padded"],
                "e08_margin_1t_pct": (b["ms_per_query_1t"] - a["ms_per_query_1t"]) / b["ms_per_query_1t"] * 100,
                "e08_margin_mt_pct": (b["ms_per_query_mt"] - a["ms_per_query_mt"]) / b["ms_per_query_mt"] * 100,
                "e08_prune_pct": a["pruned_docs_pct_fused"],
            }
    return None


def run_dataset(dataset, nt):
    flat, starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    runner = Runner(flat, starts, queries)
    pk = runner._packing
    lib = runner._get_fused_lib()
    m_mean = float(np.mean([q.shape[0] for q in queries]))
    print(f"\n=== {dataset}: {len(starts)} docs, {len(queries)} queries, "
          f"m={m_mean:.1f}, threads={'all' if nt == 0 else nt} ===")

    pdat, goff, doff, gds, _ = pk._get_panel_packing()
    Qs = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]

    def run_dense():
        for Q in Qs:
            run_fused_panel_brute(lib, pdat, goff, doff, gds, Q, K_TOP, n_threads=nt)

    def make_bond(cps):
        cfg = RunConfig(dataset=dataset, method="fused_panel_maxsim_bond",
                        dimension_order="natural", threshold_policy="oracle",
                        k=K_TOP, shrink=1.0, checkpoints=cps)
        prepared = []
        for q in queries:
            pd, go, do, gd, Qe, order = pk.dispatch_order_panel(q, "natural")
            prepared.append((pd, go, do, gd, Qe, order, build_qcum(Qe, order),
                             resolve_tau_seed(cfg, q, order, "natural", pk)))
        arr = np.asarray(cps, dtype=np.uint32)

        def _run():
            for pd, go, do, gd, Qe, order, Qc, tau in prepared:
                run_fused_panel_bond(lib, pd, go, do, gd, Qe, order, Qc,
                                     shrink=1.0, tau_seed=tau, K=K_TOP, n_threads=nt,
                                     level="doc", checkpoints=arr, bound="tight")
        return _run

    bond = {cps: make_bond(cps) for cps in CHECKPOINT_SETS}

    def ms(fn):
        t0 = time.perf_counter(); fn()
        return (time.perf_counter() - t0) / len(Qs) * 1e3

    # Warm all, then interleave round-robin (dense first each loop).
    run_dense()
    for r in bond.values():
        r()
    best_dense = float("inf")
    best_bond = {cps: float("inf") for cps in CHECKPOINT_SETS}
    for _ in range(REP):
        best_dense = min(best_dense, ms(run_dense))
        for cps in CHECKPOINT_SETS:
            best_bond[cps] = min(best_bond[cps], ms(bond[cps]))

    arms = []
    print(f"  dense-brute (interleaved) = {best_dense:.3f} ms/q")
    for cps in CHECKPOINT_SETS:
        b = best_bond[cps]
        margin = (best_dense - b) / best_dense * 100
        ref = _e08_reference(dataset, cps)
        e08m = ref[f"e08_margin_{'mt' if nt == 0 else '1t'}_pct"] if ref else float("nan")
        cells = ref["e08_cells_saved_pct"] if ref else float("nan")
        arms.append({"checkpoints": list(cps), "ms_per_query": b,
                     "margin_vs_interleaved_dense_pct": margin,
                     "e08_standalone_margin_pct": e08m,
                     "e08_cells_saved_pct": cells,
                     "e08_ref": ref})
        print(f"  C={str(list(cps)):14} bond tight = {b:8.3f} ms/q   "
              f"margin(interleaved) = {margin:+6.1f}%   "
              f"[e08 standalone {e08m:+6.1f}%,  cells {cells:+5.1f}%]")

    return {"dataset": dataset, "n_docs": len(starts), "n_queries": len(queries),
            "query_m_mean": m_mean, "dense_interleaved_ms_per_query": best_dense,
            "arms": arms}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1"):
        print("usage: ... r12c_interleaved_exact_safe <threads:0|1> [dataset ...]")
        sys.exit(1)
    nt = int(sys.argv[1])
    datasets = sys.argv[2:] if len(sys.argv) > 2 else DATASETS

    t0 = time.perf_counter()
    results = [run_dataset(ds, nt) for ds in datasets]

    payload = {
        "experiment": "r12c_interleaved_exact_safe",
        "purpose": "corrected exact-safe G1 margins with an interleaved dense baseline (R12b/R12c)",
        "threads": "all_cores" if nt == 0 else "single",
        "n_queries": N_QUERIES, "query_seed": QUERY_SEED, "k": K_TOP,
        "bound": "tight", "order": "natural", "checkpoint_sets": [list(c) for c in CHECKPOINT_SETS],
        "rep_best_of": REP, "results": results,
    }
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    tag = "mt" if nt == 0 else "1t"
    out = RESULTS_JSON / f"stage3_mechanism_r12c_interleaved_exact_safe_{tag}.json"
    # Merge by dataset so the sweep can be run in batches (slow 1T datasets apart).
    if out.exists():
        prev = json.loads(out.read_text()).get("results", [])
        kept = [r for r in prev if r["dataset"] not in {r2["dataset"] for r2 in results}]
        order = {ds: i for i, ds in enumerate(DATASETS)}
        payload["results"] = sorted(kept + results, key=lambda r: order.get(r["dataset"], 99))
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON: {out}\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
