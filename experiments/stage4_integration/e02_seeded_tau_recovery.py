"""Stage 4 e02: how much of the oracle-tau exact-safe margin does a REALISTIC
seeded threshold recover? (R8; Stage 1 §4.4 option b — candidate seeding feeds
tau_seed.)

Single responsibility: ablate the fused BOND kernel's tau_seed policy at the
R12-winning exact-safe operating point (tight bound, natural order, C={112},
shrink=1) between:

  self_bound   : tau_seed = -inf — the kernel relies only on its thread-shared
                 rising threshold (staged finalization across groups).
  seed_partial : Stage 1 §4.4 partial-dimension seed (first 32 dims of the scan
                 order rank all docs, top 5% get exact scores).  tau VALUE is
                 realistic; its cost here is a NumPy path, so no cost is
                 charged — quality-recovery reference only.
  ivf_seed_*   : the R8 production policy — a cheap FAISS-IVF token-level
                 candidate pass proposes s docs, those are exactly scored, and
                 their k-th best seeds tau (always safe: subset k-th <= true
                 k-th).  Seeding COST is measured and charged separately.
  oracle       : true k-th exact score (upper bound on pruning potential; the
                 policy behind the e08/R12c G1 margins).

Every arm is exact-safe (recall gate = 1.0 enforced via tie-aware agreement);
what changes is only how early document pruning can fire.

Timing follows the R12b/R12c methodology: the dense fused baseline is timed
INTERLEAVED with all arms, round-robin, best-of-8 (see
results/json/stage3_mechanism_r12c_* and the plan doc R12 record) — never
standalone.

Datasets  : scifact, nfcorpus, arguana, scidocs
Threads   : one setting per run (arg): 0 = all cores (G1 baseline), 1 = 1T
Queries   : 50 (seed 42, == e08/r12c)
k         : 10

Output (merged per dataset across thread tags):
  results/json/stage4_integration_e02_seeded_tau_recovery_<dataset>.json

Usage:
    uv run python -m experiments.stage4_integration.e02_seeded_tau_recovery \
        <threads:0|1> [dataset ...]
"""
from __future__ import annotations

import contextlib
import json
import sys
import time

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.data.packing import build_qcum
from bondmaxsim.kernels.fused_panel import run_fused_panel_bond, run_fused_panel_brute
from bondmaxsim.oracle.agreement import validate_boundary_tie_equivalence
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores
from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline
from bondmaxsim.testbed.runner import Runner, RunConfig
from bondmaxsim.testbed.thresholds import resolve_tau_seed
from bondmaxsim.threshold.policies import candidate_seed_threshold

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
CHECKPOINTS = (112,)          # R12a/R12c winner
ORDER = "natural"
BOUND = "tight"
N_QUERIES = 50
QUERY_SEED = 42
K_TOP = 10
REP = 8                       # interleaved best-of-N (kernel arms)
REP_SEED = 3                  # best-of-N for the seed-cost loop

# Safety margin for cross-implementation fp32 noise at near-tie thresholds
# (same rationale and value as bondmaxsim.testbed.thresholds._TAU_SEED_EPS).
TAU_EPS = 1e-3

# (nprobe, k_token, s): IVF probes per token, tokens retrieved per query
# token, and candidate docs exactly scored for the seed.
IVF_ARMS = {
    "ivf_seed_cheap":  dict(nprobe=4,  k_token=32,  s=10),
    "ivf_seed_strong": dict(nprobe=16, k_token=128, s=100),
}
ARM_NAMES = ["self_bound", "seed_partial", "ivf_seed_cheap", "ivf_seed_strong", "oracle"]

RESULTS_JSON = REPO_ROOT / "results" / "json"
FAISS_CACHE = REPO_ROOT / "data" / "faiss_indexes"


def _subsample_queries(queries, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


def _thread_guard(nt: int):
    """Pin BLAS pools to nt threads when nt > 0 (seed-cost loops use NumPy)."""
    if nt <= 0:
        return contextlib.nullcontext()
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=nt, user_api="blas")


def _ivf_tau_and_cost(ivf, queries, k, params, nt):
    """Per-query IVF-seeded tau + the measured seeding cost (ms/q, best-of-N).

    The seed pipeline is pinned to ONE thread internally regardless of the
    kernel-arm thread setting: its work items (nprobe<=16 list scans per token,
    a <=100-doc exact rerank) are so small that OMP/BLAS fan-out costs more
    than it saves (measured: 33.6 vs 8.4 ms/q on scifact at 12 threads vs 1),
    and a production seeder would make the same choice.  Recorded in the JSON
    as seed_threads=1.
    """
    import faiss

    faiss.omp_set_num_threads(1)
    ivf.index.nprobe = params["nprobe"]

    def _pass():
        taus = []
        for q in queries:
            _, scores, _ = ivf.topk(q, k=k, candidate_budget=params["s"],
                                    k_token=params["k_token"])
            taus.append(candidate_seed_threshold(scores, k) - TAU_EPS)
        return taus

    try:
        with _thread_guard(1):
            taus = _pass()                 # warm + the values we use
            best = float("inf")
            for _ in range(REP_SEED):
                t0 = time.perf_counter()
                _pass()
                best = min(best, (time.perf_counter() - t0) / len(queries) * 1e3)
    finally:
        faiss.omp_set_num_threads(nt if nt > 0 else faiss.omp_get_max_threads())
    return taus, best


def run_dataset(dataset, nt):
    flat, starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    runner = Runner(flat, starts, queries)
    pk = runner._packing
    lib = runner._get_fused_lib()
    tag = "mt" if nt == 0 else "1t"
    print(f"\n=== e02 {dataset}: {len(starts)} docs, {len(queries)} queries, "
          f"threads={'all' if nt == 0 else nt} ===")

    exact_full = [exact_maxsim_scores(q, flat, starts) for q in queries]
    exact = [topk_from_scores(scores, K_TOP) for scores in exact_full]

    # IVF seeding index (shared by both ivf arms; k_token differs per arm).
    ivf = FaissIVFBaseline(flat, starts)
    t0 = time.perf_counter()
    ivf.build(cache_path=str(FAISS_CACHE / f"{dataset}.faiss"))
    print(f"  faiss index ready in {time.perf_counter() - t0:.1f}s (n_lists={ivf.n_lists})")

    # ---- per-arm tau values (and seed costs where the policy is realistic)
    cfg_base = dict(dataset=dataset, method="fused_panel_maxsim_bond",
                    dimension_order=ORDER, k=K_TOP, shrink=1.0,
                    checkpoints=CHECKPOINTS)
    taus: dict[str, list[float]] = {}
    seed_cost: dict[str, float | None] = {}
    orders = [pk.dispatch_order_panel(q, ORDER) for q in queries]

    taus["self_bound"] = [float("-inf")] * len(queries)
    seed_cost["self_bound"] = 0.0

    cfg = RunConfig(threshold_policy="seed", **cfg_base)
    taus["seed_partial"] = [
        resolve_tau_seed(cfg, q, od[5], ORDER, pk) for q, od in zip(queries, orders)
    ]
    seed_cost["seed_partial"] = None   # NumPy path; value-reference only

    for name, params in IVF_ARMS.items():
        t, c = _ivf_tau_and_cost(ivf, queries, K_TOP, params, nt)
        taus[name] = t
        seed_cost[name] = c

    cfg = RunConfig(threshold_policy="oracle", **cfg_base)
    taus["oracle"] = [
        resolve_tau_seed(cfg, q, od[5], ORDER, pk) for q, od in zip(queries, orders)
    ]
    seed_cost["oracle"] = None         # not realizable

    # ---- prepared kernel args per query (shared across arms)
    prepared = []
    for q, od in zip(queries, orders):
        pd, go, do, gd, Qe, order = od
        prepared.append((pd, go, do, gd, Qe, order, build_qcum(Qe, order)))
    arr = np.asarray(CHECKPOINTS, dtype=np.uint32)

    def make_run(arm):
        tvals = taus[arm]

        def _run(collect=None):
            for (pd, go, do, gd, Qe, order, Qc), tau in zip(prepared, tvals):
                ids, _, stats = run_fused_panel_bond(
                    lib, pd, go, do, gd, Qe, order, Qc,
                    shrink=1.0, tau_seed=tau, K=K_TOP, n_threads=nt,
                    level="doc", checkpoints=arr, bound=BOUND)
                if collect is not None:
                    collect.append((ids, stats))
        return _run

    runs = {arm: make_run(arm) for arm in ARM_NAMES}
    Qs = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]
    pdat, goff, doff, gds, _ = pk._get_panel_packing()

    def run_dense():
        for Q in Qs:
            run_fused_panel_brute(lib, pdat, goff, doff, gds, Q, K_TOP, n_threads=nt)

    # ---- recall gate + pruning stats (untimed pass per arm)
    n_docs = len(starts)
    gate_stats = {}
    for arm in ARM_NAMES:
        got: list = []
        runs[arm](collect=got)
        agreements = [
            validate_boundary_tie_equivalence(
                ids, e_ids, e_scores, k=K_TOP, num_documents=n_docs,
                exact_scores_by_id=full_scores,
            )
            for (ids, _), (e_ids, e_scores), full_scores
            in zip(got, exact, exact_full)
        ]
        rec = min(result.recall_vs_oracle_set for result in agreements)
        pruned = float(np.mean([s[1] for _, s in got])) / n_docs * 100.0
        if not all(result.exact_gate_passed for result in agreements):
            failures = sorted({c for r in agreements for c in r.failure_codes})
            raise RuntimeError(
                f"verified exact gate failed: {dataset} arm={arm} failures={failures}"
            )
        gate_stats[arm] = dict(
            recall=rec,
            strict_top_k_set_equal=all(r.strict_top_k_set_equal for r in agreements),
            boundary_tie_equivalent=all(r.exact_gate_passed for r in agreements),
            agreement_failure_codes=[],
            pruned_docs_pct=pruned,
        )

    # ---- interleaved timing: dense first, then every arm, round-robin
    def ms(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) / len(Qs) * 1e3

    run_dense()
    for arm in ARM_NAMES:
        runs[arm]()
    best_dense = float("inf")
    best = {arm: float("inf") for arm in ARM_NAMES}
    for _ in range(REP):
        best_dense = min(best_dense, ms(run_dense))
        for arm in ARM_NAMES:
            best[arm] = min(best[arm], ms(runs[arm]))

    # ---- assemble
    oracle_margin = (best_dense - best["oracle"]) / best_dense * 100.0
    oracle_pruned = gate_stats["oracle"]["pruned_docs_pct"]
    arms_out = []
    print(f"  dense-brute (interleaved) = {best_dense:.3f} ms/q")
    for arm in ARM_NAMES:
        kern = best[arm]
        margin = (best_dense - kern) / best_dense * 100.0
        sc = seed_cost[arm]
        total = kern + sc if sc is not None else None
        total_margin = ((best_dense - total) / best_dense * 100.0) if total is not None else None
        pruned = gate_stats[arm]["pruned_docs_pct"]
        tau_gap = float(np.mean([
            (to - ta) for to, ta in zip(taus["oracle"], taus[arm]) if np.isfinite(ta)
        ])) if any(np.isfinite(t) for t in taus[arm]) else None
        arms_out.append({
            "arm": arm,
            "threshold_policy": arm,
            "ivf_params": IVF_ARMS.get(arm),
            "recall_vs_exact_at_10": gate_stats[arm]["recall"],
            "strict_top_k_set_equal": gate_stats[arm]["strict_top_k_set_equal"],
            "boundary_tie_equivalent": gate_stats[arm]["boundary_tie_equivalent"],
            "agreement_failure_codes": gate_stats[arm]["agreement_failure_codes"],
            "pruned_docs_pct": pruned,
            "prune_recovery_vs_oracle": (pruned / oracle_pruned) if oracle_pruned > 0 else None,
            "ms_per_query_kernel": kern,
            "kernel_margin_vs_dense_pct": margin,
            "seed_cost_ms_per_query": sc,
            "ms_per_query_total": total,
            "total_margin_vs_dense_pct": total_margin,
            "mean_tau_gap_vs_oracle": tau_gap,
        })
        sc_str = f"{sc:6.2f}" if sc is not None else "   n/a"
        tm_str = f"{total_margin:+6.1f}%" if total_margin is not None else "    n/a"
        print(f"  {arm:<16} prune={pruned:5.1f}%  kernel={kern:8.3f} ms/q "
              f"({margin:+6.1f}%)  seed={sc_str} ms/q  total_margin={tm_str}")

    run_payload = {
        "threads": "all_cores" if nt == 0 else "single",
        "dense_interleaved_ms_per_query": best_dense,
        "oracle_kernel_margin_pct": oracle_margin,
        "arms": arms_out,
    }

    # ---- merge into the per-dataset JSON ({"mt": ..., "1t": ...})
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    out = RESULTS_JSON / f"stage4_integration_e02_seeded_tau_recovery_{dataset}.json"
    payload = {
        "experiment": "e02_seeded_tau_recovery",
        "purpose": "R8: realistic candidate-seeded tau vs oracle tau at the "
                   "R12 exact-safe operating point (Stage 1 §4.4b)",
        "dataset": dataset, "n_docs": n_docs, "n_queries": len(queries),
        "query_seed": QUERY_SEED, "k": K_TOP,
        "kernel": {"bound": BOUND, "order": ORDER, "checkpoints": list(CHECKPOINTS),
                   "shrink": 1.0},
        "rep_best_of": REP, "rep_seed_cost_best_of": REP_SEED,
        "seed_threads": 1,
        "faiss": {"n_lists": ivf.n_lists, "metric": "inner_product"},
        "runs": {},
    }
    if out.exists():
        prev = json.loads(out.read_text())
        payload["runs"] = prev.get("runs", {})
    payload["runs"][tag] = run_payload
    out.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {out}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1"):
        print("usage: ... e02_seeded_tau_recovery <threads:0|1> [dataset ...]")
        sys.exit(1)
    nt = int(sys.argv[1])
    datasets = sys.argv[2:] if len(sys.argv) > 2 else DATASETS
    t0 = time.perf_counter()
    for ds in datasets:
        run_dataset(ds, nt)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
