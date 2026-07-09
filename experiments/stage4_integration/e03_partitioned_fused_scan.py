"""Stage 4 e03: IVF-partitioned fused BOND scan — latency-recall frontier (R8).

Single responsibility: measure the combined system arm
(bondmaxsim.partitioned_scan.PartitionedFusedScan — build-time document
clusters packed per-partition in panel layout, per-query centroid probing,
fused BOND kernel over the probed partitions with a rising seeded tau)
against the exact fused dense scan, as a latency-recall frontier over the
probe width nprobe.

This is the "PDX-IVF architecture with our kernel as the bucket scanner"
design from the R8 discussion (2026-07-08): token-level IVF pipelines
(e01's faiss/plaid/pdx arms) pay candidate-generation costs exceeding the
whole fused scan at this corpus scale, while partition probing costs one
m x P GEMM — so any wall-clock win vs the dense scan appears at whatever
corpus size the frontier crosses, measurable HERE rather than only at the
deferred R7 scale.

APPROXIMATE arm: probing can lose true top-k docs in unprobed partitions.
Results are a frontier (recall_vs_exact@10 vs ms/q per nprobe), reported
separately from every exact-safe arm (Convention 4).  nprobe = P degenerates
to the exact exhaustive scan (gate-tested in tests/test_stage4_baselines.py).

Timing: all arms + the dense fused baseline interleaved round-robin,
best-of-5 (R12b/R12c methodology).  Build cost (clustering + per-partition
packing) is reported separately from query latency (Stage 5 fairness
convention, applied early).

Datasets  : scifact, nfcorpus, arguana, scidocs
Threads   : one setting per run (arg): 0 = all cores, 1 = 1T
Queries   : 50 (seed 42, == e08/r12c)
k         : 10

Output (merged per dataset across thread tags):
  results/json/stage4_integration_e03_partitioned_fused_scan_<dataset>.json
  results/figures/stage4_integration/e03_partitioned_fused_scan_<dataset>.png

Usage:
    uv run python -m experiments.stage4_integration.e03_partitioned_fused_scan \
        <threads:0|1> [dataset ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.kernels.fused_panel import load_fused_panel_kernel, run_fused_panel_brute
from bondmaxsim.oracle.agreement import recall_at_k
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_topk
from bondmaxsim.partitioned_scan import PartitionedFusedScan
from bondmaxsim.testbed.packing_cache import PackingCache

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
NPROBES = [1, 2, 4, 8, 16, 32]        # plus P (exact control), deduped
CHECKPOINTS = (112,)                   # R12a/R12c winner
BOUND = "tight"
N_QUERIES = 50
QUERY_SEED = 42
K_TOP = 10
REP = 5

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG = REPO_ROOT / "results" / "figures" / "stage4_integration"


def _subsample_queries(queries, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


def run_dataset(dataset, nt):
    flat, starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    lib = load_fused_panel_kernel()
    tag = "mt" if nt == 0 else "1t"
    print(f"\n=== e03 {dataset}: {len(starts)} docs, {len(queries)} queries, "
          f"threads={'all' if nt == 0 else nt} ===")

    exact = [exact_maxsim_topk(q, flat, starts, K_TOP) for q in queries]

    # ---- build (reported separately from query latency)
    t0 = time.perf_counter()
    index = PartitionedFusedScan(flat, starts)
    build_stats = index.build()
    build_s = time.perf_counter() - t0
    P = build_stats["n_partitions_nonempty"]
    nprobes = sorted({min(n, P) for n in NPROBES} | {P})
    print(f"  build: {build_s:.1f}s  P={P} "
          f"(mean {build_stats['docs_per_partition_mean']:.0f} docs/partition)  "
          f"nprobe sweep: {nprobes}")

    # ---- exact-safe partition-bound accounting probe (offline, untimed).
    # Answers "why not an EXACT PDX-IVF?": a skipped partition needs a valid
    # certificate UB_p = sum_i <q_i,c_p> + m*R_p < tau.  Measure, under the
    # ORACLE tau (upper bound of potential), how many partitions that
    # certificate could ever prune.
    prunable = []
    slack = []
    for q, (_, e_scores) in zip(queries, exact):
        probe = (q @ index.centroids.T).sum(axis=0)
        ub = probe + q.shape[0] * index.radii
        tau = float(e_scores[-1])          # oracle k-th best
        prunable.append(100.0 * float(np.mean(ub < tau)))
        slack.append(float(np.min(ub) - tau))
    bound_probe = {
        "prunable_partitions_pct_oracle_tau": float(np.mean(prunable)),
        "min_ub_minus_oracle_tau_mean": float(np.mean(slack)),
        "note": "UB_p = sum_i <q_i,c_p> + m*R_p (unit-norm queries); "
                "partitions with UB_p < tau could be skipped EXACTLY",
    }
    print(f"  exact-safe partition bound: prunable={bound_probe['prunable_partitions_pct_oracle_tau']:.2f}% "
          f"of partitions at ORACLE tau  (min UB - tau = {bound_probe['min_ub_minus_oracle_tau_mean']:+.1f})")

    # ---- dense fused baseline (same packing path as e01/e02)
    pk = PackingCache(flat, starts)
    pdat, goff, doff, gds, _ = pk._get_panel_packing()
    Qs = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]

    def run_dense():
        for Q in Qs:
            run_fused_panel_brute(lib, pdat, goff, doff, gds, Q, K_TOP, n_threads=nt)

    SCANNERS = ("brute", "bond")   # brute = pure IVF saving; bond adds pruning

    def make_arm(scanner, nprobe):
        def _run(collect=None):
            for q in queries:
                out = index.search(lib, q, k=K_TOP, nprobe=nprobe,
                                   checkpoints=CHECKPOINTS, bound=BOUND,
                                   n_threads=nt, scanner=scanner)
                if collect is not None:
                    collect.append(out)
        return _run

    arms = {(s, n): make_arm(s, n) for s in SCANNERS for n in nprobes}

    # ---- quality (untimed pass per arm; recall depends only on the probe
    # set, but both scanners are gated to catch any divergence)
    quality = {}
    for key, arm in arms.items():
        got: list = []
        arm(collect=got)
        recs = [recall_at_k(ids, e_ids)
                for (ids, _, _), (e_ids, _) in zip(got, exact)]
        quality[key] = {
            "recall_mean": float(np.mean(recs)),
            "recall_min": float(np.min(recs)),
            "docs_probed_pct_mean": float(np.mean(
                [s["docs_probed_pct"] for _, _, s in got])),
            "docs_pruned_pct_of_probed_mean": float(np.mean(
                [s["docs_pruned_pct_of_probed"] for _, _, s in got])),
        }

    # ---- interleaved timing
    def ms(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) / len(queries) * 1e3

    run_dense()
    for arm in arms.values():
        arm()
    best_dense = float("inf")
    best = {key: float("inf") for key in arms}
    for _ in range(REP):
        best_dense = min(best_dense, ms(run_dense))
        for key in arms:
            best[key] = min(best[key], ms(arms[key]))

    arms_out = []
    print(f"  dense-brute (interleaved) = {best_dense:.3f} ms/q")
    for s in SCANNERS:
        for n in nprobes:
            q = quality[(s, n)]
            speedup = best_dense / best[(s, n)]
            arms_out.append({
                "scanner": s, "nprobe": n, "is_full_probe": n == P,
                "ms_per_query": best[(s, n)],
                "speedup_vs_dense": speedup,
                "recall_vs_exact_at_10_mean": q["recall_mean"],
                "recall_vs_exact_at_10_min": q["recall_min"],
                "docs_probed_pct_mean": q["docs_probed_pct_mean"],
                "docs_pruned_pct_of_probed_mean": q["docs_pruned_pct_of_probed_mean"],
            })
            print(f"  [{s:<5}] nprobe={n:>3}{'*' if n == P else ' '} "
                  f"probed={q['docs_probed_pct_mean']:5.1f}%  "
                  f"recall={q['recall_mean']:.3f} (min {q['recall_min']:.2f})  "
                  f"ms/q={best[(s, n)]:8.3f}  speedup={speedup:4.2f}x")

    run_payload = {
        "threads": "all_cores" if nt == 0 else "single",
        "dense_interleaved_ms_per_query": best_dense,
        "arms": arms_out,
    }

    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    out = RESULTS_JSON / f"stage4_integration_e03_partitioned_fused_scan_{dataset}.json"
    payload = {
        "experiment": "e03_partitioned_fused_scan",
        "purpose": "R8: IVF-partitioned panel layout with the fused BOND kernel "
                   "as the bucket scanner — latency-recall frontier vs nprobe",
        "dataset": dataset, "n_docs": len(starts), "n_queries": len(queries),
        "query_seed": QUERY_SEED, "k": K_TOP,
        "kernel": {"bound": BOUND, "order": "natural",
                   "checkpoints": list(CHECKPOINTS), "shrink": 1.0},
        "partitioning": {**build_stats, "build_seconds": build_s,
                         "kmeans_niters": index.kmeans_niters,
                         "probe_score": "sum_i <q_i, c_p>"},
        "exact_safe_partition_bound": bound_probe,
        "rep_best_of": REP,
        "runs": {},
    }
    if out.exists():
        prev = json.loads(out.read_text())
        payload["runs"] = prev.get("runs", {})
    payload["runs"][tag] = run_payload
    out.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {out}")
    _save_figure(payload, dataset)


def _save_figure(payload, dataset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in ("mt", "1t") if t in payload["runs"]]
    fig, axes = plt.subplots(1, len(tags), figsize=(6.0 * len(tags), 4.4),
                             squeeze=False)
    for ax, tag in zip(axes[0], tags):
        run = payload["runs"][tag]
        for scanner, color in (("brute", "#2a78d6"), ("bond", "#2e8b57")):
            arms = [a for a in run["arms"] if a.get("scanner", "bond") == scanner]
            xs = [a["recall_vs_exact_at_10_mean"] for a in arms]
            ys = [a["ms_per_query"] for a in arms]
            ax.plot(xs, ys, "o-", color=color,
                    label=f"partitioned fused {scanner}")
            for a in arms:
                ax.annotate(f"p{a['nprobe']}",
                            (a["recall_vs_exact_at_10_mean"], a["ms_per_query"]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6, color=color)
        ax.axhline(run["dense_interleaved_ms_per_query"], color="#5c4a9e",
                   linestyle="--", linewidth=1.2, label="dense fused (exact)")
        ax.set_xlabel("recall@10 vs exact MaxSim")
        ax.set_ylabel("ms / query (interleaved best-of-5)")
        ax.set_title("all cores" if tag == "mt" else "1 thread")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(color="#e9e8e2", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"e03 partitioned fused scan — {dataset} "
        f"(P={payload['partitioning']['n_partitions_nonempty']}, "
        f"labels = nprobe; rightmost point = full probe, exact)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e03_partitioned_fused_scan_{dataset}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {fig_path}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1"):
        print("usage: ... e03_partitioned_fused_scan <threads:0|1> [dataset ...]")
        sys.exit(1)
    nt = int(sys.argv[1])
    datasets = sys.argv[2:] if len(sys.argv) > 2 else DATASETS
    t0 = time.perf_counter()
    for ds in datasets:
        run_dataset(ds, nt)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
