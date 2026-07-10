"""Stage 5 e01: IR evaluation of the best Stage 3-4 methods (R9).

Single responsibility: take the best method per family from the R8 verdict and
evaluate them as RETRIEVAL SYSTEMS — standard qrels metrics (nDCG@10,
recall@100, MRR@10), CoRECT RC metrics, and recall-vs-exact — alongside
interleaved wall-clock latency, on all evaluable test queries.  This is the
paper's final section: the exact arms inherit exact MaxSim's IR quality by
construction; the approximate arms show what the qrels metrics hide vs
surface relative to recall_vs_exact.

Method arms (plan doc R9, one-stack interleaved controls as Stage 4 e01):
  dense_fused          : fused dense MaxSim over the full corpus — the exact
                         production baseline (RQ2 kernel).
  openblas             : the NumPy/scipy-openblas GEMM formulation of exact
                         MaxSim (exact_maxsim_topk, the oracle routine) as a
                         timed arm — the general-purpose BLAS reference the
                         fused kernel is measured against in E3, now placed
                         on the system-level quality-latency table.
  bond_exact_safe      : fused BOND, TIGHT bound, natural order, C={112},
                         self_bound tau (the free production policy;
                         recall 1.0 by construction — exactness costs
                         nothing in IR quality).
  partitioned@{16,32}  : partitioned fused scan (bond scanner) at the two
                         e03 frontier operating points that land at
                         recall_vs_exact@10 ~0.9 / ~0.95 on every dataset.
  faiss_ivf@B, plaid@B : tuned external references at matched budgets.

Protocol: retrieval depth k=100 for every arm (one run yields recall@100 and
the @10 metrics from the ranking prefix); quality over ALL evaluable test
queries; CoRECT wrapper smoke test re-run on the real dense_fused run before
any RC metric is reported (plan "CoRECT wrapper smoke test" gate); timing
interleaved round-robin after warmup with per-rep dispersion recorded (the
R12b lesson: dense is never timed standalone); index build cost and memory
reported separately from query latency.

Datasets  : scifact (300 test q), nfcorpus (323), arguana (200), scidocs (200)
Threads   : one setting per run (arg): 0 = all cores, 1 = 1T
k         : 100 retrieved; metrics at 10 and 100
Budgets   : {100, 1000, 5000} intersected with corpus size

Output (merged per dataset across thread tags):
  results/json/stage5_corect_e01_ir_evaluation_<dataset>.json
  results/figures/stage5_corect/e01_ir_evaluation_<dataset>.png

Usage:
    uv run python -m experiments.stage5_corect.e01_ir_evaluation \
        <threads:0|1> [dataset ...]
"""
from __future__ import annotations

import contextlib
import json
import platform
import sys
import time

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_ids, load_qrels_tsv, qrels_path
from bondmaxsim.data.loader import load_dataset, load_eval_queries, unpack_embeddings
from bondmaxsim.data.packing import build_qcum
from bondmaxsim.eval.corect import compute_rc_metrics, corect_smoke_test
from bondmaxsim.eval.qrels import compute_quality_metrics
from bondmaxsim.kernels.fused_panel import (
    load_fused_panel_kernel,
    run_fused_panel_bond,
    run_fused_panel_brute,
)
from bondmaxsim.oracle.agreement import exact_agreement, recall_at_k
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_topk
from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline
from bondmaxsim.baselines.plaid import PLAIDBaseline
from bondmaxsim.partitioned_scan import PartitionedFusedScan
from bondmaxsim.testbed.packing_cache import PackingCache

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
K_RETRIEVE = 100                 # retrieval depth (recall@100 needs it)
K_EVAL = 10                      # @10 metrics use the ranking prefix
CHECKPOINTS = (112,)             # R12a/R12c winner
ORDER = "natural"
BOUND = "tight"
PART_NPROBES = (16, 32)          # e03 frontier: recall@10 ~0.9 / ~0.95 (all ds)
BUDGETS = [100, 1000, 5000]
REP = 5
FAISS_NPROBE = 32                # candidate-generation arm (recall-oriented)
PLAID_N_IVF_PROBE = 8

RESULTS_JSON = REPO_ROOT / "results" / "json"
FIG_DIR = REPO_ROOT / "results" / "figures" / "stage5_corect"
FAISS_CACHE = REPO_ROOT / "data" / "faiss_indexes"


def _pin_threads(nt: int):
    """Best-effort 1T pinning across faiss / BLAS / torch for nt == 1."""
    import faiss

    faiss.omp_set_num_threads(nt if nt > 0 else faiss.omp_get_max_threads())
    try:
        import torch

        torch.set_num_threads(nt if nt > 0 else torch.get_num_threads())
    except ImportError:
        pass
    if nt > 0:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=nt, user_api="blas")
    return contextlib.nullcontext()


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _dir_bytes(path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _to_run(results, query_ids, corpus_ids) -> dict[str, dict[str, float]]:
    """Map per-query (doc_idx_array, score_array) to a ranx/CoRECT run dict.

    The score written per doc is its (desc) rank position, NOT its MaxSim score:
    ranx and pytrec_eval break score-ties differently, so we hand both the
    ranking the system actually emitted rather than the raw scores.  An earlier
    version separated ties with s_i - i*1e-7, but pytrec_eval stores scores as
    32-bit floats and one float32 ULP at nfcorpus MaxSim magnitudes (~4.5) is
    ~5.4e-7 > 1e-7, so the eps was truncated away: pytrec re-tied near-equal
    docs and broke them by doc-id, disagreeing with ranx by up to ~5e-4 nDCG@10
    (nfcorpus PLAIN-934/PLAIN-2660).  Integer ranks are exact in float32, so
    both evaluators now see identical rankings (smoke-test gap -> 0).
    """
    run: dict[str, dict[str, float]] = {}
    for qid, (ids, scores) in zip(query_ids, results):
        n = len(ids)
        run[qid] = {corpus_ids[int(d)]: float(n - i)
                    for i, d in enumerate(ids)}
    return run


def run_dataset(dataset: str, nt: int) -> None:
    flat, starts, _ = load_dataset(dataset)
    queries, query_ids = load_eval_queries(dataset)
    qrels = load_qrels_tsv(qrels_path(dataset))
    corpus_ids = load_ids(dataset)["corpus_ids"]
    n_docs = len(starts)
    budgets = sorted({min(b, n_docs) for b in BUDGETS})
    tag = "mt" if nt == 0 else "1t"

    evaluable = sum(1 for q in query_ids if q in qrels)
    print(f"\n=== e01 {dataset}: {n_docs} docs, {len(queries)} queries "
          f"({evaluable} with test qrels), k={K_RETRIEVE}, budgets={budgets}, "
          f"threads={'all' if nt == 0 else nt} ===")
    if evaluable == 0:
        raise RuntimeError(f"{dataset}: no evaluable queries — regenerate "
                           "sidecars/test queries (see data/embeddings/README.md)")

    lib = load_fused_panel_kernel()
    pk = PackingCache(flat, starts)

    # ---- exact oracle reference (NumPy, independent of the kernels)
    t0 = time.perf_counter()
    exact = [exact_maxsim_topk(q, flat, starts, k=K_RETRIEVE) for q in queries]
    print(f"  exact oracle (k={K_RETRIEVE}) in {time.perf_counter() - t0:.1f}s")

    # ---- indexes (built/cached before any timing; cost reported separately)
    index_stats: dict[str, dict] = {}

    ivf = FaissIVFBaseline(flat, starts)
    t0 = time.perf_counter()
    cache = FAISS_CACHE / f"{dataset}.faiss"
    ivf.build(cache_path=str(cache))
    ivf.index.nprobe = FAISS_NPROBE
    index_stats["faiss_ivf"] = {
        "build_or_load_s": time.perf_counter() - t0, "n_lists": ivf.n_lists,
        "disk_bytes": cache.stat().st_size if cache.exists() else None}
    print(f"  faiss ready in {index_stats['faiss_ivf']['build_or_load_s']:.1f}s")

    plaid = PLAIDBaseline(dataset, n_ivf_probe=PLAID_N_IVF_PROBE,
                          num_threads=(nt if nt > 0 else None))
    t0 = time.perf_counter()
    if plaid.exists():
        plaid.load()
    else:
        plaid.build(unpack_embeddings(flat, starts))
    plaid_dir = plaid.index_root / dataset
    index_stats["plaid"] = {
        "build_or_load_s": time.perf_counter() - t0,
        "disk_bytes": _dir_bytes(plaid_dir) if plaid_dir.exists() else None}
    print(f"  plaid ready in {index_stats['plaid']['build_or_load_s']:.1f}s")

    t0 = time.perf_counter()
    part_index = PartitionedFusedScan(flat, starts)
    part_stats = part_index.build()
    index_stats["partitioned"] = {
        "build_or_load_s": time.perf_counter() - t0, **part_stats}
    print(f"  partitioned index ready in "
          f"{index_stats['partitioned']['build_or_load_s']:.1f}s "
          f"(P={part_stats['n_partitions_nonempty']})")

    # ---- kernel arm preparation
    Qs = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]
    pdat, goff, doff, gds, _ = pk._get_panel_packing()
    index_stats["fused_panel"] = {"packing_bytes": int(pdat.nbytes)}

    def run_dense(collect=None):
        for Q in Qs:
            ids, scores = run_fused_panel_brute(lib, pdat, goff, doff, gds, Q,
                                                K_RETRIEVE, n_threads=nt)
            if collect is not None:
                collect.append((ids, scores))

    def run_blas(collect=None):
        # The oracle routine itself, timed: one GEMM (query @ flat.T) plus a
        # reduceat max-sum, threaded by the BLAS according to the run's pin.
        for Q in Qs:
            ids, scores = exact_maxsim_topk(Q, flat, starts, k=K_RETRIEVE)
            if collect is not None:
                collect.append((ids, scores))

    prepared = []
    for q in queries:
        pd, go, do, gd, Qe, order = pk.dispatch_order_panel(q, ORDER)
        prepared.append((pd, go, do, gd, Qe, order, build_qcum(Qe, order)))
    arr = np.asarray(CHECKPOINTS, dtype=np.uint32)

    def run_bond(collect=None):
        for pd, go, do, gd, Qe, order, Qc in prepared:
            ids, scores, stats = run_fused_panel_bond(
                lib, pd, go, do, gd, Qe, order, Qc,
                shrink=1.0, tau_seed=-np.inf, K=K_RETRIEVE, n_threads=nt,
                level="doc", checkpoints=arr, bound=BOUND)
            if collect is not None:
                collect.append((ids, scores, stats))

    def make_partitioned(nprobe):
        def _run(collect=None):
            for q in queries:
                ids, scores, stats = part_index.search(
                    lib, q, k=K_RETRIEVE, nprobe=nprobe,
                    checkpoints=CHECKPOINTS, bound=BOUND,
                    n_threads=nt, scanner="bond")
                if collect is not None:
                    collect.append((ids, scores, stats))
        return _run

    part_runs = {p: make_partitioned(p) for p in PART_NPROBES}

    def make_faiss(budget):
        def _run(collect=None):
            for q in queries:
                ids, scores, nc = ivf.topk(q, k=K_RETRIEVE,
                                           candidate_budget=budget)
                if collect is not None:
                    collect.append((ids, scores, nc))
        return _run

    faiss_runs = {b: make_faiss(b) for b in budgets}

    def make_plaid(budget):
        def _run(collect=None):
            plaid.set_search_params(n_full_scores=budget)
            for ids, scores in plaid.search(queries, k=K_RETRIEVE):
                if collect is not None:
                    collect.append((ids, scores))
        return _run

    plaid_runs = {b: make_plaid(b) for b in budgets}

    # ---- quality (untimed single pass per arm; run dicts + all metrics)
    exact10 = [(e_ids[:K_EVAL], e_sc[:K_EVAL]) for e_ids, e_sc in exact]

    def quality_row(results, label=None):
        """results: per-query (ids desc, scores desc). Returns metric dict."""
        if label is not None:
            print(f"    quality: {label}")
        run = _to_run(results, query_ids, corpus_ids)
        row = compute_quality_metrics(run, qrels)
        row["CoRECT_RC_metrics"] = compute_rc_metrics(run, qrels)
        row["recall_vs_exact_at_10"] = float(np.mean(
            [recall_at_k(ids[:K_EVAL], e_ids) for (ids, _), (e_ids, _)
             in zip(results, exact10)]))
        row["recall_vs_exact_at_100"] = float(np.mean(
            [recall_at_k(ids, e_ids) for (ids, _), (e_ids, _)
             in zip(results, exact)]))
        return row, run

    quality: dict[str, dict] = {}

    got: list = []
    run_dense(collect=got)
    dense_results = got
    dense_run = _to_run(dense_results, query_ids, corpus_ids)
    # CoRECT wrapper smoke test on the REAL dense run before any RC metric
    # is reported (plan gate: verify before scaling).
    corect_smoke_test(dense_run, qrels)
    print("  CoRECT smoke test on dense_fused run: OK (agrees with ranx)")
    print("  scoring quality rows (qrels + CoRECT RC metrics per arm)...")
    quality["dense_fused"], _ = quality_row(dense_results, "dense_fused")
    agree = min(exact_agreement(ids[:K_EVAL], e_ids, e_sc)
                for (ids, _), (e_ids, e_sc) in zip(dense_results, exact10))
    if agree < 1.0:
        raise RuntimeError(f"dense_fused exact-agreement@10 failed: {agree}")

    # The openblas arm IS the oracle routine, so its results are the already
    # computed `exact` list; only its wall-clock needs a timed pass.
    quality["openblas"], _ = quality_row(exact, "openblas")

    got = []
    run_bond(collect=got)
    quality["bond_exact_safe"], _ = quality_row(
        [(i, s) for i, s, _ in got], "bond_exact_safe")
    agree = min(exact_agreement(ids[:K_EVAL], e_ids, e_sc)
                for (ids, _, _), (e_ids, e_sc) in zip(got, exact10))
    if agree < 1.0:
        raise RuntimeError(f"bond exact-agreement@10 failed: {agree}")
    quality["bond_exact_safe"]["pruned_docs_pct"] = float(
        np.mean([s[1] for _, _, s in got])) / n_docs * 100.0

    for p in PART_NPROBES:
        got = []
        part_runs[p](collect=got)
        quality[f"partitioned@{p}"], _ = quality_row(
            [(i, s) for i, s, _ in got], f"partitioned@{p}")
        quality[f"partitioned@{p}"]["docs_probed_pct_mean"] = float(
            np.mean([s["docs_probed_pct"] for _, _, s in got]))

    for b in budgets:
        got = []
        faiss_runs[b](collect=got)
        quality[f"faiss@{b}"], _ = quality_row(
            [(i, s) for i, s, _ in got], f"faiss@{b}")
        quality[f"faiss@{b}"]["n_candidates_mean"] = float(
            np.mean([nc for _, _, nc in got]))
        got = []
        plaid_runs[b](collect=got)
        quality[f"plaid@{b}"], _ = quality_row(got, f"plaid@{b}")

    # ---- interleaved timing (round-robin, warmup + REP reps, dispersion kept)
    def ms(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) / len(queries) * 1e3

    arms_fns = [("dense_fused", run_dense), ("openblas", run_blas),
                ("bond_exact_safe", run_bond)]
    arms_fns += [(f"partitioned@{p}", part_runs[p]) for p in PART_NPROBES]
    arms_fns += [(f"faiss@{b}", faiss_runs[b]) for b in budgets]
    arms_fns += [(f"plaid@{b}", plaid_runs[b]) for b in budgets]

    print(f"  interleaved timing: {len(arms_fns)} arms x {REP} reps "
          "(+warmup)")
    with _pin_threads(nt):
        for _, fn in arms_fns:      # warmup
            fn()
        print("    warmup done, timing reps...")
        reps = {name: [] for name, _ in arms_fns}
        for r in range(REP):
            for name, fn in arms_fns:
                reps[name].append(ms(fn))
            print(f"    rep {r + 1}/{REP} done")

    best = {name: min(v) for name, v in reps.items()}
    dense = best["dense_fused"]

    # ---- emit rows (ResultRecord-aligned field names + arm extras)
    arms_out = []

    def emit(name, method, budget, threshold_policy, shrink, extra=None):
        q = quality[name]
        kern = best[name]
        row = {
            "arm": name,
            "method": method,
            "candidate_budget": budget,
            "dimension_order": ORDER,
            "threshold_policy": threshold_policy,
            "shrink": shrink,
            "recall_vs_exact_at_10": q["recall_vs_exact_at_10"],
            "recall_vs_exact_at_100": q["recall_vs_exact_at_100"],
            "nDCG_at_10": q["nDCG_at_10"],
            "recall_at_100": q["recall_at_100"],
            "MRR_at_10": q["MRR_at_10"],
            "CoRECT_RC_metrics": q["CoRECT_RC_metrics"],
            "ms_per_query": kern,
            "ms_per_query_reps": reps[name],
            "ms_per_query_mean": float(np.mean(reps[name])),
            "ms_per_query_std": float(np.std(reps[name])),
            "qps": 1000.0 / kern,
            "margin_vs_dense_pct": (dense - kern) / dense * 100.0,
        }
        for key in ("pruned_docs_pct", "docs_probed_pct_mean",
                    "n_candidates_mean"):
            if key in q:
                row[key] = q[key]
        if extra:
            row.update(extra)
        arms_out.append(row)
        print(f"  {name:<16} nDCG@10={row['nDCG_at_10']:.4f} "
              f"R@100={row['recall_at_100']:.4f} MRR@10={row['MRR_at_10']:.4f} "
              f"vs_exact@10={row['recall_vs_exact_at_10']:.3f} "
              f"ms/q={kern:8.3f}±{row['ms_per_query_std']:.3f}")

    emit("dense_fused", "fused_panel_maxsim_brute", None,
         "exact_full_scan", 1.0)
    emit("openblas", "numpy_openblas_maxsim", None,
         "exact_full_scan", 1.0)
    emit("bond_exact_safe", "fused_panel_maxsim_bond", None,
         "self_bound", 1.0,
         extra={"checkpoints": list(CHECKPOINTS), "bound": BOUND})
    for p in PART_NPROBES:
        emit(f"partitioned@{p}", "partitioned_fused_bond", None,
             "self_bound", 1.0,
             extra={"nprobe": p, "checkpoints": list(CHECKPOINTS),
                    "bound": BOUND,
                    "notes": "approximate via partition probing (shrink=1 "
                             "inside probed partitions)"})
    for b in budgets:
        emit(f"faiss@{b}", "faiss_ivf_rerank", b, "candidate_topk", None,
             extra={"nprobe": FAISS_NPROBE})
    for b in budgets:
        emit(f"plaid@{b}", "plaid_fastplaid", b, "candidate_topk", None,
             extra={"n_ivf_probe": PLAID_N_IVF_PROBE, "n_full_scores": b})

    run_payload = {
        "threads": "all_cores" if nt == 0 else "single",
        "thread_count": nt,
        "dense_interleaved_ms_per_query": dense,
        "arms": arms_out,
    }

    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    out = RESULTS_JSON / f"stage5_corect_e01_ir_evaluation_{dataset}.json"
    payload = {
        "experiment": "e01_ir_evaluation",
        "purpose": "R9/Stage 5: best Stage 3-4 methods evaluated as retrieval "
                   "systems (qrels + CoRECT RC metrics, interleaved latency, "
                   "build cost separate) — the paper's final section",
        "dataset": dataset, "n_docs": n_docs,
        "n_queries": len(queries), "n_evaluable_queries": evaluable,
        "query_source": ("test_queries_npz"
                         if (REPO_ROOT / "data" / "embeddings" /
                             f"{dataset}_test_queries.npz").exists()
                         else "main_npz"),
        "k_retrieve": K_RETRIEVE, "k_eval": K_EVAL, "budgets": budgets,
        "kernel": {"bound": BOUND, "order": ORDER,
                   "checkpoints": list(CHECKPOINTS), "shrink": 1.0,
                   "threshold_policy": "self_bound"},
        "faiss": {"n_lists": ivf.n_lists, "nprobe": FAISS_NPROBE,
                  "metric": "inner_product"},
        "plaid": {"backend": "fast_plaid", "nbits": plaid.nbits,
                  "kmeans_niters": plaid.kmeans_niters,
                  "n_ivf_probe": PLAID_N_IVF_PROBE},
        "partitioned": {**part_stats, "scanner": "bond",
                        "nprobes": list(PART_NPROBES)},
        "indexes": index_stats,
        "embedding_note": "docs encoded text-only (no title concat) by the "
                          "archive cache; consistent across arms but absolute "
                          "nDCG trails published BEIR numbers",
        "rep_best_of": REP,
        "machine": {"host": platform.node(), "cpu": _cpu_model(),
                    "os": platform.system().lower()},
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
    """Quality-latency frontier: nDCG@10 (and recall@100) vs interleaved ms/q,
    one panel per thread tag; exact arms sit at recall 1.0 by construction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in ("mt", "1t") if t in payload["runs"]]
    metrics = [("nDCG_at_10", "nDCG@10"), ("recall_at_100", "recall@100")]
    fig, axes = plt.subplots(len(metrics), len(tags),
                             figsize=(6.0 * len(tags), 4.2 * len(metrics)),
                             squeeze=False)
    styles = {
        "dense_fused": ("#5c4a9e", "*", 130),
        "openblas": ("#8a8a8a", "v", 55),
        "bond_exact_safe": ("#2e8b57", "P", 70),
        "partitioned": ("#2e8b57", "o", 40),
        "faiss": ("#2a78d6", "s", 40),
        "plaid": ("#eb6834", "D", 40),
    }
    for col, tag in enumerate(tags):
        arms = payload["runs"][tag]["arms"]
        for r, (field, label) in enumerate(metrics):
            ax = axes[r][col]
            for a in arms:
                fam = a["arm"].split("@")[0]
                color, marker, size = styles.get(fam, ("#888888", "x", 40))
                ax.scatter(a["ms_per_query"], a[field], c=color, marker=marker,
                           s=size, zorder=3)
                ax.annotate(a["arm"], (a["ms_per_query"], a[field]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6, color=color)
            ax.set_xscale("log")
            ax.set_xlabel("ms / query (interleaved best-of-%d)"
                          % payload["rep_best_of"])
            ax.set_ylabel(label)
            ax.set_title(f"{dataset} — "
                         f"{'all cores' if tag == 'mt' else '1 thread'}")
            ax.grid(color="#e9e8e2", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("e01 IR evaluation — quality-latency frontier "
                 "(qrels metrics; exact arms are leftmost-at-ceiling)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"e01_ir_evaluation_{dataset}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {fig_path}")


def main():
    # Line-buffer stdout so progress is visible live when piped/redirected
    # (block buffering otherwise hides everything until exit, and loses it
    # entirely if the run is killed, e.g. by a `timeout` wrapper).
    sys.stdout.reconfigure(line_buffering=True)
    if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1"):
        print("usage: ... e01_ir_evaluation <threads:0|1> [dataset ...]")
        sys.exit(1)
    nt = int(sys.argv[1])
    datasets = sys.argv[2:] if len(sys.argv) > 2 else DATASETS
    t0 = time.perf_counter()
    for ds in datasets:
        run_dataset(ds, nt)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
