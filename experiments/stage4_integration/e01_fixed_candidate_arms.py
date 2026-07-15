"""Stage 4 e01: method arms at FIXED candidate-set sizes (R8).

Single responsibility: compare, on the same machine / queries / thread
setting, the production-path exact methods against the candidate-generation
baselines, holding the candidate budget FIXED per comparison point so an
IVF/PLAID speedup can never be mistaken for a scoring-kernel speedup
(Stage 1 §5.4 / Convention 7; plan doc Stage 4 "Important control").

Method arms:
  dense_fused           : fused dense MaxSim over the full corpus — the exact
                          production baseline (RQ2 kernel).
  bond_exact_safe_seeded: fused BOND (tight bound, natural order, C={112},
                          shrink=1) with the REALISTIC IVF-seeded tau of
                          e02's ivf_seed_cheap arm; kernel and seed costs
                          reported separately and summed.  Exhaustive and
                          exact — candidate seeding only feeds tau_seed.
  faiss_ivf_rerank@B    : FAISS-IVF token retrieval + doc aggregation to
                          exactly B candidates + exact MaxSim rerank
                          (recall < 1 possible: the true top-10 may miss the
                          candidate set).
  plaid@B               : PyLate FastPlaid with n_full_scores = B (its
                          candidate-budget analog), n_ivf_probe = 8.

Note (plan Stage 4 method list): "BOND kernel as reranker over a fixed
candidate set" is superseded by the seeded-tau full-corpus arm — at C={112}
the kernel already prunes 88–98% of docs itself and the panel packing is
corpus-level; a per-query candidate repack would time the packer, not the
mechanism.  pdx_ivf is deferred with R7 (PDX-sigmod IVF needs its own build
at a corpus scale this machine cannot hold; the FAISS arm covers the
candidate-generation control at this scale).

Timing: all arms interleaved round-robin, best-of-5, per the R12b/R12c
methodology (dense never timed standalone).

Datasets  : scifact, nfcorpus, arguana, scidocs
Threads   : one setting per run (arg): 0 = all cores (decision baseline), 1 = 1T
Queries   : 50 (seed 42, == e08/r12c)
k         : 10
Budgets   : {100, 500, 1000, 5000} intersected with corpus size

Output (merged per dataset across thread tags):
  results/json/stage4_integration_e01_fixed_candidate_arms_<dataset>.json

Usage:
    uv run python -m experiments.stage4_integration.e01_fixed_candidate_arms \
        <threads:0|1> [dataset ...]
"""
from __future__ import annotations

import contextlib
import json
import sys
import time

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset, unpack_embeddings
from bondmaxsim.data.packing import build_qcum
from bondmaxsim.kernels.fused_panel import run_fused_panel_bond, run_fused_panel_brute
from bondmaxsim.oracle.agreement import recall_at_k, validate_boundary_tie_equivalence
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores
from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline
from bondmaxsim.baselines.pdx_ivf import PDXIVFBaseline
from bondmaxsim.baselines.plaid import PLAIDBaseline
from bondmaxsim.partitioned_scan import PartitionedFusedScan
from bondmaxsim.testbed.runner import Runner
from bondmaxsim.threshold.policies import candidate_seed_threshold

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
BUDGETS = [100, 500, 1000, 5000]
CHECKPOINTS = (112,)             # R12a/R12c winner
ORDER = "natural"
BOUND = "tight"
N_QUERIES = 50
QUERY_SEED = 42
K_TOP = 10
REP = 5
TAU_EPS = 1e-3                   # == testbed.thresholds._TAU_SEED_EPS rationale
SEED_IVF = dict(nprobe=4, k_token=32, s=10)   # e02 ivf_seed_cheap
FAISS_NPROBE = 32                # candidate-generation arm (recall-oriented)
PLAID_N_IVF_PROBE = 8

RESULTS_JSON = REPO_ROOT / "results" / "json"
FAISS_CACHE = REPO_ROOT / "data" / "faiss_indexes"


def _subsample_queries(queries, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


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


def run_dataset(dataset, nt):
    flat, starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    n_docs = len(starts)
    budgets = sorted({min(b, n_docs) for b in BUDGETS})
    runner = Runner(flat, starts, queries)
    pk = runner._packing
    lib = runner._get_fused_lib()
    tag = "mt" if nt == 0 else "1t"
    print(f"\n=== e01 {dataset}: {n_docs} docs, {len(queries)} queries, "
          f"budgets={budgets}, threads={'all' if nt == 0 else nt} ===")

    exact_full = [exact_maxsim_scores(q, flat, starts) for q in queries]
    exact = [topk_from_scores(scores, K_TOP) for scores in exact_full]

    # ---- indexes (built/cached before any timing)
    ivf = FaissIVFBaseline(flat, starts)
    t0 = time.perf_counter()
    ivf.build(cache_path=str(FAISS_CACHE / f"{dataset}.faiss"))
    print(f"  faiss ready in {time.perf_counter() - t0:.1f}s (n_lists={ivf.n_lists})")

    plaid = PLAIDBaseline(dataset, n_ivf_probe=PLAID_N_IVF_PROBE,
                          num_threads=(nt if nt > 0 else None))
    t0 = time.perf_counter()
    if plaid.exists():
        plaid.load()
    else:
        plaid.build(unpack_embeddings(flat, starts))
    print(f"  plaid ready in {time.perf_counter() - t0:.1f}s")

    # Mikel-branch flat PDX-IVF (pdxearch build required; skipped gracefully
    # if the in-memory index cannot be built, e.g. memory pressure on scidocs
    # — the materialization duplicates the corpus).
    pdx = None
    try:
        t0 = time.perf_counter()
        pdx = PDXIVFBaseline(flat, starts, nprobe=FAISS_NPROBE)
        pdx.build()
        print(f"  pdx_ivf ready in {time.perf_counter() - t0:.1f}s "
              f"(n_buckets={pdx.n_buckets})")
    except Exception as exc:                      # noqa: BLE001
        pdx = None
        print(f"  pdx_ivf SKIPPED: {type(exc).__name__}: {exc}")

    # ---- kernel arm preparation
    Qs = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]
    pdat, goff, doff, gds, _ = pk._get_panel_packing()

    def run_dense(collect=None):
        for Q in Qs:
            ids, _ = run_fused_panel_brute(lib, pdat, goff, doff, gds, Q, K_TOP,
                                           n_threads=nt)
            if collect is not None:
                collect.append(ids)

    # Realistic seeded tau (e02 ivf_seed_cheap) + its measured cost.  The seed
    # pipeline is pinned to ONE thread regardless of the arm thread setting:
    # its work items are too small for OMP/BLAS fan-out to pay (see e02
    # driver docstring); recorded as seed_threads=1 in the JSON.
    import faiss as _faiss

    _faiss.omp_set_num_threads(1)
    ivf.index.nprobe = SEED_IVF["nprobe"]

    def _seed_pass():
        taus = []
        for q in queries:
            _, scores, _ = ivf.topk(q, k=K_TOP, candidate_budget=SEED_IVF["s"],
                                    k_token=SEED_IVF["k_token"])
            taus.append(candidate_seed_threshold(scores, K_TOP) - TAU_EPS)
        return taus

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        taus = _seed_pass()
        seed_cost = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            _seed_pass()
            seed_cost = min(seed_cost, (time.perf_counter() - t0) / len(queries) * 1e3)
    _faiss.omp_set_num_threads(nt if nt > 0 else _faiss.omp_get_max_threads())

    prepared = []
    for q in queries:
        pd, go, do, gd, Qe, order = pk.dispatch_order_panel(q, ORDER)
        prepared.append((pd, go, do, gd, Qe, order, build_qcum(Qe, order)))
    arr = np.asarray(CHECKPOINTS, dtype=np.uint32)

    def run_bond(collect=None):
        for (pd, go, do, gd, Qe, order, Qc), tau in zip(prepared, taus):
            ids, _, stats = run_fused_panel_bond(
                lib, pd, go, do, gd, Qe, order, Qc,
                shrink=1.0, tau_seed=tau, K=K_TOP, n_threads=nt,
                level="doc", checkpoints=arr, bound=BOUND)
            if collect is not None:
                collect.append((ids, stats))

    def make_faiss(budget):
        def _run(collect=None):
            for q in queries:
                ids, _, nc = ivf.topk(q, k=K_TOP, candidate_budget=budget)
                if collect is not None:
                    collect.append((ids, nc))
        return _run

    faiss_runs = {b: make_faiss(b) for b in budgets}

    def make_pdx(budget):
        def _run(collect=None):
            for q in queries:
                ids, _, nc = pdx.topk(q, k=K_TOP, candidate_budget=budget)
                if collect is not None:
                    collect.append((ids, nc))
        return _run

    pdx_runs = {b: make_pdx(b) for b in budgets} if pdx is not None else {}

    def make_plaid(budget):
        def _run(collect=None):
            plaid.set_search_params(n_full_scores=budget)
            for ids, _ in plaid.search(queries, k=K_TOP):
                if collect is not None:
                    collect.append(ids)
        return _run

    plaid_runs = {b: make_plaid(b) for b in budgets}

    # Partitioned fused scan (our kernel + IVF partition skipping, e03 arm)
    # at nprobe matched to each budget: probed docs ~= B, so it competes in
    # the same fixed-candidate row as the token-level pipelines.
    t0 = time.perf_counter()
    part_index = PartitionedFusedScan(flat, starts)
    part_stats = part_index.build()
    P = part_stats["n_partitions_nonempty"]
    docs_per_part = part_stats["docs_per_partition_mean"]
    print(f"  partitioned index ready in {time.perf_counter() - t0:.1f}s "
          f"(P={P}, ~{docs_per_part:.0f} docs/partition)")

    def _nprobe_for(budget):
        return int(np.clip(round(budget / docs_per_part), 1, P))

    def make_partitioned(budget):
        nprobe = _nprobe_for(budget)

        def _run(collect=None):
            for q in queries:
                out = part_index.search(lib, q, k=K_TOP, nprobe=nprobe,
                                        checkpoints=CHECKPOINTS, bound=BOUND,
                                        n_threads=nt, scanner="bond")
                if collect is not None:
                    collect.append(out)
        return _run

    part_runs = {b: make_partitioned(b) for b in budgets}

    # ---- quality (untimed single pass per arm)
    # The seed pipeline above intentionally used the cheap SEED_IVF nprobe;
    # the candidate-generation arms are measured at FAISS_NPROBE throughout.
    ivf.index.nprobe = FAISS_NPROBE
    quality = {}
    got: list = []
    run_dense(collect=got)
    dense_agreements = [
        validate_boundary_tie_equivalence(
            ids, e_ids, e_sc, k=K_TOP, num_documents=n_docs,
            exact_scores_by_id=full_scores,
        )
        for ids, (e_ids, e_sc), full_scores in zip(got, exact, exact_full)
    ]
    quality["dense_fused"] = dict(
        recall=min(r.recall_vs_oracle_set for r in dense_agreements),
        strict_top_k_set_equal=all(r.strict_top_k_set_equal for r in dense_agreements),
        boundary_tie_equivalent=all(r.exact_gate_passed for r in dense_agreements),
        agreement_failure_codes=sorted({c for r in dense_agreements for c in r.failure_codes}),
    )

    got = []
    run_bond(collect=got)
    bond_agreements = [
        validate_boundary_tie_equivalence(
            ids, e_ids, e_sc, k=K_TOP, num_documents=n_docs,
            exact_scores_by_id=full_scores,
        )
        for (ids, _), (e_ids, e_sc), full_scores in zip(got, exact, exact_full)
    ]
    rec = min(r.recall_vs_oracle_set for r in bond_agreements)
    if not all(r.exact_gate_passed for r in bond_agreements):
        failures = sorted({c for r in bond_agreements for c in r.failure_codes})
        raise RuntimeError(f"verified exact gate failed on seeded BOND arm: {failures}")
    quality["bond"] = dict(
        recall=rec,
        strict_top_k_set_equal=all(r.strict_top_k_set_equal for r in bond_agreements),
        boundary_tie_equivalent=all(r.exact_gate_passed for r in bond_agreements),
        agreement_failure_codes=[],
        pruned_docs_pct=float(np.mean([s[1] for _, s in got])) / n_docs * 100.0)

    for b in budgets:
        got = []
        faiss_runs[b](collect=got)
        quality[f"faiss@{b}"] = dict(
            recall=float(np.mean([recall_at_k(ids, e_ids) for (ids, _), (e_ids, _)
                                  in zip(got, exact)])),
            n_candidates_mean=float(np.mean([nc for _, nc in got])))
        got = []
        plaid_runs[b](collect=got)
        quality[f"plaid@{b}"] = dict(
            recall=float(np.mean([recall_at_k(ids, e_ids) for ids, (e_ids, _)
                                  in zip(got, exact)])))
        if pdx is not None:
            got = []
            pdx_runs[b](collect=got)
            quality[f"pdx@{b}"] = dict(
                recall=float(np.mean([recall_at_k(ids, e_ids) for (ids, _), (e_ids, _)
                                      in zip(got, exact)])),
                n_candidates_mean=float(np.mean([nc for _, nc in got])))
        got = []
        part_runs[b](collect=got)
        quality[f"partitioned@{b}"] = dict(
            recall=float(np.mean([recall_at_k(ids, e_ids)
                                  for (ids, _, _), (e_ids, _)
                                  in zip(got, exact)])),
            docs_probed_pct_mean=float(np.mean(
                [s["docs_probed_pct"] for _, _, s in got])))

    # ---- interleaved timing
    def ms(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) / len(queries) * 1e3

    with _pin_threads(nt):
        arms_fns = [("dense_fused", run_dense), ("bond", run_bond)]
        arms_fns += [(f"faiss@{b}", faiss_runs[b]) for b in budgets]
        arms_fns += [(f"plaid@{b}", plaid_runs[b]) for b in budgets]
        arms_fns += [(f"pdx@{b}", pdx_runs[b]) for b in budgets if pdx is not None]
        arms_fns += [(f"partitioned@{b}", part_runs[b]) for b in budgets]
        for _, fn in arms_fns:      # warmup
            fn()
        best = {name: float("inf") for name, _ in arms_fns}
        for _ in range(REP):
            for name, fn in arms_fns:
                best[name] = min(best[name], ms(fn))

    dense = best["dense_fused"]
    arms_out = []

    def emit(name, method, budget, extra=None, seed_ms=None):
        kern = best[name]
        total = kern + seed_ms if seed_ms is not None else kern
        row = {
            "arm": name, "method": method, "candidate_budget": budget,
            "recall_vs_exact_at_10": quality[name]["recall"] if name in quality else None,
            "ms_per_query": kern,
            "seed_cost_ms_per_query": seed_ms,
            "ms_per_query_total": total,
            "margin_vs_dense_pct": (dense - total) / dense * 100.0,
        }
        if extra:
            row.update(extra)
        arms_out.append(row)
        rec = row["recall_vs_exact_at_10"]
        print(f"  {name:<14} recall={rec:5.3f}  ms/q={kern:8.3f}"
              f"{'' if seed_ms is None else f' (+{seed_ms:.2f} seed)'}"
              f"  margin={row['margin_vs_dense_pct']:+6.1f}%")

    emit("dense_fused", "fused_panel_maxsim_brute", None)
    emit("bond", "fused_panel_maxsim_bond", None, seed_ms=seed_cost, extra={
        "threshold_policy": "ivf_seed_cheap", "ivf_seed_params": SEED_IVF,
        "pruned_docs_pct": quality["bond"]["pruned_docs_pct"],
        "checkpoints": list(CHECKPOINTS), "bound": BOUND, "order": ORDER})
    for b in budgets:
        emit(f"faiss@{b}", "faiss_ivf_rerank", b, extra={
            "nprobe": FAISS_NPROBE,
            "n_candidates_mean": quality[f"faiss@{b}"]["n_candidates_mean"]})
    for b in budgets:
        emit(f"plaid@{b}", "plaid_fastplaid", b, extra={
            "n_ivf_probe": PLAID_N_IVF_PROBE, "n_full_scores": b})
    if pdx is not None:
        for b in budgets:
            emit(f"pdx@{b}", "pdx_ivf_rerank", b, extra={
                "nprobe": pdx.nprobe, "n_buckets": pdx.n_buckets,
                "n_candidates_mean": quality[f"pdx@{b}"]["n_candidates_mean"]})
    for b in budgets:
        emit(f"partitioned@{b}", "partitioned_fused_bond", b, extra={
            "nprobe": _nprobe_for(b),
            "docs_probed_pct_mean": quality[f"partitioned@{b}"]["docs_probed_pct_mean"],
            "checkpoints": list(CHECKPOINTS), "bound": BOUND})

    run_payload = {
        "threads": "all_cores" if nt == 0 else "single",
        "dense_interleaved_ms_per_query": dense,
        "arms": arms_out,
    }

    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    out = RESULTS_JSON / f"stage4_integration_e01_fixed_candidate_arms_{dataset}.json"
    payload = {
        "experiment": "e01_fixed_candidate_arms",
        "purpose": "R8: method separation at fixed candidate budgets "
                   "(IVF/PLAID wins must not be read as BOND wins and vice versa)",
        "dataset": dataset, "n_docs": n_docs, "n_queries": len(queries),
        "query_seed": QUERY_SEED, "k": K_TOP, "budgets": budgets,
        "kernel": {"bound": BOUND, "order": ORDER, "checkpoints": list(CHECKPOINTS),
                   "shrink": 1.0},
        "faiss": {"n_lists": ivf.n_lists, "nprobe": FAISS_NPROBE,
                  "metric": "inner_product"},
        "plaid": {"backend": "fast_plaid", "nbits": plaid.nbits,
                  "kmeans_niters": plaid.kmeans_niters,
                  "n_ivf_probe": PLAID_N_IVF_PROBE},
        "pdx_ivf": ({"n_buckets": pdx.n_buckets, "nprobe": pdx.nprobe,
                     "metric": "l2sq_unit_norm"} if pdx is not None else None),
        "partitioned": {**part_stats, "scanner": "bond",
                        "nprobe_per_budget": {b: _nprobe_for(b) for b in budgets}},
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
    """Separation plot: ms/q vs candidate budget per method, exhaustive arms as
    horizontal lines (the earlier-draft e03 'mechanism vs candidate generation'
    figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in ("mt", "1t") if t in payload["runs"]]
    fig, axes = plt.subplots(1, len(tags), figsize=(6.2 * len(tags), 4.4),
                             squeeze=False)
    for ax, tag in zip(axes[0], tags):
        run = payload["runs"][tag]
        arms = {a["arm"]: a for a in run["arms"]}
        budgets = payload["budgets"]
        for name, color, label in (
                ("faiss", "#2a78d6", "faiss-IVF + exact rerank"),
                ("plaid", "#eb6834", "PLAID (fast_plaid)"),
                ("pdx", "#b4468e", "PDX-IVF + exact rerank (Mikel)"),
                ("partitioned", "#2e8b57", "partitioned fused BOND (ours)")):
            xs = [b for b in budgets if f"{name}@{b}" in arms]
            ys = [arms[f"{name}@{b}"]["ms_per_query"] for b in xs]
            ax.plot(xs, ys, "o-", color=color, label=label)
            for x, b in zip(xs, xs):
                ax.annotate(f"{arms[f'{name}@{b}']['recall_vs_exact_at_10']:.2f}",
                            (x, arms[f"{name}@{b}"]["ms_per_query"]),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=6, color=color, ha="center")
        ax.axhline(arms["dense_fused"]["ms_per_query"], color="#5c4a9e",
                   linestyle="--", linewidth=1.2, label="dense fused (exact)")
        ax.axhline(arms["bond"]["ms_per_query_total"], color="#2e8b57",
                   linestyle="-.", linewidth=1.2,
                   label="BOND exact-safe + seed (exact)")
        ax.set_xscale("log")
        ax.set_xlabel("candidate budget (docs)")
        ax.set_ylabel("ms / query (interleaved best-of-5)")
        ax.set_title(f"{'all cores' if tag == 'mt' else '1 thread'}")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", color="#e9e8e2", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"e01 fixed-candidate arms — {dataset} (numbers above points = "
        f"recall@10 vs exact; exact arms are budget-free lines)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig_dir = REPO_ROOT / "results" / "figures" / "stage4_integration"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"e01_fixed_candidate_arms_{dataset}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {fig_path}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1"):
        print("usage: ... e01_fixed_candidate_arms <threads:0|1> [dataset ...]")
        sys.exit(1)
    nt = int(sys.argv[1])
    datasets = sys.argv[2:] if len(sys.argv) > 2 else DATASETS
    t0 = time.perf_counter()
    for ds in datasets:
        run_dataset(ds, nt)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
