"""Stage 5 e02: paired significance test for the e01 nDCG@10 inversions.

Single responsibility: several approximate arms in e01 land marginally ABOVE
the exact dense scan on nDCG@10 (partitioned on arguana, partitioned@32 and
faiss@100 on scidocs).  This driver reproduces the e01 quality pass (untimed,
same arms and parameters) and asks whether any such difference is systematic:
per-query nDCG@10 for each arm and a paired sign-flip permutation test of the
arm minus the exact dense scan (bondmaxsim.eval.significance).

Arms: dense_fused (baseline), partitioned@{16,32}, faiss@100 — the arms whose
recall_vs_exact@10 < 1 puts them in inversion range; the exact-safe BOND arm
is identical to dense by construction and is not re-tested.

Output:
  results/json/stage5_corect_e02_paired_significance.json

Usage:
    uv run python -m experiments.stage5_corect.e02_paired_significance \
        [dataset ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_ids, load_qrels_tsv, qrels_path
from bondmaxsim.data.loader import load_dataset, load_eval_queries
from bondmaxsim.eval.qrels import per_query_ndcg_at_10
from bondmaxsim.eval.significance import paired_permutation_test
from bondmaxsim.kernels.fused_panel import (
    load_fused_panel_kernel,
    run_fused_panel_brute,
)
from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline
from bondmaxsim.partitioned_scan import PartitionedFusedScan
from bondmaxsim.testbed.packing_cache import PackingCache

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
K_RETRIEVE = 100                 # as e01; nDCG@10 uses the ranking prefix
CHECKPOINTS = (112,)
BOUND = "tight"
PART_NPROBES = (16, 32)
FAISS_BUDGET = 100
FAISS_NPROBE = 32
N_PERMUTATIONS = 20_000

RESULTS_JSON = REPO_ROOT / "results" / "json"
FAISS_CACHE = REPO_ROOT / "data" / "faiss_indexes"


def _to_run(results, query_ids, corpus_ids) -> dict[str, dict[str, float]]:
    """Rank-based run dict, identical to e01 (see its tie-handling note)."""
    run: dict[str, dict[str, float]] = {}
    for qid, (ids, _) in zip(query_ids, results):
        n = len(ids)
        run[qid] = {corpus_ids[int(d)]: float(n - i)
                    for i, d in enumerate(ids)}
    return run


def run_dataset(dataset: str) -> dict:
    flat, starts, _ = load_dataset(dataset)
    queries, query_ids = load_eval_queries(dataset)
    qrels = load_qrels_tsv(qrels_path(dataset))
    corpus_ids = load_ids(dataset)["corpus_ids"]
    print(f"\n=== e02 {dataset}: {len(starts)} docs, {len(queries)} queries ===")

    lib = load_fused_panel_kernel()
    pk = PackingCache(flat, starts)
    pdat, goff, doff, gds, _ = pk._get_panel_packing()
    Qs = [np.ascontiguousarray(q, dtype=np.float32) for q in queries]

    t0 = time.perf_counter()
    arms: dict[str, list] = {}
    arms["dense_fused"] = [
        run_fused_panel_brute(lib, pdat, goff, doff, gds, Q, K_RETRIEVE,
                              n_threads=0) for Q in Qs]

    part = PartitionedFusedScan(flat, starts)
    part.build()
    for p in PART_NPROBES:
        arms[f"partitioned@{p}"] = [
            part.search(lib, q, k=K_RETRIEVE, nprobe=p,
                        checkpoints=CHECKPOINTS, bound=BOUND,
                        n_threads=0, scanner="bond")[:2]
            for q in queries]

    ivf = FaissIVFBaseline(flat, starts)
    ivf.build(cache_path=str(FAISS_CACHE / f"{dataset}.faiss"))
    ivf.index.nprobe = FAISS_NPROBE
    arms[f"faiss@{FAISS_BUDGET}"] = [
        ivf.topk(q, k=K_RETRIEVE, candidate_budget=FAISS_BUDGET)[:2]
        for q in queries]
    print(f"  retrieval pass done in {time.perf_counter() - t0:.1f}s")

    ndcg = {name: per_query_ndcg_at_10(
                _to_run(res, query_ids, corpus_ids), qrels)
            for name, res in arms.items()}

    tests = {}
    for name in [a for a in arms if a != "dense_fused"]:
        t = paired_permutation_test(ndcg[name], ndcg["dense_fused"],
                                    n_permutations=N_PERMUTATIONS)
        tests[name] = t
        print(f"  {name:<16} vs dense: mean dnDCG@10 = {t['mean_diff']:+.4f}  "
              f"changed {t['n_changed']}/{t['n_queries']} "
              f"(help {t['n_helped']} / hurt {t['n_hurt']})  "
              f"p = {t['p_value']:.3f}")

    return {
        "n_queries": len(queries),
        "mean_ndcg_at_10": {n: float(np.mean(list(v.values())))
                            for n, v in ndcg.items()},
        "paired_vs_dense": tests,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    payload = {
        "experiment": "e02_paired_significance",
        "purpose": "paired sign-flip permutation test (per-query nDCG@10, "
                   "two-sided) of each approximate arm against the exact "
                   "dense scan — are the e01 nDCG inversions systematic?",
        "k_retrieve": K_RETRIEVE,
        "kernel": {"bound": BOUND, "checkpoints": list(CHECKPOINTS),
                   "nprobes": list(PART_NPROBES)},
        "faiss": {"candidate_budget": FAISS_BUDGET, "nprobe": FAISS_NPROBE},
        "n_permutations": N_PERMUTATIONS,
        "datasets": {},
    }
    t0 = time.perf_counter()
    for ds in datasets:
        payload["datasets"][ds] = run_dataset(ds)
    out = RESULTS_JSON / "stage5_corect_e02_paired_significance.json"
    if out.exists():
        prev = json.loads(out.read_text())
        prev_ds = prev.get("datasets", {})
        prev_ds.update(payload["datasets"])
        payload["datasets"] = prev_ds
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON: {out}\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
