"""Stage 3 e04: exact-safe cells/latency sweep across datasets and corpus sizes.

Single responsibility: for each dataset, run the wide-block kernel at shrink=1
with the oracle threshold policy over a range of corpus sizes (subsets of the
full corpus).  Tests whether more candidates (larger corpus) improve document
pruning — a downward cells%-vs-size slope would mean that small samples
understate the benefit of BOND pruning (exp-11 probe from the research plan).

Each "corpus size" is realised by taking the first N documents from the full
packed corpus.  Oracle policy is used so that the threshold is always the true
k-th score within the subset (eliminating threshold-quality variance as a
confound).  Throughput mode is also run so that both work reduction (cells%)
and wall-clock (ms/q) are measured at each size.

Wall-clock instruments are the Stage 3b fused panel kernels
(docs/stage3b_fused_panel_maxsim_kernel.md, revised 2026-07-03): the BOND arm
runs fused_panel_maxsim_bond at 1 thread AND all cores (post-revision the
all-cores arm sits at the dense DRAM floor, so 1T is the sensitive lens for
checkpoint overhead) and two dense baselines are run at each corpus size for
the speedup ratio — dense_fused (fused panel brute; the decision-gate
baseline, also 1T + all cores) and dense_numpy (row-major NumPy/BLAS, all
cores).  The fused kernel's own docs-pruned rate is recorded alongside the
wide-block accounting cells% (they answer different questions — plan R2).

Datasets  : scifact, nfcorpus, arguana, scidocs
Sizes     : 250, 500, 1000, 2000, full (dataset-dependent; capped at n_docs)
Order     : natural (isolates corpus-size effect)
Policy    : oracle
Shrink    : 1.0
Queries   : 50 (same deterministic subsample as e01/e02)
k (top-k) : 10

Outputs per dataset:
  results/json/stage3_mechanism_e04_exact_safe_pruning_<dataset>.json
  results/figures/stage3_mechanism/e04_exact_safe_pruning_<dataset>.png

Usage:
    uv run python -m experiments.stage3_mechanism.e04_exact_safe_pruning [dataset ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.testbed.runner import Runner, RunConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS   = ["scifact", "nfcorpus", "arguana", "scidocs"]
SIZE_STEPS = [250, 500, 1000, 2000]   # plus "full" always appended
ORDER      = "natural"
POLICY     = "oracle"
K_TOP      = 10
N_QUERIES  = 50
QUERY_SEED = 42
N_REPEATS  = 5

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample_queries(queries: list[np.ndarray], n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


def _subset_corpus(
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    n_docs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (flat_tokens_sub, doc_starts_sub) for the first n_docs documents."""
    end_token = int(doc_starts[n_docs]) if n_docs < len(doc_starts) else flat_tokens.shape[0]
    return flat_tokens[:end_token], doc_starts[:n_docs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> None:
    print(f"\n=== e04 exact-safe pruning sweep: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    full_n_docs = len(doc_starts)
    print(f"  full corpus: {full_n_docs} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    # Build size steps; skip sizes >= full corpus (full is always the last entry).
    sizes = [s for s in SIZE_STEPS if s < full_n_docs] + [full_n_docs]
    print(f"  sizes: {sizes}")

    arms: list[dict] = []

    for n_docs in sizes:
        ft_sub, ds_sub = _subset_corpus(flat_tokens, doc_starts, n_docs)
        runner = Runner(ft_sub, ds_sub, queries)
        cfg = RunConfig(
            dataset=dataset,
            method="wide_block_maxsim_bond",
            dimension_order=ORDER,
            threshold_policy=POLICY,
            k=K_TOP,
            shrink=1.0,
            candidate_budget=n_docs,
        )

        t1 = time.perf_counter()
        rec_acc = runner.accounting_mode(cfg)
        t_acc = time.perf_counter() - t1

        # Wall-clock BOND: Stage 3b fused panel kernel, all cores.
        cfg_fused = RunConfig(
            dataset=dataset,
            method="fused_panel_maxsim_bond",
            dimension_order=ORDER,
            threshold_policy=POLICY,
            k=K_TOP,
            shrink=1.0,
            candidate_budget=n_docs,
        )
        t1 = time.perf_counter()
        rec_thr = runner.throughput_mode(cfg_fused, n_repeats=N_REPEATS, n_threads=0)
        rec_thr_1t = runner.throughput_mode(cfg_fused, n_repeats=N_REPEATS, n_threads=1)
        t_thr = time.perf_counter() - t1

        # Dense baselines at this corpus size.
        cfg_brute = RunConfig(
            dataset=dataset,
            method="fused_panel_maxsim_bond",
            dimension_order="natural",
            threshold_policy="none",
            k=K_TOP,
            shrink=1.0,
            candidate_budget=n_docs,
        )
        rec_brute_fused = runner.brute_force_mode(cfg_brute, kind="fused", n_repeats=N_REPEATS,
                                                  n_threads=0)
        rec_brute_fused_1t = runner.brute_force_mode(cfg_brute, kind="fused", n_repeats=N_REPEATS,
                                                     n_threads=1)
        rec_brute_numpy = runner.brute_force_mode(cfg_brute, kind="numpy", n_repeats=N_REPEATS,
                                                  n_threads=0)

        arm = {
            "n_docs": n_docs,
            "total_tokens": ft_sub.shape[0],
            "recall_vs_exact_at_10": rec_acc.recall_vs_exact_at_10,
            "cells_scanned_pct": rec_acc.cells_scanned_pct,
            "pruned_docs_pct": rec_acc.pruned_docs_pct,
            "tokens_pruned_pct": rec_acc.tokens_pruned_pct,
            "pruned_docs_pct_fused": rec_thr.pruned_docs_pct,
            "ms_per_query_bond": rec_thr.ms_per_query,
            "ms_per_query_bond_1t": rec_thr_1t.ms_per_query,
            "ms_per_query_brute_fused": rec_brute_fused.ms_per_query,
            "ms_per_query_brute_fused_1t": rec_brute_fused_1t.ms_per_query,
            "ms_per_query_brute_numpy": rec_brute_numpy.ms_per_query,
            "speedup_vs_fused": rec_brute_fused.ms_per_query / rec_thr.ms_per_query
                                if rec_thr.ms_per_query > 0 else None,
            "speedup_vs_numpy": rec_brute_numpy.ms_per_query / rec_thr.ms_per_query
                                if rec_thr.ms_per_query > 0 else None,
            "qps_bond": rec_thr.qps,
        }
        arms.append(arm)
        print(f"  n_docs={n_docs:>5}  "
              f"recall={arm['recall_vs_exact_at_10']:.3f}  "
              f"cells={arm['cells_scanned_pct']:.2f}%  "
              f"fused_prune={arm['pruned_docs_pct_fused']:.2f}%  "
              f"bond={arm['ms_per_query_bond']:.3f}ms (1T={arm['ms_per_query_bond_1t']:.2f})  "
              f"dense_fused={arm['ms_per_query_brute_fused']:.3f}ms "
              f"(1T={arm['ms_per_query_brute_fused_1t']:.2f})  "
              f"dense_numpy={arm['ms_per_query_brute_numpy']:.3f}ms  "
              f"[acc={t_acc:.1f}s thr={t_thr:.1f}s]")

        if rec_acc.recall_vs_exact_at_10 < 1.0:
            raise RuntimeError(
                f"Exact-agreement failed at shrink=1: n_docs={n_docs} "
                f"recall={rec_acc.recall_vs_exact_at_10}"
            )

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_JSON / f"stage3_mechanism_e04_exact_safe_pruning_{dataset}.json"
    payload = {
        "experiment": "e04_exact_safe_pruning",
        "dataset": dataset,
        "method_accounting": "wide_block_maxsim_bond",
        "method_wallclock": "fused_panel_maxsim_bond",
        "full_n_docs": full_n_docs,
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k": K_TOP,
        "shrink": 1.0,
        "order": ORDER,
        "policy": POLICY,
        "n_repeats_throughput": N_REPEATS,
        "sizes": sizes,
        "arms": arms,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure: cells% and ms/q vs corpus size.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e04_exact_safe_pruning_{dataset}.png"
    _save_figure(arms, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], dataset: str, out_path: Path) -> None:
    xs         = [a["n_docs"] for a in arms]
    cells      = [a["cells_scanned_pct"] for a in arms]
    ms_bond    = [a["ms_per_query_bond"] for a in arms]
    ms_numpy   = [a["ms_per_query_brute_numpy"] for a in arms]
    ms_fused   = [a["ms_per_query_brute_fused"] for a in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(xs, cells, marker="o", color="#2a78d6", linewidth=2, label="BOND oracle")
    ax1.set_xlabel("corpus size (docs)")
    ax1.set_ylabel("cells scanned %")
    ax1.set_title("Algorithmic work vs corpus size")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8)
    ax1.grid(color="#e9e8e2", linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(xs, ms_bond,  marker="o", color="#2a78d6", linewidth=2, label="BOND fused (all cores)")
    ax2.plot(xs, ms_fused, marker="D", color="#5c4a9e", linewidth=2, label="dense fused (all cores)")
    ax2.plot(xs, ms_numpy, marker="^", color="#888888", linewidth=2, label="dense NumPy (all cores)")
    ax2.set_xlabel("corpus size (docs)")
    ax2.set_ylabel("ms / query (best-of-5)")
    ax2.set_title("Wall-clock latency vs corpus size")
    ax2.legend(fontsize=8)
    ax2.grid(color="#e9e8e2", linewidth=0.6)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"e04 exact-safe sweep — {dataset}  "
        f"(accounting: wide kernel; wall-clock: fused kernels; "
        f"oracle policy, natural order, shrink=1, {N_QUERIES} queries)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {out_path}")


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    for ds in datasets:
        run_dataset(ds)
