"""Stage 3 e07: cache/layout penalty of dim reordering vs natural order.

Single responsibility: quantify the wall-clock overhead introduced by
non-natural dimension orders (bond, pca) relative to natural order by
measuring both the per-query pre-processing cost (building Q_eff, order
permutation, and Qcum) and the total throughput (ms/query), then
computing the reorder fraction = pre-processing time / total time.

tau_seed resolution is deliberately EXCLUDED from the timed pre-processing:
the oracle policy computes a full exact MaxSim scan per query (an ablation
instrument, not deployable preprocessing — see thresholds.resolve_tau_seed),
and a real system uses self_bound (tau = -inf, zero cost) or an in-kernel
seed at the first checkpoint.  Including it would charge every order the
cost of a dense scan and swamp the reorder cost being measured.

A large reorder fraction means the SIMD kernel spends most of its time on
bookkeeping rather than arithmetic; a small fraction means dimension ordering
is essentially free and the cells% reduction translates to a proportional
latency reduction.

The wall-clock instrument is the Stage 3b fused panel BOND kernel
(docs/stage3b_fused_panel_maxsim_kernel.md); dense baselines:
  dense_fused — fused panel brute (decision-gate baseline)
  dense_numpy — row-major NumPy/BLAS

For each order, the pre-processing cost is timed by running the packing
dispatch in isolation (without the kernel call).  The Runner's fused
throughput mode pre-builds per-query inputs OUTSIDE its timing loop, so its
ms/query is pure kernel time; total = kernel + pre-processing, and
reorder_fraction = prep / total.

Datasets  : scifact, nfcorpus, arguana, scidocs
Orders    : natural, bond, pca  (+ two brute-force baselines)
Policy    : oracle (eliminates threshold variance)
Shrink    : 1.0
Queries   : all queries (no subsample — tighter timing statistics)
k (top-k) : 10
N_REPEATS : 10 (more repeats for tighter timing)

Outputs per dataset:
  results/json/stage3_mechanism_e07_cache_layout_sensitivity_<dataset>.json
  results/figures/stage3_mechanism/e07_cache_layout_sensitivity_<dataset>.png

Usage:
    uv run python -m experiments.stage3_mechanism.e07_cache_layout_sensitivity [dataset ...]
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
from bondmaxsim.data.packing import build_qcum
from bondmaxsim.testbed.runner import Runner, RunConfig
from bondmaxsim.testbed.packing_cache import PackingCache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS    = ["scifact", "nfcorpus", "arguana", "scidocs"]
ORDER_NAMES = ["natural", "bond", "pca"]
POLICY      = "oracle"
K_TOP       = 10
N_REPEATS   = 10

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"

ORDER_COLORS = {"natural": "#2a78d6", "bond": "#eb6834", "pca": "#1baf7a"}


# ---------------------------------------------------------------------------
# Pre-processing timer
# ---------------------------------------------------------------------------

def time_preprocessing(
    packing: PackingCache,
    queries: list[np.ndarray],
    order: str,
    n_repeats: int,
) -> float:
    """Return best-of-n_repeats total wall-clock for dispatch + Qcum over all
    queries, in seconds.  tau_seed resolution is excluded (see module
    docstring).  Does NOT call the C++ kernel.
    """
    # Warm up (triggers lazy builds so timing reflects steady state).
    for query in queries:
        pd_, go, do_, gds, Q_eff, ord_ = packing.dispatch_order_panel(query, order)
        build_qcum(Q_eff, ord_)

    best_s = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for query in queries:
            pd_, go, do_, gds, Q_eff, ord_ = packing.dispatch_order_panel(query, order)
            build_qcum(Q_eff, ord_)
        best_s = min(best_s, time.perf_counter() - t0)
    return best_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> None:
    print(f"\n=== e07 cache/layout sensitivity: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    # Use all queries for tighter timing statistics.
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries (all)")

    runner = Runner(flat_tokens, doc_starts, queries)
    # Access packing cache for pre-processing timing.
    packing = PackingCache(flat_tokens, doc_starts)

    arms: list[dict] = []
    brute_arms: list[dict] = []

    for order in ORDER_NAMES:
        cfg = RunConfig(
            dataset=dataset,
            method="fused_panel_maxsim_bond",
            dimension_order=order,
            threshold_policy=POLICY,
            k=K_TOP,
            shrink=1.0,
        )

        # Kernel-only throughput, all cores.  The Runner's fused throughput
        # mode pre-builds all per-query inputs OUTSIDE its timing loop
        # (fused_modes.run_fused_bond_mode), so rec.ms_per_query is pure
        # kernel time; total = kernel + separately-timed pre-processing.
        rec = runner.throughput_mode(cfg, n_repeats=N_REPEATS, n_threads=0)

        # Pre-processing only (no kernel, no tau_seed).
        prep_best_s = time_preprocessing(packing, queries, order, N_REPEATS)

        ms_per_query_kernel = rec.ms_per_query
        ms_per_query_prep   = prep_best_s / len(queries) * 1e3
        ms_per_query_total  = ms_per_query_kernel + ms_per_query_prep
        reorder_fraction = ms_per_query_prep / ms_per_query_total if ms_per_query_total > 0 else 0.0

        arm = {
            "dimension_order": order,
            "ms_per_query_total": ms_per_query_total,
            "ms_per_query_preprocess": ms_per_query_prep,
            "ms_per_query_kernel": ms_per_query_kernel,
            "reorder_fraction": reorder_fraction,
            "qps": rec.qps,
            "recall_vs_exact_at_10": rec.recall_vs_exact_at_10,
        }
        arms.append(arm)
        print(f"  order={order:<7}  "
              f"total={ms_per_query_total:.4f}ms  "
              f"prep={ms_per_query_prep:.4f}ms  "
              f"kernel={ms_per_query_kernel:.4f}ms  "
              f"reorder_frac={reorder_fraction:.3f}  "
              f"recall={rec.recall_vs_exact_at_10:.3f}")

    # Brute-force baselines (no pre-processing timing needed — brute has no reorder step).
    cfg_brute = RunConfig(
        dataset=dataset,
        method="fused_panel_maxsim_bond",
        dimension_order="natural",
        threshold_policy="none",
        k=K_TOP,
        shrink=1.0,
    )
    for kind, label in [("fused", "dense_fused"), ("numpy", "dense_numpy")]:
        rec = runner.brute_force_mode(cfg_brute, kind=kind, n_repeats=N_REPEATS,
                                      n_threads=0)
        brute_arms.append({
            "dimension_order": label,
            "ms_per_query_total": rec.ms_per_query,
            "ms_per_query_preprocess": 0.0,
            "ms_per_query_kernel": rec.ms_per_query,
            "reorder_fraction": 0.0,
            "qps": rec.qps,
            "recall_vs_exact_at_10": rec.recall_vs_exact_at_10,
        })
        print(f"  order={label:<12}  total={rec.ms_per_query:.4f}ms  (brute, no reorder)")

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = (
        RESULTS_JSON / f"stage3_mechanism_e07_cache_layout_sensitivity_{dataset}.json"
    )
    payload = {
        "experiment": "e07_cache_layout_sensitivity",
        "dataset": dataset,
        "method": "fused_panel_maxsim_bond",
        "n_docs": len(doc_starts),
        "total_tokens": flat_tokens.shape[0],
        "n_queries": len(queries),
        "k": K_TOP,
        "policy": POLICY,
        "shrink": 1.0,
        "n_repeats": N_REPEATS,
        "orders": ORDER_NAMES,
        "baselines": ["dense_fused", "dense_numpy"],
        "arms": arms,
        "brute_arms": brute_arms,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure: stacked bar of prep vs kernel latency per order.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e07_cache_layout_sensitivity_{dataset}.png"
    _save_figure(arms, brute_arms, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], brute_arms: list[dict], dataset: str, out_path: Path) -> None:
    orders    = [a["dimension_order"]          for a in arms]
    prep_ms   = [a["ms_per_query_preprocess"]  for a in arms]
    kernel_ms = [a["ms_per_query_kernel"]      for a in arms]
    total_ms  = [a["ms_per_query_total"]       for a in arms]
    frac      = [a["reorder_fraction"]         for a in arms]
    colors    = [ORDER_COLORS.get(o, "gray")   for o in orders]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(orders))
    ax1.bar(x, kernel_ms, label="kernel", color=colors, width=0.5, alpha=0.85)
    ax1.bar(x, prep_ms,   label="preprocess", color=colors, width=0.5,
            bottom=kernel_ms, alpha=0.4, hatch="//")
    # Horizontal reference lines for brute baselines.
    brute_colors = {"dense_fused": "#5c4a9e", "dense_numpy": "#888888"}
    for ba in brute_arms:
        c = brute_colors.get(ba["dimension_order"], "gray")
        ax1.axhline(ba["ms_per_query_total"], color=c, linestyle="--", linewidth=1.2,
                    label=ba["dimension_order"])
    ax1.set_xticks(x); ax1.set_xticklabels(orders)
    ax1.set_ylabel("ms / query (best-of-10)")
    ax1.set_title("Latency: kernel + preprocess vs brute baselines")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(x, total_ms):
        ax1.text(xi, v * 1.01, f"{v:.3f}", ha="center", fontsize=8)

    ax2.bar(x, [f * 100 for f in frac], color=colors, width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(orders)
    ax2.set_ylabel("reorder fraction %\n(preprocess / total)")
    ax2.set_title("Pre-processing share of total latency")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax2.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(x, frac):
        ax2.text(xi, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=8)

    fig.suptitle(
        f"e07 cache/layout sensitivity — {dataset}  "
        f"(fused panel BOND kernel, oracle policy, shrink=1, {N_REPEATS} repeats)",
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
