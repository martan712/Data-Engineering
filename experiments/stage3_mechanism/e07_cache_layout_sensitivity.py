"""Stage 3 e07: cache/layout penalty of dim reordering vs natural order.

Single responsibility: quantify the wall-clock cost of non-natural dimension
orders (bond, pca) relative to natural order, split into the three places it
can live:

  1. IN-KERNEL access-pattern penalty (the headline metric): kernel ms/query
     per order, and kernel_penalty_vs_natural_pct = (kernel - kernel_nat) /
     kernel_nat.  A permuted order reads each dimension as one scattered
     64 B line instead of a sequential stream — this is where the "cache/
     layout sensitivity" actually shows up (2026-07-03 finding: the per-query
     pre-processing is ~0.1% of total, so the old prep/total "reorder
     fraction" carried no signal).
  2. PER-QUERY pre-processing (µs/query): order permutation (bond argsort /
     pca rotation of Q) + Qcum.  Timed by running the packing dispatch in
     isolation; the Runner's fused throughput mode pre-builds per-query
     inputs OUTSIDE its timing loop, so its ms/query is pure kernel time and
     total = kernel + prep.
  3. ONE-TIME per-corpus (index-build) costs, measured once on a fresh
     PackingCache and reported in seconds: corpus stats + natural panel
     packing (natural/bond arms) and PCA fit + a full ROTATED corpus packing
     (pca arm — 2x panel memory).  Correctly amortized out of per-query
     latency, but reported so the pca arm's extra index cost stays visible.

tau_seed resolution is deliberately EXCLUDED from the timed pre-processing:
the oracle policy computes a full exact MaxSim scan per query (an ablation
instrument, not deployable preprocessing — see thresholds.resolve_tau_seed),
and a real system uses self_bound (tau = -inf, zero cost) or an in-kernel
seed at the first checkpoint.  Including it would charge every order the
cost of a dense scan and swamp the reorder cost being measured.

The wall-clock instrument is the Stage 3b fused panel BOND kernel
(docs/stage3b_fused_panel_maxsim_kernel.md); dense baselines:
  dense_fused — fused panel brute (decision-gate baseline)
  dense_numpy — row-major NumPy/BLAS

Cross-run caution: dense wall-clock drifts a few percent between runs
(machine conditions); only within-run deltas are comparable.

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


def time_corpus_prep(flat_tokens: np.ndarray, doc_starts: np.ndarray) -> dict:
    """One-time per-corpus (index-build) costs in seconds, measured once on a
    FRESH PackingCache (the experiment's own cache is already warm): corpus
    stats (mean, in the constructor), the natural panel packing shared by the
    natural/bond arms, and the pca arm's PCA fit + rotated corpus packing.
    """
    t0 = time.perf_counter()
    pc = PackingCache(flat_tokens, doc_starts)
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    pc._get_panel_packing()
    t_pack = time.perf_counter() - t0

    t0 = time.perf_counter()
    pc.get_pca_rotation()
    t_pca_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    pc._get_panel_packing_rot()
    t_pack_rot = time.perf_counter() - t0

    del pc
    return {
        "corpus_stats_s": t_init,
        "panel_packing_s": t_pack,
        "pca_fit_s": t_pca_fit,
        "panel_packing_rot_s": t_pack_rot,
    }


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
            "strict_top_k_set_equal": rec.strict_top_k_set_equal,
            "boundary_tie_equivalent": rec.boundary_tie_equivalent,
            "agreement_failure_codes": rec.agreement_failure_codes,
        }
        arms.append(arm)

    # Headline metric: in-kernel penalty of each order vs natural.
    kernel_natural = arms[0]["ms_per_query_kernel"]
    for arm in arms:
        arm["kernel_penalty_vs_natural_pct"] = (
            100.0 * (arm["ms_per_query_kernel"] - kernel_natural) / kernel_natural
        )
        print(f"  order={arm['dimension_order']:<7}  "
              f"total={arm['ms_per_query_total']:.4f}ms  "
              f"prep={arm['ms_per_query_preprocess']*1e3:.1f}us  "
              f"kernel={arm['ms_per_query_kernel']:.4f}ms  "
              f"penalty_vs_natural={arm['kernel_penalty_vs_natural_pct']:+.1f}%  "
              f"recall={arm['recall_vs_exact_at_10']:.3f}")

    # One-time per-corpus (index-build) costs, measured once.
    corpus_prep = time_corpus_prep(flat_tokens, doc_starts)
    print(f"  one-time: stats={corpus_prep['corpus_stats_s']:.2f}s  "
          f"pack={corpus_prep['panel_packing_s']:.2f}s  "
          f"pca_fit={corpus_prep['pca_fit_s']:.2f}s  "
          f"pack_rot={corpus_prep['panel_packing_rot_s']:.2f}s")

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
            "strict_top_k_set_equal": rec.strict_top_k_set_equal,
            "boundary_tie_equivalent": rec.boundary_tie_equivalent,
            "agreement_failure_codes": rec.agreement_failure_codes,
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
        "corpus_prep_s": corpus_prep,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure: kernel latency / per-query prep (us) / kernel penalty vs natural.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e07_cache_layout_sensitivity_{dataset}.png"
    _save_figure(arms, brute_arms, corpus_prep, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], brute_arms: list[dict], corpus_prep: dict,
                 dataset: str, out_path: Path) -> None:
    orders    = [a["dimension_order"]               for a in arms]
    prep_us   = [a["ms_per_query_preprocess"] * 1e3 for a in arms]
    kernel_ms = [a["ms_per_query_kernel"]           for a in arms]
    penalty   = [a["kernel_penalty_vs_natural_pct"] for a in arms]
    colors    = [ORDER_COLORS.get(o, "gray")        for o in orders]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(orders))

    # Panel 1: kernel latency vs dense baselines (per-query prep is ~0.1% of
    # total — panel 2 shows it at its own scale instead of an invisible stack).
    ax1.bar(x, kernel_ms, color=colors, width=0.5)
    brute_colors = {"dense_fused": "#5c4a9e", "dense_numpy": "#888888"}
    for ba in brute_arms:
        c = brute_colors.get(ba["dimension_order"], "gray")
        ax1.axhline(ba["ms_per_query_total"], color=c, linestyle="--", linewidth=1.2,
                    label=ba["dimension_order"])
    ax1.set_xticks(x); ax1.set_xticklabels(orders)
    ax1.set_ylabel("kernel ms / query (best-of-10)")
    ax1.set_title("Kernel latency vs dense baselines")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(x, kernel_ms):
        ax1.text(xi, v * 1.01, f"{v:.3f}", ha="center", fontsize=8)

    # Panel 2: per-query pre-processing at its own (us) scale, with the
    # one-time per-corpus index-build costs annotated.
    ax2.bar(x, prep_us, color=colors, width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(orders)
    ax2.set_ylabel("preprocess us / query\n(order permutation + Qcum)")
    ax2.set_title("Per-query pre-processing (note: microseconds)")
    ax2.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax2.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(x, prep_us):
        ax2.text(xi, v * 1.01, f"{v:.1f}", ha="center", fontsize=8)
    ax2.text(0.02, 0.98,
             "one-time per corpus:\n"
             f"  panel packing  {corpus_prep['panel_packing_s']:.2f}s\n"
             f"  pca fit        {corpus_prep['pca_fit_s']:.2f}s\n"
             f"  rotated pack   {corpus_prep['panel_packing_rot_s']:.2f}s",
             transform=ax2.transAxes, va="top", ha="left", fontsize=7,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f6f5f0", edgecolor="#ccc"))

    # Panel 3: the headline metric — in-kernel access-pattern penalty of each
    # order relative to natural (same arithmetic, different memory pattern).
    ax3.bar(x, penalty, color=colors, width=0.5)
    ax3.axhline(0.0, color="#666666", linewidth=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(orders)
    ax3.set_ylabel("kernel-time penalty vs natural %")
    ax3.set_title("In-kernel layout penalty (the reorder cost)")
    ax3.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.margins(y=0.18)  # keep labels of negative bars inside the axes
    span = max(max(penalty) - min(penalty), 1.0)
    for xi, v in zip(x, penalty):
        off = 0.03 * span
        ax3.text(xi, v + (off if v >= 0 else -off), f"{v:+.1f}%",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

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
