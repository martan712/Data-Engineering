"""Stage 3 e03: dimension-order ablation — accounting + throughput across datasets.

Single responsibility: compare natural / bond / pca dimension orders on
cells_scanned_pct (accounting mode, algorithmic work) and ms_per_query
(throughput mode, wall-clock), and confirm that shrink=1 recall is 1.0 for
all orders (correctness regression, Stage 1 §8 item 5).

Brute-force baselines are included so the wall-clock decision gate can be
evaluated (Stage 3b: the gate baseline is the strongest dense kernel, and the
threading factor must be an explicit arm — see
docs/stage3b_fused_panel_maxsim_kernel.md §6):
  brute_pdx       — wide_block_maxsim_brute: same columnar layout, no pruning
  brute_numpy     — exact_maxsim_topk, OpenBLAS default threads (all cores)
  brute_numpy_1t  — same, BLAS pinned to 1 thread (thread-fair vs kernels)
  brute_fused_1t  — fused_panel_maxsim_brute, 1 thread (Stage 3b kernel)
  brute_fused_mt  — same, OpenMP default threads (decision-gate baseline)

One arm = dataset × order × policy.  Policy is fixed to 'oracle' so that
threshold variance does not confound the order comparison; a separate
threshold-policy comparison is in e06.

Datasets  : scifact, nfcorpus, arguana, scidocs
Orders    : natural, bond, pca  (+ two brute-force baselines)
Policy    : oracle
Shrink    : 1.0 (exact-safe arm only)
Queries   : 50 (accounting), 50 (throughput — same subsample)
k (top-k) : 10

Outputs per dataset:
  results/json/stage3_mechanism_e03_order_ablation_<dataset>.json
  results/figures/stage3_mechanism/e03_order_ablation_<dataset>.png

Usage:
    uv run python -m experiments.stage3_mechanism.e03_order_ablation [dataset ...]
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

DATASETS    = ["scifact", "nfcorpus", "arguana", "scidocs"]
ORDER_NAMES = ["natural", "bond", "pca"]
POLICY      = "oracle"
K_TOP       = 10
N_QUERIES   = 50
QUERY_SEED  = 42
N_REPEATS   = 5   # throughput repeats (best-of)

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"

ORDER_COLORS = {
    "natural":        "#2a78d6",
    "bond":           "#eb6834",
    "pca":            "#1baf7a",
    "brute_pdx":      "#888888",
    "brute_numpy":    "#bbbbbb",
    "brute_numpy_1t": "#999999",
    "brute_fused_1t": "#5c4a9e",
    "brute_fused_mt": "#8a76d0",
}

# (kind, n_threads, label) triples for Runner.brute_force_mode.
BRUTE_ARMS = [
    ("pdx",   1, "brute_pdx"),
    ("numpy", 0, "brute_numpy"),      # OpenBLAS default = all cores
    ("numpy", 1, "brute_numpy_1t"),   # thread-fair vs single-threaded kernels
    ("fused", 1, "brute_fused_1t"),   # Stage 3b fused panel kernel
    ("fused", 0, "brute_fused_mt"),   # decision-gate baseline (all cores)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample_queries(queries: list[np.ndarray], n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> None:
    print(f"\n=== e03 order ablation: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    runner = Runner(flat_tokens, doc_starts, queries)
    arms: list[dict] = []

    for order in ORDER_NAMES:
        cfg = RunConfig(
            dataset=dataset,
            method="wide_block_maxsim_bond",
            dimension_order=order,
            threshold_policy=POLICY,
            k=K_TOP,
            shrink=1.0,
        )

        # Accounting mode.
        t1 = time.perf_counter()
        rec_acc = runner.accounting_mode(cfg)
        t_acc = time.perf_counter() - t1

        # Throughput mode.
        t1 = time.perf_counter()
        rec_thr = runner.throughput_mode(cfg, n_repeats=N_REPEATS)
        t_thr = time.perf_counter() - t1

        arm = {
            "arm_type": "bond",
            "dimension_order": order,
            "threshold_policy": POLICY,
            "recall_vs_exact_at_10": rec_acc.recall_vs_exact_at_10,
            "cells_scanned_pct": rec_acc.cells_scanned_pct,
            "pruned_docs_pct": rec_acc.pruned_docs_pct,
            "tokens_pruned_pct": rec_acc.tokens_pruned_pct,
            "ms_per_query": rec_thr.ms_per_query,
            "qps": rec_thr.qps,
        }
        arms.append(arm)
        print(f"  order={order:<7}  "
              f"recall={arm['recall_vs_exact_at_10']:.3f}  "
              f"cells={arm['cells_scanned_pct']:.2f}%  "
              f"prune_docs={arm['pruned_docs_pct']:.2f}%  "
              f"tok_prune={arm['tokens_pruned_pct']:.2f}%  "
              f"ms/q={arm['ms_per_query']:.3f}  "
              f"[acc={t_acc:.1f}s thr={t_thr:.1f}s]")

        if rec_acc.recall_vs_exact_at_10 < 1.0:
            raise RuntimeError(
                f"Exact-agreement failed at shrink=1: "
                f"order={order!r} recall={rec_acc.recall_vs_exact_at_10}"
            )

    # Brute-force baselines — needed for the Stage 3 wall-clock decision gate.
    cfg_brute = RunConfig(
        dataset=dataset,
        method="wide_block_maxsim_bond",
        dimension_order="natural",
        threshold_policy="none",
        k=K_TOP,
        shrink=1.0,
    )
    for kind, n_threads, label in BRUTE_ARMS:
        t1 = time.perf_counter()
        rec = runner.brute_force_mode(cfg_brute, kind=kind, n_repeats=N_REPEATS,
                                      n_threads=n_threads)
        elapsed = time.perf_counter() - t1
        arm = {
            "arm_type": "brute",
            "dimension_order": label,
            "threshold_policy": "none",
            "recall_vs_exact_at_10": rec.recall_vs_exact_at_10,
            "cells_scanned_pct": None,
            "pruned_docs_pct": None,
            "tokens_pruned_pct": None,
            "ms_per_query": rec.ms_per_query,
            "qps": rec.qps,
        }
        arms.append(arm)
        print(f"  order={label:<12}  "
              f"recall={arm['recall_vs_exact_at_10']:.3f}  "
              f"ms/q={arm['ms_per_query']:.3f}  [{elapsed:.1f}s]")

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_JSON / f"stage3_mechanism_e03_order_ablation_{dataset}.json"
    payload = {
        "experiment": "e03_order_ablation",
        "dataset": dataset,
        "method": "wide_block_maxsim_bond",
        "n_docs": len(doc_starts),
        "total_tokens": flat_tokens.shape[0],
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k": K_TOP,
        "shrink": 1.0,
        "policy": POLICY,
        "orders": ORDER_NAMES,
        "baselines": [label for _, _, label in BRUTE_ARMS],
        "n_repeats_throughput": N_REPEATS,
        "arms": arms,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure: grouped bar chart — cells% and ms/q per order.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e03_order_ablation_{dataset}.png"
    _save_figure(arms, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], dataset: str, out_path: Path) -> None:
    bond_arms  = [a for a in arms if a["arm_type"] == "bond"]
    brute_arms = [a for a in arms if a["arm_type"] == "brute"]

    all_arms  = bond_arms + brute_arms
    labels    = [a["dimension_order"] for a in all_arms]
    cells_raw = [a["cells_scanned_pct"] for a in all_arms]
    ms_q      = [a["ms_per_query"] for a in all_arms]
    colors    = [ORDER_COLORS.get(l, "gray") for l in labels]
    # Brute arms have no cells_scanned_pct; show 100% as reference.
    cells     = [v if v is not None else 100.0 for v in cells_raw]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(all_arms))

    ax1.bar(x[:len(bond_arms)], cells[:len(bond_arms)],
            color=colors[:len(bond_arms)], width=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=10, fontsize=8)
    ax1.set_ylabel("cells scanned %")
    ax1.set_title("Algorithmic work (accounting, BOND arms only)")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(x[:len(bond_arms)], cells[:len(bond_arms)]):
        ax1.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8)

    ax2.bar(x, ms_q, color=colors, width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=10, fontsize=8)
    ax2.set_ylabel("ms / query (best-of-5)")
    ax2.set_title("Wall-clock latency (all arms + baselines)")
    ax2.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax2.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(x, ms_q):
        ax2.text(xi, v * 1.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        f"e03 dimension-order ablation — {dataset}  "
        f"(wide kernel, oracle policy, shrink=1, {N_QUERIES} queries)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {out_path}")


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    for ds in datasets:
        run_dataset(ds)
