"""Stage 3 e03: dimension-order ablation — accounting + wall-clock across datasets.

Single responsibility: compare natural / bond / pca dimension orders on
cells_scanned_pct (wide-block ACCOUNTING kernel, true algorithmic work) and
ms_per_query (Stage 3b fused panel BOND kernel, wall-clock), and confirm that
shrink=1 recall is 1.0 for all orders (correctness regression, Stage 1 §8
item 5).

Instruments (docs/stage3b_fused_panel_maxsim_kernel.md §6) — THREE-ARM
isolation on the identical microkernel, so any wall-clock difference is
attributable to the pruning mechanism alone:
  - algorithmic work : wide-block accounting kernel (unchanged Stage 2
    instrument; live-set cells counter, token+doc pruning stats)
  - BOND doc-level   : fused_panel_maxsim_bond (document bound checkpoints
    at dims 32/64), 1 thread and all cores
  - BOND token-level : fused_panel_maxsim_bond_token (same + Stage 1 §2.4
    token domination test, lane masks, dead-panel skip)
  - dense baselines  : fused_panel_maxsim_brute (decision-gate baseline) and
    NumPy/BLAS row-major, each at 1 thread and all cores

The legacy wide-block THROUGHPUT kernel and the wide-block dense scan
("brute_pdx") are deliberately absent: both are superseded by the fused panel
kernels (Stage 3b §1 — the old wall-clock numbers measured codegen artifacts,
not the mechanism).

One arm = dataset × order × policy.  Policy is fixed to 'oracle' so that
threshold variance does not confound the order comparison; a separate
threshold-policy comparison is in e06.

Datasets  : scifact, nfcorpus, arguana, scidocs
Orders    : natural, bond, pca  (+ dense baselines)
Policy    : oracle
Shrink    : 1.0 (exact-safe arm only)
Queries   : 50
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
    "natural":     "#2a78d6",
    "bond":        "#eb6834",
    "pca":         "#1baf7a",
    "dense_fused": "#5c4a9e",
    "dense_numpy": "#888888",
}

# (level, method) pairs for the two BOND pruning granularities.
BOND_LEVELS = [
    ("doc",   "fused_panel_maxsim_bond"),
    ("token", "fused_panel_maxsim_bond_token"),
]

# Dense baselines: (kind, label) — each is run at 1 thread and all cores.
DENSE_BASELINES = [
    ("fused", "dense_fused"),   # fused panel brute: decision-gate baseline
    ("numpy", "dense_numpy"),   # row-major NumPy/BLAS
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

    # BOND arms: accounting (algorithmic work) once per order + fused
    # wall-clock at both pruning levels (doc / token).
    for order in ORDER_NAMES:
        cfg_acc = RunConfig(
            dataset=dataset,
            method="wide_block_maxsim_bond",
            dimension_order=order,
            threshold_policy=POLICY,
            k=K_TOP,
            shrink=1.0,
        )
        t1 = time.perf_counter()
        rec_acc = runner.accounting_mode(cfg_acc)
        t_acc = time.perf_counter() - t1

        for level, method in BOND_LEVELS:
            cfg_fused = RunConfig(
                dataset=dataset,
                method=method,
                dimension_order=order,
                threshold_policy=POLICY,
                k=K_TOP,
                shrink=1.0,
            )
            t1 = time.perf_counter()
            rec_1t = runner.throughput_mode(cfg_fused, n_repeats=N_REPEATS, n_threads=1)
            rec_mt = runner.throughput_mode(cfg_fused, n_repeats=N_REPEATS, n_threads=0)
            t_thr = time.perf_counter() - t1

            recall = min(rec_acc.recall_vs_exact_at_10,
                         rec_1t.recall_vs_exact_at_10,
                         rec_mt.recall_vs_exact_at_10)
            arm = {
                "arm_type": "bond",
                "prune_level": level,
                "dimension_order": order,
                "threshold_policy": POLICY,
                "recall_vs_exact_at_10": recall,
                "cells_scanned_pct": rec_acc.cells_scanned_pct,
                "pruned_docs_pct": rec_acc.pruned_docs_pct,
                "tokens_pruned_pct": rec_acc.tokens_pruned_pct,
                "pruned_docs_pct_fused": rec_mt.pruned_docs_pct,
                "tokens_pruned_pct_fused": rec_mt.tokens_pruned_pct,
                "ms_per_query_1t": rec_1t.ms_per_query,
                "ms_per_query_mt": rec_mt.ms_per_query,
                "qps_mt": rec_mt.qps,
            }
            arms.append(arm)
            tok = (f"fused_tok_prune={arm['tokens_pruned_pct_fused']:.2f}%  "
                   if arm["tokens_pruned_pct_fused"] is not None else "")
            print(f"  bond_{level:<5}[{order:<7}]  recall={recall:.3f}  "
                  f"cells={arm['cells_scanned_pct']:.2f}%  "
                  f"fused_doc_prune={arm['pruned_docs_pct_fused']:.2f}%  {tok}"
                  f"ms/q 1T={arm['ms_per_query_1t']:.3f}  MT={arm['ms_per_query_mt']:.3f}  "
                  f"[acc={t_acc:.1f}s thr={t_thr:.1f}s]")

            if recall < 1.0:
                raise RuntimeError(
                    f"Exact-agreement failed at shrink=1: level={level!r} "
                    f"order={order!r} recall={recall}"
                )

    # Dense baselines (no pruning), each at 1 thread and all cores.
    cfg_brute = RunConfig(
        dataset=dataset,
        method="fused_panel_maxsim_bond",
        dimension_order="natural",
        threshold_policy="none",
        k=K_TOP,
        shrink=1.0,
    )
    for kind, label in DENSE_BASELINES:
        t1 = time.perf_counter()
        rec_1t = runner.brute_force_mode(cfg_brute, kind=kind, n_repeats=N_REPEATS,
                                         n_threads=1)
        rec_mt = runner.brute_force_mode(cfg_brute, kind=kind, n_repeats=N_REPEATS,
                                         n_threads=0)
        elapsed = time.perf_counter() - t1
        arm = {
            "arm_type": "dense",
            "prune_level": None,
            "dimension_order": label,
            "threshold_policy": "none",
            "recall_vs_exact_at_10": min(rec_1t.recall_vs_exact_at_10,
                                         rec_mt.recall_vs_exact_at_10),
            "cells_scanned_pct": None,
            "pruned_docs_pct": None,
            "tokens_pruned_pct": None,
            "pruned_docs_pct_fused": None,
            "tokens_pruned_pct_fused": None,
            "ms_per_query_1t": rec_1t.ms_per_query,
            "ms_per_query_mt": rec_mt.ms_per_query,
            "qps_mt": rec_mt.qps,
        }
        arms.append(arm)
        print(f"  {label:<15}  recall={arm['recall_vs_exact_at_10']:.3f}  "
              f"ms/q 1T={arm['ms_per_query_1t']:.3f}  MT={arm['ms_per_query_mt']:.3f}  "
              f"[{elapsed:.1f}s]")

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_JSON / f"stage3_mechanism_e03_order_ablation_{dataset}.json"
    payload = {
        "experiment": "e03_order_ablation",
        "dataset": dataset,
        "method_accounting": "wide_block_maxsim_bond",
        "methods_wallclock": [m for _, m in BOND_LEVELS],
        "n_docs": len(doc_starts),
        "total_tokens": flat_tokens.shape[0],
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k": K_TOP,
        "shrink": 1.0,
        "policy": POLICY,
        "orders": ORDER_NAMES,
        "baselines": [label for _, label in DENSE_BASELINES],
        "n_repeats_throughput": N_REPEATS,
        "arms": arms,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e03_order_ablation_{dataset}.png"
    _save_figure(arms, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], dataset: str, out_path: Path) -> None:
    bond_arms  = [a for a in arms if a["arm_type"] == "bond"]
    dense_arms = [a for a in arms if a["arm_type"] == "dense"]
    all_arms   = bond_arms + dense_arms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2),
                                   gridspec_kw={"width_ratios": [1, 2]})

    # Left: algorithmic work (accounting kernel; one bar per ORDER — the
    # accounting numbers are identical for both fused pruning levels).
    acc_arms = [a for a in bond_arms if a["prune_level"] == "doc"]
    xb = np.arange(len(acc_arms))
    cells = [a["cells_scanned_pct"] for a in acc_arms]
    acc_colors = [ORDER_COLORS.get(a["dimension_order"], "gray") for a in acc_arms]
    ax1.bar(xb, cells, color=acc_colors, width=0.5)
    ax1.set_xticks(xb)
    ax1.set_xticklabels([a["dimension_order"] for a in acc_arms], fontsize=9)
    ax1.set_ylabel("cells scanned %")
    ax1.set_title("Algorithmic work (accounting kernel)")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)
    for xi, v in zip(xb, cells):
        ax1.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8)

    # Right: wall-clock, paired 1T / all-cores bars per arm.
    def _label(a: dict) -> str:
        if a["arm_type"] == "bond":
            return f"{a['prune_level']}\n{a['dimension_order']}"
        return a["dimension_order"].replace("_", "\n")

    labels = [_label(a) for a in all_arms]
    colors = [ORDER_COLORS.get(a["dimension_order"], "gray") for a in all_arms]
    x = np.arange(len(all_arms))
    w = 0.38
    ms_1t = [a["ms_per_query_1t"] for a in all_arms]
    ms_mt = [a["ms_per_query_mt"] for a in all_arms]
    ax2.bar(x - w / 2, ms_1t, width=w, color=colors, alpha=0.45, label="1 thread")
    ax2.bar(x + w / 2, ms_mt, width=w, color=colors, label="all cores")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("ms / query (best-of-5)")
    ax2.set_title("Wall-clock latency — dense vs doc-prune vs token-prune "
                  "(identical microkernel)")
    ax2.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=8, frameon=False)
    for xi, v in zip(x - w / 2, ms_1t):
        ax2.text(xi, v * 1.01, f"{v:.1f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, ms_mt):
        ax2.text(xi, v * 1.01, f"{v:.1f}", ha="center", fontsize=7)

    fig.suptitle(
        f"e03 dimension-order ablation — {dataset}  "
        f"(oracle policy, shrink=1, {N_QUERIES} queries)",
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
