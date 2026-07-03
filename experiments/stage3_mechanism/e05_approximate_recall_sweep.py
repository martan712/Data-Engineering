"""Stage 3 e05: approximate recall sweep — shrink ∈ [0.5, 1.0], work AND wall-clock.

Single responsibility: trace the shrink recall/cost frontier per dataset ×
dimension order, on BOTH instruments (R5 — the RQ4 / gate-G2 decision
experiment):

  accounting arm : wide-block kernel, self_bound policy — cells_scanned_pct
                   vs recall@10 (the work-based frontier; upper envelope of
                   the mechanism, unchanged from the original e05).
  wall-clock arm : fused panel doc-level BOND kernel (Stage 3b, revised
                   2026-07-03), self_bound policy, all cores — ms/query vs
                   recall@10 against the fused DENSE all-cores baseline.
                   Gate G2 reads THIS frontier: a shrink<1 point dominates if
                   it has lower latency than dense at recall >= 0.99.

The approximate arm uses self_bound policy (tau rises from the documents'
own bounds; shrink < 1 scales the Cauchy-Schwarz residual, collapsing the
bound earlier).  Oracle and seed policies are omitted here: oracle would give
an unrealistically tight threshold and seed requires an external pre-filter.

Datasets  : scifact, nfcorpus
Orders    : natural, bond, pca
Policy    : self_bound (realistic approximate arm)
Shrink    : [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
Queries   : 50 (same deterministic subsample as e01/e02)
k (top-k) : 10

shrink = 1.0 is the exact-safe arm (recall must equal 1.0).
shrink < 1.0 is approximate — recall is measured vs exact@10 but is not
guaranteed (Stage 1 §3).

Outputs per dataset:
  results/json/stage3_mechanism_e05_approximate_recall_sweep_<dataset>.json
  results/figures/stage3_mechanism/e05_approximate_recall_sweep_<dataset>.png

Usage:
    uv run python -m experiments.stage3_mechanism.e05_approximate_recall_sweep [dataset ...]
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

DATASETS    = ["scifact", "nfcorpus"]
ORDER_NAMES = ["natural", "bond", "pca"]
POLICY      = "self_bound"
SHRINK_VALUES = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
K_TOP       = 10
N_QUERIES   = 50
QUERY_SEED  = 42
N_REPEATS   = 5   # wall-clock repeats (best-of), fused arms

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"

ORDER_COLORS  = {"natural": "#2a78d6", "bond": "#eb6834", "pca": "#1baf7a"}
ORDER_MARKERS = {"natural": "o", "bond": "s", "pca": "^"}


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
    print(f"\n=== e05 approximate recall sweep: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    runner = Runner(flat_tokens, doc_starts, queries)
    arms: list[dict] = []
    fused_arms: list[dict] = []

    for order in ORDER_NAMES:
        for shrink in SHRINK_VALUES:
            cfg = RunConfig(
                dataset=dataset,
                method="wide_block_maxsim_bond",
                dimension_order=order,
                threshold_policy=POLICY,
                k=K_TOP,
                shrink=shrink,
            )
            t1 = time.perf_counter()
            rec = runner.accounting_mode(cfg)
            elapsed = time.perf_counter() - t1

            arm = {
                "dimension_order": order,
                "shrink": shrink,
                "threshold_policy": POLICY,
                "recall_vs_exact_at_10": rec.recall_vs_exact_at_10,
                "cells_scanned_pct": rec.cells_scanned_pct,
                "pruned_docs_pct": rec.pruned_docs_pct,
                "tokens_pruned_pct": rec.tokens_pruned_pct,
            }
            arms.append(arm)
            exact_label = " [EXACT]" if shrink == 1.0 else ""
            print(f"  order={order:<7}  shrink={shrink:.2f}  "
                  f"recall={arm['recall_vs_exact_at_10']:.3f}  "
                  f"cells={arm['cells_scanned_pct']:.2f}%  "
                  f"prune_docs={arm['pruned_docs_pct']:.2f}%  "
                  f"{elapsed:.1f}s{exact_label}")

    # Wall-clock frontier (R5 / gate G2): fused doc-level BOND, all cores,
    # same order × shrink grid; compare against the fused dense baseline.
    for order in ORDER_NAMES:
        for shrink in SHRINK_VALUES:
            cfg_fused = RunConfig(
                dataset=dataset,
                method="fused_panel_maxsim_bond",
                dimension_order=order,
                threshold_policy=POLICY,
                k=K_TOP,
                shrink=shrink,
            )
            t1 = time.perf_counter()
            rec = runner.throughput_mode(cfg_fused, n_repeats=N_REPEATS, n_threads=0)
            elapsed = time.perf_counter() - t1

            farm = {
                "dimension_order": order,
                "shrink": shrink,
                "threshold_policy": POLICY,
                "recall_vs_exact_at_10": rec.recall_vs_exact_at_10,
                "ms_per_query_mt": rec.ms_per_query,
                "pruned_docs_pct_fused": rec.pruned_docs_pct,
            }
            fused_arms.append(farm)
            exact_label = " [EXACT]" if shrink == 1.0 else ""
            print(f"  fused[{order:<7}] shrink={shrink:.2f}  "
                  f"recall={farm['recall_vs_exact_at_10']:.3f}  "
                  f"ms/q MT={farm['ms_per_query_mt']:.2f}  "
                  f"prune_docs={farm['pruned_docs_pct_fused']:.2f}%  "
                  f"{elapsed:.1f}s{exact_label}")

            if shrink == 1.0 and farm["recall_vs_exact_at_10"] < 1.0:
                raise RuntimeError(
                    f"Exact-agreement failed at shrink=1 on the fused kernel: "
                    f"order={order!r} recall={farm['recall_vs_exact_at_10']}"
                )

    # Fused dense baseline (all cores) — the G2 reference latency.
    cfg_brute = RunConfig(dataset=dataset, method="fused_panel_maxsim_bond",
                          dimension_order="natural", threshold_policy="none",
                          k=K_TOP, shrink=1.0)
    rec_dense = runner.brute_force_mode(cfg_brute, kind="fused",
                                        n_repeats=N_REPEATS, n_threads=0)
    dense_ms_mt = rec_dense.ms_per_query
    print(f"  dense_fused MT baseline: {dense_ms_mt:.2f} ms/q")

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_JSON / f"stage3_mechanism_e05_approximate_recall_sweep_{dataset}.json"
    payload = {
        "experiment": "e05_approximate_recall_sweep",
        "dataset": dataset,
        "method_accounting": "wide_block_maxsim_bond",
        "method_wallclock": "fused_panel_maxsim_bond",
        "n_docs": len(doc_starts),
        "total_tokens": flat_tokens.shape[0],
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k": K_TOP,
        "policy": POLICY,
        "orders": ORDER_NAMES,
        "shrink_values": SHRINK_VALUES,
        "n_repeats_throughput": N_REPEATS,
        "dense_fused_ms_per_query_mt": dense_ms_mt,
        "arms": arms,
        "fused_arms": fused_arms,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure: cells% and ms/q vs recall@10 frontiers, one curve per order.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e05_approximate_recall_sweep_{dataset}.png"
    _save_figure(arms, fused_arms, dense_ms_mt, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], fused_arms: list[dict], dense_ms_mt: float,
                 dataset: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    def _frontier(ax, source, ykey, ylabel, title):
        for order in ORDER_NAMES:
            order_arms = [a for a in source if a["dimension_order"] == order]
            recall = [a["recall_vs_exact_at_10"] for a in order_arms]
            ys     = [a[ykey]                    for a in order_arms]
            shrink = [a["shrink"]                for a in order_arms]

            color  = ORDER_COLORS.get(order, "gray")
            marker = ORDER_MARKERS.get(order, "o")
            ax.plot(recall, ys, color=color, marker=marker,
                    linewidth=2, markersize=6, label=order)
            for r, y, s in zip(recall, ys, shrink):
                ax.annotate(
                    f"{s:.2f}", (r, y),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=7, color=color,
                )
        ax.set_xlabel("recall@10 vs exact MaxSim")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.05)
        ax.axvline(1.0, color="#9a9992", linestyle=":", linewidth=0.8)
        ax.grid(color="#e9e8e2", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    _frontier(ax1, arms, "cells_scanned_pct", "cells scanned %",
              "Work frontier (wide-block accounting)")
    ax1.set_ylim(0, 105)

    _frontier(ax2, fused_arms, "ms_per_query_mt", "ms / query (all cores)",
              "Wall-clock frontier (fused BOND) — gate G2")
    ax2.axhline(dense_ms_mt, color="#5c4a9e", linestyle="--", linewidth=1.2,
                label="dense fused")
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"e05 shrink sweep — {dataset}  (self_bound, shrink ∈ {SHRINK_VALUES})",
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
