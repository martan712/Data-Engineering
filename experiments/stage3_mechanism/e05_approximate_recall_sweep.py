"""Stage 3 e05: approximate recall sweep — shrink ∈ [0.5, 1.0] vs recall frontier.

Single responsibility: for each dataset × dimension order × shrink value, run
the wide-block kernel in accounting mode with the self_bound threshold policy
and record cells_scanned_pct vs recall_vs_exact@10.  This traces the
cells%-recall Pareto frontier for each order arm, isolating the approximate
arm (shrink < 1) from the exact arm (shrink = 1, Stage 1 §3).

The approximate arm uses self_bound policy (the tau_seed is derived from the
document's own running BOND upper-bound at the first fetch boundary, multiplied
by shrink).  Oracle and seed policies are omitted here: oracle would give an
unrealistically tight threshold and seed requires an external pre-filter.

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

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_JSON / f"stage3_mechanism_e05_approximate_recall_sweep_{dataset}.json"
    payload = {
        "experiment": "e05_approximate_recall_sweep",
        "dataset": dataset,
        "method": "wide_block_maxsim_bond",
        "n_docs": len(doc_starts),
        "total_tokens": flat_tokens.shape[0],
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k": K_TOP,
        "policy": POLICY,
        "orders": ORDER_NAMES,
        "shrink_values": SHRINK_VALUES,
        "arms": arms,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure: cells% vs recall@10 frontier, one curve per order.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e05_approximate_recall_sweep_{dataset}.png"
    _save_figure(arms, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], dataset: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for order in ORDER_NAMES:
        order_arms = [a for a in arms if a["dimension_order"] == order]
        recall = [a["recall_vs_exact_at_10"] for a in order_arms]
        cells  = [a["cells_scanned_pct"]    for a in order_arms]
        shrink = [a["shrink"]               for a in order_arms]

        color  = ORDER_COLORS.get(order, "gray")
        marker = ORDER_MARKERS.get(order, "o")
        ax.plot(recall, cells, color=color, marker=marker,
                linewidth=2, markersize=6, label=order)

        # Annotate shrink values.
        for r, c, s in zip(recall, cells, shrink):
            ax.annotate(
                f"{s:.2f}", (r, c),
                textcoords="offset points", xytext=(4, 2),
                fontsize=7, color=color,
            )

    ax.set_xlabel("recall@10 vs exact MaxSim")
    ax.set_ylabel("cells scanned %")
    ax.set_title(f"e05 cells%–recall frontier — {dataset}  (self_bound, shrink sweep)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 105)
    ax.axvline(1.0, color="#9a9992", linestyle=":", linewidth=0.8)
    ax.grid(color="#e9e8e2", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {out_path}")


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    for ds in datasets:
        run_dataset(ds)
