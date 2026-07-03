"""Stage 3 e09: bound-tightness ablation — tight Cauchy-Schwarz vs cheap query-only.

Single responsibility: measure whether the TIGHT per-document bound
    UB = sum_i max_j (P_ij + resq_i * resd_j)
pays for its doc-side residual machinery (sumsq fused into the hot loop, one
sqrt per lane per checkpoint, resd loads in the UB reduction) against the
CHEAP query-only bound
    UB = sum_i max_j P_ij + sum_i resq_i
(resd_j <= 1 relaxation) on the SAME fused doc-level kernel, same checkpoints,
same threading — so any difference is attributable to the bound alone.

Motivation: the original BOND paper (de Vries et al., SIGMOD 2002) compared a
cheap query-only criterion (H_q) against a tighter per-vector one (H_h) and
found the tighter bound's extra pruning "not large enough for the additional
bookkeeping to pay off"; its complex Euclidean bound (E_v) lost outright on
CPU cost.  Our tight arm is the E_v-analog; the cheap arm is the H_q-analog.
Full analysis: docs/bond2002_bound_cost_analysis.md.

Expectations (either outcome is informative):
  - cheap wins wall-clock  -> the e07 checkpoint overhead was largely the
    tight bound's price, not MaxSim's price; adopt cheap as default.
  - tight wins wall-clock  -> the residual envelope's extra pruning pays for
    itself; MaxSim genuinely diverges from the 2002 lesson (resd falls fast
    under energy-concentrating orders, unlike the histogram setting).

Both arms are exact-safe at shrink=1 (UB_cheap >= UB_tight can only prune
LESS); the recall gate raises on any violation.

Datasets  : scifact, nfcorpus, arguana, scidocs
Bounds    : tight, cheap  (methods fused_panel_maxsim_bond{,_cheap})
Orders    : natural, bond, pca  (bound slack grows with energy concentration)
Policy    : oracle (fixed tau — isolates the bound; shared-tau noise absent)
Shrink    : 1.0
Queries   : all queries (tight timing statistics, as e07)
k (top-k) : 10
N_REPEATS : 10
Baseline  : dense_fused (fused panel brute, all cores)

Outputs per dataset:
  results/json/stage3_mechanism_e09_bound_tightness_ablation_<dataset>.json
  results/figures/stage3_mechanism/e09_bound_tightness_ablation_<dataset>.png

Usage:
    uv run python -m experiments.stage3_mechanism.e09_bound_tightness_ablation [dataset ...]
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
BOUNDS      = {"tight": "fused_panel_maxsim_bond",
               "cheap": "fused_panel_maxsim_bond_cheap"}
ORDER_NAMES = ["natural", "bond", "pca"]
POLICY      = "oracle"
K_TOP       = 10
N_REPEATS   = 10

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"

BOUND_COLORS = {"tight": "#2a78d6", "cheap": "#eb6834"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> None:
    print(f"\n=== e09 bound-tightness ablation: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries (all)")

    runner = Runner(flat_tokens, doc_starts, queries)
    arms: list[dict] = []

    for order in ORDER_NAMES:
        for bound, method in BOUNDS.items():
            cfg = RunConfig(
                dataset=dataset,
                method=method,
                dimension_order=order,
                threshold_policy=POLICY,
                k=K_TOP,
                shrink=1.0,
            )
            rec = runner.throughput_mode(cfg, n_repeats=N_REPEATS, n_threads=0)

            arm = {
                "bound": bound,
                "dimension_order": order,
                "ms_per_query": rec.ms_per_query,
                "qps": rec.qps,
                "pruned_docs_pct": rec.pruned_docs_pct,
                "recall_vs_exact_at_10": rec.recall_vs_exact_at_10,
            }
            arms.append(arm)
            print(f"  order={order:<7} bound={bound:<5}  "
                  f"ms/q={rec.ms_per_query:.4f}  "
                  f"prune_docs={rec.pruned_docs_pct:.2f}%  "
                  f"recall={rec.recall_vs_exact_at_10:.3f}")

            if rec.recall_vs_exact_at_10 < 1.0:
                raise RuntimeError(
                    f"Exact-agreement failed at shrink=1: bound={bound!r} "
                    f"order={order!r} recall={rec.recall_vs_exact_at_10}"
                )

    # Dense baseline (no checkpoints, no bound) at the same thread count.
    cfg_brute = RunConfig(
        dataset=dataset,
        method="fused_panel_maxsim_bond",
        dimension_order="natural",
        threshold_policy="none",
        k=K_TOP,
        shrink=1.0,
    )
    rec_dense = runner.brute_force_mode(cfg_brute, kind="fused",
                                        n_repeats=N_REPEATS, n_threads=0)
    dense_arm = {
        "bound": "none",
        "dimension_order": "dense_fused",
        "ms_per_query": rec_dense.ms_per_query,
        "qps": rec_dense.qps,
        "pruned_docs_pct": 0.0,
        "recall_vs_exact_at_10": rec_dense.recall_vs_exact_at_10,
    }
    print(f"  dense_fused             ms/q={rec_dense.ms_per_query:.4f}  (baseline)")

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = (
        RESULTS_JSON / f"stage3_mechanism_e09_bound_tightness_ablation_{dataset}.json"
    )
    payload = {
        "experiment": "e09_bound_tightness_ablation",
        "dataset": dataset,
        "methods": BOUNDS,
        "n_docs": len(doc_starts),
        "total_tokens": flat_tokens.shape[0],
        "n_queries": len(queries),
        "k": K_TOP,
        "policy": POLICY,
        "shrink": 1.0,
        "n_repeats": N_REPEATS,
        "orders": ORDER_NAMES,
        "arms": arms,
        "dense_arm": dense_arm,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e09_bound_tightness_ablation_{dataset}.png"
    _save_figure(arms, dense_arm, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], dense_arm: dict, dataset: str, out_path: Path) -> None:
    x = np.arange(len(ORDER_NAMES))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for k, bound in enumerate(BOUNDS):
        sel = [a for a in arms if a["bound"] == bound]
        sel = sorted(sel, key=lambda a: ORDER_NAMES.index(a["dimension_order"]))
        ms    = [a["ms_per_query"]    for a in sel]
        prune = [a["pruned_docs_pct"] for a in sel]
        off = (k - 0.5) * width
        ax1.bar(x + off, ms, width=width, label=bound, color=BOUND_COLORS[bound])
        ax2.bar(x + off, prune, width=width, label=bound, color=BOUND_COLORS[bound])
        for xi, v in zip(x + off, ms):
            ax1.text(xi, v * 1.01, f"{v:.2f}", ha="center", fontsize=7)
        for xi, v in zip(x + off, prune):
            ax2.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=7)

    ax1.axhline(dense_arm["ms_per_query"], color="#5c4a9e", linestyle="--",
                linewidth=1.2, label="dense_fused")
    ax1.set_xticks(x); ax1.set_xticklabels(ORDER_NAMES)
    ax1.set_ylabel(f"ms / query (best-of-{N_REPEATS}, kernel-only)")
    ax1.set_title("Wall-clock: tight vs cheap bound")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.set_xticks(x); ax2.set_xticklabels(ORDER_NAMES)
    ax2.set_ylabel("pruned docs %")
    ax2.set_title("Pruning lost to the looser bound")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"e09 bound-tightness ablation — {dataset}  "
        f"(fused doc-level BOND, oracle policy, shrink=1, all cores)",
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
