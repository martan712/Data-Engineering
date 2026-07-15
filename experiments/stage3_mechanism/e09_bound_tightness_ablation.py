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

The default checkpoint set {32, 64} answers this at EARLY checkpoints, where
e02's survival curves say almost nothing is prunable — and indeed (2026-07-03)
the third outcome held: both bounds lose to dense because neither prunes.
e08 then showed the interesting regime is LATE checkpoints (C={112} and
supersets), where the exact-safe kernel prunes 88-98% of documents and beats
dense on 3 of 4 datasets.  R12a reruns THIS comparison at the e08-winning sets:
that is where pruning fires, so that is where the bound choice actually decides.
The winning bound becomes the doc-level default (bond2002 doc §6 criteria).

Datasets  : scifact, nfcorpus, arguana, scidocs
Bounds    : tight, cheap  (methods fused_panel_maxsim_bond{,_cheap})
Orders    : natural, bond, pca  (bound slack grows with energy concentration)
Checkpts  : {32,64} (early, control), {112}, {64,112}, {32,64,96,112} (e08 winners)
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
# {32,64} = early control (recorded default; both bounds prune ~nothing).
# The rest are the e08-winning late sets where pruning fires and the bound
# choice actually decides (R12a).
CHECKPOINT_SETS = [
    (32, 64),
    (112,),
    (64, 112),
    (32, 64, 96, 112),
]
POLICY      = "oracle"
K_TOP       = 10
N_REPEATS   = 10

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"

BOUND_COLORS = {"tight": "#2a78d6", "cheap": "#eb6834"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cps_label(cps: tuple[int, ...]) -> str:
    return "{" + ",".join(str(c) for c in cps) + "}"


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

    for cps in CHECKPOINT_SETS:
        for order in ORDER_NAMES:
            for bound, method in BOUNDS.items():
                cfg = RunConfig(
                    dataset=dataset,
                    method=method,
                    dimension_order=order,
                    threshold_policy=POLICY,
                    k=K_TOP,
                    shrink=1.0,
                    checkpoints=cps,
                )
                rec = runner.throughput_mode(cfg, n_repeats=N_REPEATS, n_threads=0)

                arm = {
                    "checkpoints": list(cps),
                    "bound": bound,
                    "dimension_order": order,
                    "ms_per_query": rec.ms_per_query,
                    "qps": rec.qps,
                    "pruned_docs_pct": rec.pruned_docs_pct,
                    "recall_vs_exact_at_10": rec.recall_vs_exact_at_10,
                    "strict_top_k_set_equal": rec.strict_top_k_set_equal,
                    "boundary_tie_equivalent": rec.boundary_tie_equivalent,
                    "agreement_failure_codes": rec.agreement_failure_codes,
                }
                arms.append(arm)
                print(f"  C={_cps_label(cps):<16} order={order:<7} bound={bound:<5}  "
                      f"ms/q={rec.ms_per_query:.4f}  "
                      f"prune_docs={rec.pruned_docs_pct:.2f}%  "
                      f"recall={rec.recall_vs_exact_at_10:.3f}")

                if not rec.boundary_tie_equivalent:
                    raise RuntimeError(
                        f"Verified exact gate failed at shrink=1: bound={bound!r} "
                        f"order={order!r} checkpoints={cps} "
                        f"failures={rec.agreement_failure_codes}"
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
        "strict_top_k_set_equal": rec_dense.strict_top_k_set_equal,
        "boundary_tie_equivalent": rec_dense.boundary_tie_equivalent,
        "agreement_failure_codes": rec_dense.agreement_failure_codes,
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
        "checkpoint_sets": [list(c) for c in CHECKPOINT_SETS],
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
    ncol = len(CHECKPOINT_SETS)

    fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol + 1, 7),
                             squeeze=False)

    for ci, cps in enumerate(CHECKPOINT_SETS):
        ax1, ax2 = axes[0][ci], axes[1][ci]
        cps_list = list(cps)
        for k, bound in enumerate(BOUNDS):
            sel = [a for a in arms
                   if a["bound"] == bound and a["checkpoints"] == cps_list]
            sel = sorted(sel, key=lambda a: ORDER_NAMES.index(a["dimension_order"]))
            ms    = [a["ms_per_query"]    for a in sel]
            prune = [a["pruned_docs_pct"] for a in sel]
            off = (k - 0.5) * width
            ax1.bar(x + off, ms, width=width, label=bound, color=BOUND_COLORS[bound])
            ax2.bar(x + off, prune, width=width, label=bound, color=BOUND_COLORS[bound])
            for xi, v in zip(x + off, ms):
                ax1.text(xi, v * 1.01, f"{v:.1f}", ha="center", fontsize=6)
            for xi, v in zip(x + off, prune):
                ax2.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=6)

        ax1.axhline(dense_arm["ms_per_query"], color="#5c4a9e", linestyle="--",
                    linewidth=1.2, label="dense_fused")
        ax1.set_xticks(x); ax1.set_xticklabels(ORDER_NAMES, fontsize=7)
        ax1.set_title(f"C={_cps_label(cps)}", fontsize=9)
        ax2.set_xticks(x); ax2.set_xticklabels(ORDER_NAMES, fontsize=7)
        ax2.set_ylim(0, 105)
        for ax in (ax1, ax2):
            ax.grid(axis="y", color="#e9e8e2", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        if ci == 0:
            ax1.set_ylabel(f"ms / query (best-of-{N_REPEATS})")
            ax2.set_ylabel("pruned docs %")
            ax1.legend(fontsize=7)

    fig.suptitle(
        f"e09 bound-tightness ablation — {dataset}  "
        f"(fused doc-level BOND, oracle policy, shrink=1, all cores)\n"
        f"top: wall-clock (tight vs cheap; dashed = dense);  "
        f"bottom: pruned docs %",
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
