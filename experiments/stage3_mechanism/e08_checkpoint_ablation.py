"""Stage 3 e08: checkpoint-set ablation on the fused BOND kernel (R3).

Single responsibility: sweep the fused doc-level BOND kernel's bound-checkpoint
set C (parameterized in the kernel as of the 2026-07-03 revision; default
{32, 64}) and measure, per C: wall-clock (1T + all cores), the kernel's own
docs-pruned rate, and the R2 NumPy checkpoint simulator's PREDICTED cells%
(padded wall-clock convention) — the instrument-aligned accounting that the
wide-block kernel cannot provide (it answers the upper-envelope question).

This closes RQ3 with numbers instead of a guess: e02's survival curves say
pruning only becomes possible late in the scan, so late checkpoints prune
more but save fewer dims per pruned doc, while early checkpoints are cheap
insurance that (on scifact) almost never fires.  The wall-clock-optimal C
per dataset is the exact-safe ceiling the fused algorithm can reach.

Candidate sets are subsets of {32, 48, 64, 96, 112} (e02 grid) spanning
early/mid/late singles, the Stage 3b default pair, a late pair, and a dense
grid.  Orders: natural (sequential access control) and bond (best pruning);
pca is omitted — it is access-pattern-equivalent to natural (pack-time
rotation, identity scan order).  Policy: oracle (isolates the checkpoint
effect from threshold quality).  Shrink: 1.0 (exact-safe arm; recall gate).

Datasets  : scifact (default; others via argv)
Queries   : 50 (same deterministic subsample as e01/e02)
k (top-k) : 10

Outputs per dataset:
  results/json/stage3_mechanism_e08_checkpoint_ablation_<dataset>.json
  results/figures/stage3_mechanism/e08_checkpoint_ablation_<dataset>.png

Usage:
    uv run python -m experiments.stage3_mechanism.e08_checkpoint_ablation [dataset ...]
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
from bondmaxsim.oracle.checkpoint_sim import simulate_fused_doc_pruning
from bondmaxsim.testbed.packing_cache import PackingCache
from bondmaxsim.testbed.runner import Runner, RunConfig
from bondmaxsim.testbed.thresholds import resolve_tau_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS   = ["scifact"]
CHECKPOINT_SETS = [
    (32,),
    (48,),
    (64,),
    (96,),
    (112,),
    (32, 64),            # Stage 3b default
    (64, 112),
    (32, 64, 96, 112),
]
ORDER_NAMES = ["natural", "bond"]
POLICY      = "oracle"
K_TOP       = 10
N_QUERIES   = 50
QUERY_SEED  = 42
N_REPEATS   = 5

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"

ORDER_COLORS = {"natural": "#2a78d6", "bond": "#eb6834"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample_queries(queries: list[np.ndarray], n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


def _cps_label(cps: tuple[int, ...]) -> str:
    return "{" + ",".join(str(c) for c in cps) + "}"


def _simulate_order(
    packing: PackingCache,
    queries: list[np.ndarray],
    order_name: str,
    cps: tuple[int, ...],
    k: int,
) -> tuple[float, float]:
    """Mean over queries of (predicted padded cells%, predicted docs pruned %)
    under the fused doc-checkpoint policy with the oracle tau."""
    cfg = RunConfig(dataset="", method="fused_panel_maxsim_bond",
                    dimension_order=order_name, threshold_policy=POLICY,
                    k=k, shrink=1.0)
    flat_eff = (packing.get_flat_tokens_rot() if order_name == "pca"
                else packing.flat_tokens)
    cells, pruned = [], []
    for q in queries:
        _, _, _, _, Q_eff, order = packing.dispatch_order_panel(q, order_name)
        Qcum = build_qcum(Q_eff, order)
        tau = resolve_tau_seed(cfg, q, order, order_name, packing)
        sim = simulate_fused_doc_pruning(
            Q_eff, flat_eff, packing.doc_starts, order, Qcum, cps, tau,
        )
        cells.append(sim.cells_scanned_pct_padded)
        pruned.append(100.0 * sim.docs_pruned_total / packing.num_docs)
    return float(np.mean(cells)), float(np.mean(pruned))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> None:
    print(f"\n=== e08 checkpoint ablation: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    runner = Runner(flat_tokens, doc_starts, queries)
    packing = PackingCache(flat_tokens, doc_starts)
    arms: list[dict] = []

    for order in ORDER_NAMES:
        for cps in CHECKPOINT_SETS:
            cfg = RunConfig(
                dataset=dataset,
                method="fused_panel_maxsim_bond",
                dimension_order=order,
                threshold_policy=POLICY,
                k=K_TOP,
                shrink=1.0,
                checkpoints=cps,
            )
            t1 = time.perf_counter()
            rec_1t = runner.throughput_mode(cfg, n_repeats=N_REPEATS, n_threads=1)
            rec_mt = runner.throughput_mode(cfg, n_repeats=N_REPEATS, n_threads=0)
            sim_cells, sim_pruned = _simulate_order(packing, queries, order, cps, K_TOP)
            elapsed = time.perf_counter() - t1

            recall = min(rec_1t.recall_vs_exact_at_10, rec_mt.recall_vs_exact_at_10)
            arm = {
                "dimension_order": order,
                "checkpoints": list(cps),
                "threshold_policy": POLICY,
                "recall_vs_exact_at_10": recall,
                "ms_per_query_1t": rec_1t.ms_per_query,
                "ms_per_query_mt": rec_mt.ms_per_query,
                "pruned_docs_pct_fused": rec_mt.pruned_docs_pct,
                "sim_cells_scanned_pct_padded": sim_cells,
                "sim_pruned_docs_pct": sim_pruned,
            }
            arms.append(arm)
            print(f"  [{order:<7}] C={_cps_label(cps):<16} recall={recall:.3f}  "
                  f"ms/q 1T={arm['ms_per_query_1t']:.2f} MT={arm['ms_per_query_mt']:.2f}  "
                  f"prune={arm['pruned_docs_pct_fused']:.2f}%  "
                  f"sim_cells={sim_cells:.1f}%  [{elapsed:.1f}s]")

            if recall < 1.0:
                raise RuntimeError(
                    f"Exact-agreement failed at shrink=1: order={order!r} "
                    f"checkpoints={cps} recall={recall}"
                )

    # Dense fused baseline (no checkpoints) at both thread counts.
    cfg_brute = RunConfig(dataset=dataset, method="fused_panel_maxsim_bond",
                          dimension_order="natural", threshold_policy="none",
                          k=K_TOP, shrink=1.0)
    rec_d1 = runner.brute_force_mode(cfg_brute, kind="fused", n_repeats=N_REPEATS,
                                     n_threads=1)
    rec_dm = runner.brute_force_mode(cfg_brute, kind="fused", n_repeats=N_REPEATS,
                                     n_threads=0)
    baseline = {"ms_per_query_1t": rec_d1.ms_per_query,
                "ms_per_query_mt": rec_dm.ms_per_query}
    print(f"  dense_fused        ms/q 1T={baseline['ms_per_query_1t']:.2f} "
          f"MT={baseline['ms_per_query_mt']:.2f}")

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_JSON / f"stage3_mechanism_e08_checkpoint_ablation_{dataset}.json"
    payload = {
        "experiment": "e08_checkpoint_ablation",
        "dataset": dataset,
        "method": "fused_panel_maxsim_bond",
        "n_docs": len(doc_starts),
        "total_tokens": flat_tokens.shape[0],
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k": K_TOP,
        "shrink": 1.0,
        "policy": POLICY,
        "orders": ORDER_NAMES,
        "checkpoint_sets": [list(c) for c in CHECKPOINT_SETS],
        "n_repeats_throughput": N_REPEATS,
        "dense_fused_baseline": baseline,
        "arms": arms,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e08_checkpoint_ablation_{dataset}.png"
    _save_figure(arms, baseline, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], baseline: dict, dataset: str, out_path: Path) -> None:
    labels = [_cps_label(tuple(a["checkpoints"])) for a in arms
              if a["dimension_order"] == ORDER_NAMES[0]]
    x = np.arange(len(labels))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    for oi, order in enumerate(ORDER_NAMES):
        oarms = [a for a in arms if a["dimension_order"] == order]
        color = ORDER_COLORS.get(order, "gray")
        ax1.bar(x + (oi - 0.5) * w, [a["ms_per_query_1t"] for a in oarms],
                width=w, color=color, alpha=0.85, label=f"{order} 1T")
        ax2.bar(x + (oi - 0.5) * w, [a["sim_cells_scanned_pct_padded"] for a in oarms],
                width=w, color=color, alpha=0.85, label=f"{order} predicted cells%")

    ax1.axhline(baseline["ms_per_query_1t"], color="#5c4a9e", linestyle="--",
                linewidth=1.2, label="dense fused 1T")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7, rotation=20)
    ax1.set_ylabel("ms / query (best-of-5, 1 thread)")
    ax1.set_title("Wall-clock vs checkpoint set")
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.axhline(100.0, color="#9a9992", linestyle=":", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7, rotation=20)
    ax2.set_ylabel("simulator-predicted cells % (padded)")
    ax2.set_title("Instrument-aligned accounting (R2 simulator)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=7, frameon=False)
    ax2.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"e08 checkpoint-set ablation — {dataset}  "
        f"(fused doc-level BOND, oracle policy, shrink=1, {N_QUERIES} queries)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {out_path}")


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    for ds in datasets:
        run_dataset(ds)
