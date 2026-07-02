"""Stage 3 e02: two-level pruning-rate (survival) curves from the wide kernel.

Single responsibility: for each dataset x threshold policy (self_bound /
oracle / seed) x dimension order (natural / bond / ada), run the wide-block
kernel in accounting mode at shrink=1 over a query subsample and collect the
per-fetch-boundary document-survival and token-survival curves from the
Runner's e02 side channel (Runner.last_block_doc_live /
Runner.last_block_token_live), plus the scalar tokens_pruned_pct /
pruned_docs_pct / cells_scanned_pct from the ResultRecord.

Derived per arm (bondmaxsim.oracle.bound_trajectory helpers):
  - dims-to-prune-50% / 90% of documents
  - early-token-pruning rate (fraction of tokens pruned before D/4 dims,
    Stage 1 §6: exp-10 skips bound evaluation before D/4)

NumPy cross-check: for one arm per dataset (oracle policy, natural order) the
survival curves are also computed with the pure-NumPy prefix-grid engine
(bondmaxsim.oracle.bound_trajectory.survival_trajectories) on a 10-query
subset and compared against the kernel curves on the same subset.  The two
engines use different bookkeeping (prefix grid vs live-set compaction with
per-document token sets), so agreement is qualitative, not bit-exact.

Datasets  : scifact, nfcorpus (default; override via sys.argv[1:])
Queries   : 50 (deterministic seed=42, e01 conventions)
k (top-k) : 10

Outputs per dataset:
  results/json/stage3_mechanism_e02_pruning_rate_<dataset>.json
  results/figures/stage3_mechanism/e02_pruning_rate_<dataset>.png

Usage:
    uv run python -m experiments.stage3_mechanism.e02_pruning_rate [dataset ...]
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
from bondmaxsim.data.packing import DEFAULT_FETCH
from bondmaxsim.oracle.bound_trajectory import (
    dims_to_prune_pct,
    early_token_pruning_rate,
    survival_trajectories,
)
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores
from bondmaxsim.ordering.orders import natural_order
from bondmaxsim.threshold.policies import oracle_threshold
from bondmaxsim.testbed.runner import Runner, RunConfig

# ---------------------------------------------------------------------------
# Constants (e01 conventions)
# ---------------------------------------------------------------------------

DATASETS = ["scifact", "nfcorpus"]
POLICIES = ["self_bound", "oracle", "seed"]
ORDER_NAMES = ["natural", "bond", "ada"]
K_TOP = 10
N_QUERIES = 50   # subsample for tractability
QUERY_SEED = 42
D = 128
N_CROSSCHECK_QUERIES = 10   # NumPy prefix-grid engine subset

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"

# Fixed categorical hues (policy) + linestyle (order) so all 9 lines per panel
# stay distinguishable without 9 hues.
POLICY_COLORS = {"self_bound": "#2a78d6", "oracle": "#1baf7a", "seed": "#eb6834"}
ORDER_STYLES  = {"natural": "-", "bond": "--", "ada": ":"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample_queries(queries: list[np.ndarray], n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


def fetch_boundary_dims(D: int) -> np.ndarray:
    """Dims-scanned value at each fetch boundary the kernel actually reaches.

    The wide kernel evaluates bounds after each DEFAULT_FETCH block; boundary b
    corresponds to min(cumsum(DEFAULT_FETCH)[:b+1], D) dims scanned.  Entries
    past the first boundary reaching D are never visited (side-channel slots
    stay zero) and are dropped here.
    """
    cum = np.cumsum(DEFAULT_FETCH.astype(np.int64))
    n_used = int(np.searchsorted(cum, D)) + 1
    return np.minimum(cum[:n_used], D)


def run_arm(
    runner: Runner,
    dataset: str,
    policy: str,
    order: str,
    n_docs: int,
    total_tokens: int,
    dims_grid: np.ndarray,
) -> dict:
    """Run one wide-kernel accounting arm and derive its survival curves."""
    cfg = RunConfig(
        dataset=dataset,
        method="wide_block_maxsim_bond",
        dimension_order=order,
        threshold_policy=policy,
        k=K_TOP,
        shrink=1.0,
    )
    record = runner.accounting_mode(cfg)
    n_used = len(dims_grid)
    doc_live   = runner.last_block_doc_live[:n_used]    # mean over queries
    token_live = runner.last_block_token_live[:n_used]

    doc_survival   = (doc_live / n_docs).astype(np.float64)
    token_survival = (token_live / total_tokens).astype(np.float64)

    return {
        "threshold_policy": policy,
        "dimension_order": order,
        "recall_vs_exact_at_10": record.recall_vs_exact_at_10,
        "cells_scanned_pct": record.cells_scanned_pct,
        "pruned_docs_pct": record.pruned_docs_pct,
        "tokens_pruned_pct": record.tokens_pruned_pct,
        "doc_survival": doc_survival.tolist(),
        "token_survival": token_survival.tolist(),
        "dims_to_prune_50pct_docs": dims_to_prune_pct(doc_survival, dims_grid, 50),
        "dims_to_prune_90pct_docs": dims_to_prune_pct(doc_survival, dims_grid, 90),
        "early_token_pruning_rate": early_token_pruning_rate(
            token_survival, dims_grid, D
        ),
    }


# ---------------------------------------------------------------------------
# NumPy cross-check (oracle policy, natural order)
# ---------------------------------------------------------------------------

def numpy_crosscheck(
    runner: Runner,
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    queries: list[np.ndarray],
    dataset: str,
    dims_grid: np.ndarray,
    n_docs: int,
    total_tokens: int,
) -> dict:
    """Compare kernel vs pure-NumPy survival curves on a small query subset.

    Arm: oracle threshold policy, natural order.  Both engines are run on the
    SAME N_CROSSCHECK_QUERIES queries; the NumPy engine
    (survival_trajectories, prefix grid = the kernel's fetch boundaries) uses
    the exact k-th score as a fixed tau, while the kernel additionally prunes
    tokens inside per-document live sets — so agreement is qualitative.
    """
    sub = queries[:N_CROSSCHECK_QUERIES]

    # Kernel curves on the subset (fresh Runner so the 50-query side channel
    # for the main arms is not clobbered).
    sub_runner = Runner(flat_tokens, doc_starts, sub)
    cfg = RunConfig(
        dataset=dataset,
        method="wide_block_maxsim_bond",
        dimension_order="natural",
        threshold_policy="oracle",
        k=K_TOP,
        shrink=1.0,
    )
    sub_runner.accounting_mode(cfg)
    n_used = len(dims_grid)
    kern_doc   = sub_runner.last_block_doc_live[:n_used] / n_docs
    kern_token = sub_runner.last_block_token_live[:n_used] / total_tokens

    # NumPy prefix-grid engine on the same subset with the same grid.
    np_doc_sum   = np.zeros(n_used, dtype=np.float64)
    np_token_sum = np.zeros(n_used, dtype=np.float64)
    for query in sub:
        exact_scores = exact_maxsim_scores(query, flat_tokens, doc_starts)
        tau = oracle_threshold(exact_scores, K_TOP)
        doc_frac, token_frac = survival_trajectories(
            query, flat_tokens, doc_starts, natural_order(query), tau, dims_grid
        )
        np_doc_sum   += doc_frac
        np_token_sum += token_frac
    np_doc   = np_doc_sum / len(sub)
    np_token = np_token_sum / len(sub)

    return {
        "arm": "oracle/natural",
        "n_queries": len(sub),
        "dims_grid": dims_grid.tolist(),
        "kernel_doc_survival": kern_doc.tolist(),
        "numpy_doc_survival": np_doc.tolist(),
        "kernel_token_survival": kern_token.tolist(),
        "numpy_token_survival": np_token.tolist(),
        "doc_max_abs_diff": float(np.max(np.abs(kern_doc - np_doc))),
        "token_max_abs_diff": float(np.max(np.abs(kern_token - np_token))),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def save_figure(
    dims_grid: np.ndarray,
    arms: list[dict],
    dataset: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    panels = [("doc_survival", "Document survival"),
              ("token_survival", "Token survival")]

    for ax, (key, title) in zip(axes, panels):
        for arm in arms:
            policy, order = arm["threshold_policy"], arm["dimension_order"]
            ax.plot(
                dims_grid, arm[key],
                color=POLICY_COLORS[policy],
                linestyle=ORDER_STYLES[order],
                linewidth=2.0,
                label=f"{policy} / {order}",
            )
        ax.axvline(D // 4, color="#9a9992", linestyle=":", linewidth=0.8)
        ax.annotate("D/4", xy=(D // 4, 1.0), xytext=(D // 4 + 2, 0.97),
                    fontsize=7, color="#52514e")
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.set_xlabel("dimensions scanned")
        ax.set_xlim(0, D)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color="#e9e8e2", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("fraction live")
    axes[1].legend(fontsize=7, ncol=1, loc="upper right", framealpha=0.9)

    fig.suptitle(f"e02 pruning-rate survival curves — {dataset} "
                 f"(wide kernel, shrink=1, {N_QUERIES} queries)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> None:
    print(f"\n=== e02 pruning rate: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    n_docs = len(doc_starts)
    total_tokens = flat_tokens.shape[0]
    print(f"  corpus: {n_docs} docs, {total_tokens} tokens, {len(queries)} queries")

    dims_grid = fetch_boundary_dims(D)
    runner = Runner(flat_tokens, doc_starts, queries)

    arms: list[dict] = []
    for policy in POLICIES:
        for order in ORDER_NAMES:
            t1 = time.perf_counter()
            arm = run_arm(runner, dataset, policy, order, n_docs, total_tokens, dims_grid)
            arms.append(arm)
            print(f"  policy={policy:<10} order={order:<7} "
                  f"{time.perf_counter() - t1:6.1f}s  "
                  f"recall={arm['recall_vs_exact_at_10']:.3f}  "
                  f"d50={arm['dims_to_prune_50pct_docs']:>3}  "
                  f"d90={arm['dims_to_prune_90pct_docs']:>3}  "
                  f"earlyTok={arm['early_token_pruning_rate']:.3f}  "
                  f"cells={arm['cells_scanned_pct']:.2f}%")

    # NumPy cross-check on one arm.
    print("  numpy cross-check (oracle/natural) ...", end=" ", flush=True)
    t1 = time.perf_counter()
    crosscheck = numpy_crosscheck(
        runner, flat_tokens, doc_starts, queries, dataset,
        dims_grid, n_docs, total_tokens,
    )
    print(f"{time.perf_counter() - t1:.1f}s  "
          f"doc_max_abs_diff={crosscheck['doc_max_abs_diff']:.4f}  "
          f"token_max_abs_diff={crosscheck['token_max_abs_diff']:.4f}")

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_JSON / f"stage3_mechanism_e02_pruning_rate_{dataset}.json"
    payload = {
        "experiment": "e02_pruning_rate",
        "dataset": dataset,
        "method": "wide_block_maxsim_bond",
        "n_docs": n_docs,
        "total_tokens": total_tokens,
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k": K_TOP,
        "shrink": 1.0,
        "D": D,
        "fetch_boundary_dims": dims_grid.tolist(),
        "policies": POLICIES,
        "orders": ORDER_NAMES,
        "arms": arms,
        "numpy_crosscheck": crosscheck,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Write figure.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e02_pruning_rate_{dataset}.png"
    save_figure(dims_grid, arms, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    for ds in datasets:
        run_dataset(ds)
