"""Stage 3 e01: Bound-slack vs dimension prefix (UB_d(k)/score trajectories).

Single responsibility: for each dataset/order combination, aggregate
UB_d(k)/score(q,d) slack ratios over a subsample of queries and report
how quickly the Cauchy-Schwarz bound tightens.

Datasets  : scifact, nfcorpus
Orders    : natural, bond, pca
k (top-k) : 10
Subsample : 50 queries (deterministic seed=42; see N_QUERIES below)

Outputs per dataset:
  results/json/stage3_mechanism_e01_bound_slack_<dataset>.json
  results/figures/stage3_mechanism/e01_bound_slack_<dataset>.png

Usage:
    .venv/bin/python -m experiments.stage3_mechanism.e01_bound_slack
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
from bondmaxsim.oracle.bound_trajectory import doc_ub_trajectory, make_prefix_grid
from bondmaxsim.ordering.orders import pca_order, bond_order, natural_order

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = ["scifact", "nfcorpus"]
ORDER_NAMES = ["natural", "bond", "pca"]
K_TOP = 10
N_QUERIES = 50   # subsample for tractability
QUERY_SEED = 42
N_GRID_POINTS = 33
D = 128

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_order(name: str, query: np.ndarray, mu: np.ndarray, R: np.ndarray):
    """Return (query_eff, flat_tokens_modifier, order) for the given order name."""
    if name == "natural":
        return query, None, natural_order(query)
    elif name == "bond":
        return query, None, bond_order(query, mu)
    elif name == "pca":
        q_rot, order = pca_order(query, R)
        return q_rot, "rotated", order
    else:
        raise ValueError(f"Unknown order: {name!r}")


def _make_pca_rotation(D: int) -> np.ndarray:
    """Deterministic orthogonal rotation (matches Runner)."""
    g = np.random.default_rng(123).standard_normal((D, D))
    Q_r, _ = np.linalg.qr(g)
    return Q_r.astype(np.float32)


def _subsample_queries(queries: list[np.ndarray], n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(queries), size=min(n, len(queries)), replace=False)
    return [queries[i] for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Per-query trajectory computation
# ---------------------------------------------------------------------------

def compute_slack_trajectories(
    flat_tokens: np.ndarray,
    flat_tokens_rot: np.ndarray,
    doc_starts: np.ndarray,
    queries: list[np.ndarray],
    order_name: str,
    prefix_grid: np.ndarray,
    mu: np.ndarray,
    R: np.ndarray,
) -> dict:
    """
    Returns dict with keys: slack_mean [S], slack_p10 [S], slack_p50 [S],
    slack_p90 [S], slack_at_D4 float, slack_at_D2 float.
    """
    D = flat_tokens.shape[1]
    all_slack: list[np.ndarray] = []  # one [num_docs] per (query, step) → collect per step

    step_slack_lists: list[list[float]] = [[] for _ in range(len(prefix_grid))]

    for query in queries:
        # Resolve effective query and corpus for this order.
        if order_name == "pca":
            q_eff = (query @ R).astype(np.float32)
            ft_eff = flat_tokens_rot
            order = natural_order(query)
        elif order_name == "bond":
            q_eff = query
            ft_eff = flat_tokens
            order = bond_order(query, mu)
        else:  # natural
            q_eff = query
            ft_eff = flat_tokens
            order = natural_order(query)

        ub_traj, exact_scores = doc_ub_trajectory(
            q_eff, ft_eff, doc_starts, order, prefix_grid
        )

        # Avoid division by zero for zero-score docs (shouldn't happen with real data).
        safe_scores = np.where(exact_scores > 1e-8, exact_scores, 1e-8)

        # slack[d, s] = UB_d(k_s) / score_d
        slack = ub_traj / safe_scores[:, np.newaxis]  # [num_docs, S]

        for s in range(len(prefix_grid)):
            step_slack_lists[s].extend(slack[:, s].tolist())

    # Aggregate across all (query, doc) pairs.
    slack_mean = np.array([np.mean(v) for v in step_slack_lists], dtype=np.float64)
    slack_p10  = np.array([np.percentile(v, 10) for v in step_slack_lists])
    slack_p50  = np.array([np.percentile(v, 50) for v in step_slack_lists])
    slack_p90  = np.array([np.percentile(v, 90) for v in step_slack_lists])

    # Slack at D/4 and D/2.
    D4_idx = int(np.searchsorted(prefix_grid, D // 4, side='right')) - 1
    D4_idx = max(0, D4_idx)
    D2_idx = int(np.searchsorted(prefix_grid, D // 2, side='right')) - 1
    D2_idx = max(0, D2_idx)

    return {
        "slack_mean": slack_mean.tolist(),
        "slack_p10": slack_p10.tolist(),
        "slack_p50": slack_p50.tolist(),
        "slack_p90": slack_p90.tolist(),
        "slack_at_D4": float(slack_mean[D4_idx]),
        "slack_at_D2": float(slack_mean[D2_idx]),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def save_figure(
    prefix_grid: np.ndarray,
    results_by_order: dict[str, dict],
    dataset: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"natural": "steelblue", "bond": "darkorange", "pca": "forestgreen"}

    for order_name, res in results_by_order.items():
        c = colors.get(order_name, "gray")
        ax.plot(prefix_grid, res["slack_mean"], label=f"{order_name} (mean)", color=c)
        ax.fill_between(
            prefix_grid,
            res["slack_p10"],
            res["slack_p90"],
            alpha=0.15,
            color=c,
            label=f"{order_name} p10–p90",
        )

    ax.axvline(D // 4, color="gray", linestyle=":", linewidth=0.8, label="D/4")
    ax.axvline(D // 2, color="gray", linestyle="--", linewidth=0.8, label="D/2")
    ax.axhline(1.0, color="black", linestyle="-", linewidth=0.6)
    ax.set_xlabel("Prefix k (dims scanned)")
    ax.set_ylabel("UB_d(k) / score(q, d)")
    ax.set_title(f"Bound slack vs prefix — {dataset}")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, D)
    ax.set_ylim(bottom=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> None:
    print(f"\n=== e01 bound slack: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    mu = flat_tokens.mean(axis=0).astype(np.float32)
    R  = _make_pca_rotation(D)
    flat_tokens_rot = (flat_tokens @ R).astype(np.float32)

    prefix_grid = make_prefix_grid(D, N_GRID_POINTS)

    results_by_order: dict[str, dict] = {}
    for order_name in ORDER_NAMES:
        print(f"  order={order_name} ...", end=" ", flush=True)
        t1 = time.perf_counter()
        res = compute_slack_trajectories(
            flat_tokens, flat_tokens_rot, doc_starts, queries,
            order_name, prefix_grid, mu, R,
        )
        elapsed = time.perf_counter() - t1
        results_by_order[order_name] = res
        print(f"{elapsed:.1f}s  slack@D/4={res['slack_at_D4']:.3f}  "
              f"slack@D/2={res['slack_at_D2']:.3f}")

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_JSON / f"stage3_mechanism_e01_bound_slack_{dataset}.json"
    payload = {
        "experiment": "e01_bound_slack",
        "dataset": dataset,
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k_top": K_TOP,
        "D": D,
        "prefix_grid": prefix_grid.tolist(),
        "orders": ORDER_NAMES,
        "results": results_by_order,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Write figure.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e01_bound_slack_{dataset}.png"
    save_figure(prefix_grid, results_by_order, dataset, fig_path)

    elapsed_total = time.perf_counter() - t0
    print(f"  Done in {elapsed_total:.1f}s")


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    for ds in datasets:
        run_dataset(ds)
