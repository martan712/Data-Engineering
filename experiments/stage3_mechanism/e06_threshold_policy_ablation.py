"""Stage 3 e06: threshold-policy ablation — self_bound / oracle / seed.

Single responsibility: compare the three threshold policies on cells_scanned_pct,
pruned_docs_pct, tokens_pruned_pct, and recall_vs_exact@10, with fixed dimension
order (natural) and shrink=1.0 (exact-safe arm).  Policy choice is a mechanism
question, so the comparison runs on the wide-block ACCOUNTING kernel; the
winning policy (lowest cells%) then gets ONE fused doc-level wall-clock
confirmation arm (1T + all cores) — R6: wall-clock confirmation needs one
arm, not nine.

  self_bound — threshold derived from the document's own running BOND upper-bound;
               available without any prior information; tends to give a weak
               (low) threshold at the start of the scan.
  oracle     — uses the exact k-th score as tau (ideal ceiling; not realistic
               in production but quantifies maximum pruning potential).
  seed       — tau_seed comes from a prior-pass or external ANN estimate; first-
               class for wide-block scans where tau_k = -inf mid-pass
               (Stage 1 §4.4 / §8 item 6).

The seed policy in the testbed uses a lightweight "self-seed" derived from the
first fetch boundary scores over a small warmup pass, simulating a prior-pass
seed.  This is a conservative lower bound on realistic seed quality.

Datasets  : scifact, nfcorpus, arguana, scidocs
Policy    : self_bound, oracle, seed
Order     : natural
Shrink    : 1.0
Queries   : 50 (same deterministic subsample as e01/e02)
k (top-k) : 10

Outputs per dataset:
  results/json/stage3_mechanism_e06_threshold_policy_ablation_<dataset>.json
  results/figures/stage3_mechanism/e06_threshold_policy_ablation_<dataset>.png

Usage:
    uv run python -m experiments.stage3_mechanism.e06_threshold_policy_ablation [dataset ...]
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

DATASETS  = ["scifact", "nfcorpus", "arguana", "scidocs"]
POLICIES  = ["self_bound", "oracle", "seed"]
ORDER     = "natural"
K_TOP     = 10
N_QUERIES = 50
QUERY_SEED = 42
N_REPEATS = 5   # wall-clock repeats (best-of), fused confirmation arm

RESULTS_JSON = REPO_ROOT / "results" / "json"
RESULTS_FIG  = REPO_ROOT / "results" / "figures" / "stage3_mechanism"

POLICY_COLORS = {"self_bound": "#2a78d6", "oracle": "#1baf7a", "seed": "#eb6834"}


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
    print(f"\n=== e06 threshold-policy ablation: {dataset} ===")
    t0 = time.perf_counter()

    flat_tokens, doc_starts, queries = load_dataset(dataset)
    queries = _subsample_queries(queries, N_QUERIES, QUERY_SEED)
    print(f"  corpus: {len(doc_starts)} docs, {flat_tokens.shape[0]} tokens, "
          f"{len(queries)} queries")

    runner = Runner(flat_tokens, doc_starts, queries)
    arms: list[dict] = []

    for policy in POLICIES:
        cfg = RunConfig(
            dataset=dataset,
            method="wide_block_maxsim_bond",
            dimension_order=ORDER,
            threshold_policy=policy,
            k=K_TOP,
            shrink=1.0,
        )
        t1 = time.perf_counter()
        rec = runner.accounting_mode(cfg)
        elapsed = time.perf_counter() - t1

        arm = {
            "threshold_policy": policy,
            "dimension_order": ORDER,
            "recall_vs_exact_at_10": rec.recall_vs_exact_at_10,
            "strict_top_k_set_equal": rec.strict_top_k_set_equal,
            "boundary_tie_equivalent": rec.boundary_tie_equivalent,
            "agreement_failure_codes": rec.agreement_failure_codes,
            "cells_scanned_pct": rec.cells_scanned_pct,
            "pruned_docs_pct": rec.pruned_docs_pct,
            "tokens_pruned_pct": rec.tokens_pruned_pct,
        }
        arms.append(arm)
        print(f"  policy={policy:<10}  "
              f"recall={arm['recall_vs_exact_at_10']:.3f}  "
              f"cells={arm['cells_scanned_pct']:.2f}%  "
              f"prune_docs={arm['pruned_docs_pct']:.2f}%  "
              f"tok_prune={arm['tokens_pruned_pct']:.2f}%  "
              f"{elapsed:.1f}s")

        if not rec.boundary_tie_equivalent:
            raise RuntimeError(
                f"Verified exact gate failed at shrink=1: "
                f"policy={policy!r} failures={rec.agreement_failure_codes}"
            )

    # Fused wall-clock confirmation arm for the winning policy only (R6).
    winner = min(arms, key=lambda a: a["cells_scanned_pct"])["threshold_policy"]
    cfg_fused = RunConfig(
        dataset=dataset,
        method="fused_panel_maxsim_bond",
        dimension_order=ORDER,
        threshold_policy=winner,
        k=K_TOP,
        shrink=1.0,
    )
    rec_1t = runner.throughput_mode(cfg_fused, n_repeats=N_REPEATS, n_threads=1)
    rec_mt = runner.throughput_mode(cfg_fused, n_repeats=N_REPEATS, n_threads=0)
    cfg_brute = RunConfig(dataset=dataset, method="fused_panel_maxsim_bond",
                          dimension_order="natural", threshold_policy="none",
                          k=K_TOP, shrink=1.0)
    rec_dense = runner.brute_force_mode(cfg_brute, kind="fused",
                                        n_repeats=N_REPEATS, n_threads=0)
    fused_arm = {
        "threshold_policy": winner,
        "dimension_order": ORDER,
        "recall_vs_exact_at_10": min(rec_1t.recall_vs_exact_at_10,
                                     rec_mt.recall_vs_exact_at_10),
        "strict_top_k_set_equal": bool(
            rec_1t.strict_top_k_set_equal and rec_mt.strict_top_k_set_equal
        ),
        "boundary_tie_equivalent": bool(
            rec_1t.boundary_tie_equivalent and rec_mt.boundary_tie_equivalent
        ),
        "agreement_failure_codes": sorted(set(
            (rec_1t.agreement_failure_codes or []) +
            (rec_mt.agreement_failure_codes or [])
        )),
        "ms_per_query_1t": rec_1t.ms_per_query,
        "ms_per_query_mt": rec_mt.ms_per_query,
        "pruned_docs_pct_fused": rec_mt.pruned_docs_pct,
        "dense_fused_ms_per_query_mt": rec_dense.ms_per_query,
    }
    print(f"  fused[{winner}]  recall={fused_arm['recall_vs_exact_at_10']:.3f}  "
          f"ms/q 1T={fused_arm['ms_per_query_1t']:.2f} "
          f"MT={fused_arm['ms_per_query_mt']:.2f}  "
          f"prune={fused_arm['pruned_docs_pct_fused']:.2f}%  "
          f"(dense MT={fused_arm['dense_fused_ms_per_query_mt']:.2f})")
    if not fused_arm["boundary_tie_equivalent"]:
        raise RuntimeError(
            f"Verified exact gate failed on the fused kernel: "
            f"policy={winner!r} failures={fused_arm['agreement_failure_codes']}"
        )

    # Write JSON.
    RESULTS_JSON.mkdir(parents=True, exist_ok=True)
    json_path = (
        RESULTS_JSON / f"stage3_mechanism_e06_threshold_policy_ablation_{dataset}.json"
    )
    payload = {
        "experiment": "e06_threshold_policy_ablation",
        "dataset": dataset,
        "method": "wide_block_maxsim_bond",
        "n_docs": len(doc_starts),
        "total_tokens": flat_tokens.shape[0],
        "n_queries": len(queries),
        "query_seed": QUERY_SEED,
        "k": K_TOP,
        "shrink": 1.0,
        "order": ORDER,
        "policies": POLICIES,
        "n_repeats_throughput": N_REPEATS,
        "arms": arms,
        "fused_confirmation_arm": fused_arm,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {json_path}")

    # Figure: stacked metrics per policy.
    RESULTS_FIG.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_FIG / f"e06_threshold_policy_ablation_{dataset}.png"
    _save_figure(arms, dataset, fig_path)

    print(f"  Done in {time.perf_counter() - t0:.1f}s")


def _save_figure(arms: list[dict], dataset: str, out_path: Path) -> None:
    policies    = [a["threshold_policy"] for a in arms]
    cells       = [a["cells_scanned_pct"] for a in arms]
    prune_docs  = [a["pruned_docs_pct"] for a in arms]
    prune_toks  = [a["tokens_pruned_pct"] for a in arms]
    colors      = [POLICY_COLORS.get(p, "gray") for p in policies]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(policies))

    for ax, values, title, ylim in [
        (axes[0], cells,      "cells scanned %",   (0, 105)),
        (axes[1], prune_docs, "pruned docs %",     (0, 105)),
        (axes[2], prune_toks, "tokens pruned %",   (0, 105)),
    ]:
        ax.bar(x, values, color=colors, width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(policies, rotation=10)
        ax.set_ylabel(title)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#e9e8e2", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for xi, v in zip(x, values):
            ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=8)

    fig.suptitle(
        f"e06 threshold-policy ablation — {dataset}  "
        f"(wide kernel, natural order, shrink=1, {N_QUERIES} queries)",
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
