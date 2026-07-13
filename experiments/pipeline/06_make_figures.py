"""Generate the figures for the project writeup from saved result JSONs.

Reads the result artifacts produced by experiments 28/31/32/33 and writes three
PNG figures into docs/figures/:

  fig1_ivf_scaling.png      - Historical component timing and candidate-pool
                              fraction as corpus size grows.
  fig2_baselines.png        - End-to-end latency and qrels recall@10 across
                              exact / FAISS-IVF / PLAID / PDX-IVF.
  fig3_bond_pruning.png     - MaxSim multi-vector BOND pruning curve: live
                              documents vs dimensions scanned, raw vs PCA.
  fig4_selector_gap.png     - Selector-gap sweep (exp 35): agreement@10 vs
                              total latency as the re-rank budget C grows,
                              SciFact-full and NFCorpus. Skipped if the sweep
                              JSONs are missing.

Runs in the Windows venv (matplotlib).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _paths import PROJECT_ROOT
RESULTS = PROJECT_ROOT / "results" / "legacy"
FIGURES = PROJECT_ROOT / "docs" / "figures"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def ivf_run(subset: dict, nprobe: int, top_l: int) -> dict:
    for nprobe_run in subset["nprobe_runs"]:
        if nprobe_run["nprobe"] != nprobe:
            continue
        for l_run in nprobe_run["l_runs"]:
            if l_run["top_l"] == top_l:
                return l_run
    raise KeyError(f"nprobe={nprobe} L={top_l} not found")


def figure_scaling() -> None:
    scaling = load("scifact_pdx_ivf_scaling_curve.json")
    full = load("scifact_full_ivf_benchmark.json")

    sizes, exact_s, ivf_s, pool_fraction = [], [], [], []
    for subset in scaling["subsets"]:
        run = ivf_run(subset, nprobe=8, top_l=100)
        sizes.append(subset["documents"])
        exact_s.append(subset["exact"]["seconds"])
        ivf_s.append(run["candidate_generation_seconds"])
        pool_fraction.append(run["pool_coverage"]["mean_candidate_pool_size"] / subset["documents"])
    # Append the full-corpus point (nprobe=8, L=100).
    for run in full["runs"]:
        if run["nprobe"] == 8 and run["top_l"] == 100:
            sizes.append(full["dataset"]["documents"])
            exact_s.append(full["exact_reference"]["seconds"])
            ivf_s.append(run["candidate_generation_seconds"])
            pool_fraction.append(
                run["pool_coverage"]["mean_candidate_pool_size"] / full["dataset"]["documents"]
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(sizes, exact_s, "o-", color="#c0392b", label="Exact MaxSim (NumPy)")
    ax1.plot(sizes, ivf_s, "s-", color="#27ae60", label="IVF-on-PDX candidate gen")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("corpus size (documents)")
    ax1.set_ylabel("time for 50 queries (s)")
    ax1.set_title("Historical component times (unmatched work)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.plot(sizes, pool_fraction, "D-", color="#2c3e50")
    for x, y in zip(sizes, pool_fraction):
        ax2.annotate(f"{100 * y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    ax2.set_xscale("log")
    ax2.set_xlabel("corpus size (documents)")
    ax2.set_ylabel("mean unique candidate pool / corpus")
    ax2.set_title("Candidate-pool fraction decreases with scale")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("IVF-on-PDX candidate generation (nprobe=8, L=100), SciFact", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_ivf_scaling.png", dpi=150)
    plt.close(fig)


def figure_baselines() -> None:
    data = load("scifact_full_baselines.json")
    exact = data["exact_reference"]
    faiss = next(r for r in data["faiss_ivf"]["runs"] if r["nprobe"] == 8)
    plaid = data["plaid"]
    pdx = data["pdx_ivf_best_from_saved_result"]

    labels = ["Exact\nMaxSim", "FAISS-IVF\n+rerank", "PLAID\n(CPU)", "PDX-IVF\n+rerank\n[WSL]"]
    times = [exact["seconds"], faiss["total_seconds"], plaid["retrieve_seconds"], pdx["total_seconds"]]
    recalls = [
        exact["qrels_metrics"]["recall@10"],
        faiss["qrels_metrics"]["recall@10"],
        plaid["qrels_metrics"]["recall@10"],
        pdx["qrels_recall@10"],
    ]
    colors = ["#c0392b", "#27ae60", "#e67e22", "#2980b9"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    bars1 = ax1.bar(labels, times, color=colors)
    ax1.set_yscale("log")
    ax1.set_ylabel("time for 50 queries (s, log)")
    ax1.set_title("Query latency")
    for bar, value in zip(bars1, times):
        ax1.annotate(f"{value:.2f}s", (bar.get_x() + bar.get_width() / 2, value),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)

    bars2 = ax2.bar(labels, recalls, color=colors)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("qrels recall@10")
    ax2.set_title("Retrieval quality")
    for bar, value in zip(bars2, recalls):
        ax2.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                     textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)

    fig.suptitle(
        "Baselines on full SciFact (5183 docs, 50 queries)\n"
        "exact/FAISS/PLAID same machine; PDX [WSL] absolute time not directly comparable",
        fontweight="bold",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_baselines.png", dpi=150)
    plt.close(fig)


def figure_pruning() -> None:
    raw = load("maxsim_bond_instrumentation_scifact_full.json")["summaries_by_threshold_mode"]
    pca = load("maxsim_bond_instrumentation_scifact_full_pca.json")["summaries_by_threshold_mode"]

    def curve(summary: dict) -> tuple[list[int], list[float]]:
        items = sorted(((int(d), v) for d, v in summary["mean_live_fraction_by_dim"].items()))
        return [d for d, _ in items], [v for _, v in items]

    series = [
        ("raw, self-bound", raw["bound"], "#c0392b", "--"),
        ("raw, oracle threshold", raw["oracle"], "#e74c3c", "-"),
        ("PCA, self-bound", pca["bound"], "#2980b9", "--"),
        ("PCA, oracle threshold", pca["oracle"], "#27ae60", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, summary, color, style in series:
        dims, live = curve(summary)
        inverse = summary.get(
            "inverse_dimension_scan_ratio",
            1.0 / summary["mean_work_ratio"],
        )
        ax.plot(dims, live, style, color=color, marker="o", label=f"{label} ({inverse:.2f}x inverse scan ratio)")
    ax.set_xlabel("dimensions scanned (of 128)")
    ax.set_ylabel("fraction of documents still live")
    ax.set_title(
        "MaxSim multi-vector BOND pruning (full SciFact, k=10)\n"
        "legend shows inverse dimension-scan ratio, not kernel speedup"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="lower-left = more pruning")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_bond_pruning.png", dpi=150)
    plt.close(fig)


def figure_selector_gap() -> None:
    datasets = [
        ("SciFact-full", "scifact_full_selector_gap_sweep.json", "#2980b9"),
        ("NFCorpus", "nfcorpus_selector_gap_sweep.json", "#27ae60"),
    ]
    if not all((RESULTS / name).exists() for _, name, _ in datasets):
        print("  fig4 skipped (selector-gap sweep JSONs missing)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for label, name, color in datasets:
        data = load(name)
        rows = [r for r in data["sweep"] if r["policy"] == "approx_score"]
        rows.sort(key=lambda r: r["mean_selected_candidates"])
        times = [r["total_seconds"] for r in rows]
        agreement = [r["agreement@10"] for r in rows]
        selected_counts = [r["mean_selected_candidates"] for r in rows]

        ax1.plot(times, agreement, "o-", color=color, label=label)
        for row, x, y in zip(rows, times, agreement):
            ax1.annotate(f"C={row['effective_c_label']}", (x, y),
                         textcoords="offset points", xytext=(4, -10), fontsize=8, color=color)
        ax2.plot(selected_counts, agreement, "o-", color=color, label=label)
        for row, x, y in zip(rows, selected_counts, agreement):
            ax2.annotate(f"C={row['effective_c_label']}", (x, y),
                         textcoords="offset points", xytext=(4, -10), fontsize=8, color=color)

    for ax in (ax1, ax2):
        ax.axhline(1.0, color="#c0392b", linestyle=":", alpha=0.7)
        ax.set_ylabel("exact agreement@10")
        ax.set_ylim(0.55, 1.03)
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax1.set_xlabel("total time for 50 queries (s)")
    ax1.set_title("Quality vs latency as re-rank budget C grows")
    ax2.set_xlabel("mean documents reranked per query")
    ax2.set_title("Quality vs rerank work")

    fig.suptitle(
        "Selector-gap sweep (FAISS-IVF nprobe=8, L=100; policy=approx_score)\n"
        "the C=50 plateau is a budget artifact: C=pool reaches pool-coverage-limited agreement",
        fontweight="bold",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_selector_gap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_scaling()
    figure_baselines()
    figure_pruning()
    figure_selector_gap()
    print(f"Wrote figures to {FIGURES}")
    for png in sorted(FIGURES.glob("*.png")):
        print(f"  {png.name}")


if __name__ == "__main__":
    main()
