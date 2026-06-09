"""Generate the figures for the project writeup from saved result JSONs.

Reads the result artifacts produced by experiments 28/31/32/33 and writes three
PNG figures into docs/figures/:

  fig1_ivf_scaling.png      - IVF candidate-gen vs exact MaxSim, and speedup
                              growing with corpus size.
  fig2_baselines.png        - End-to-end latency and qrels recall@10 across
                              exact / FAISS-IVF / PLAID / PDX-IVF.
  fig3_bond_pruning.png     - MaxSim multi-vector BOND pruning curve: live
                              documents vs dimensions scanned, raw vs PCA.

Runs in the Windows venv (matplotlib).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "artifacts" / "results"
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

    sizes, exact_s, ivf_s, speedup = [], [], [], []
    for subset in scaling["subsets"]:
        run = ivf_run(subset, nprobe=8, top_l=100)
        sizes.append(subset["documents"])
        exact_s.append(subset["exact"]["seconds"])
        ivf_s.append(run["candidate_generation_seconds"])
        speedup.append(run["speedup_vs_exact"])
    # Append the full-corpus point (nprobe=8, L=100).
    for run in full["runs"]:
        if run["nprobe"] == 8 and run["top_l"] == 100:
            sizes.append(full["dataset"]["documents"])
            exact_s.append(full["exact_reference"]["seconds"])
            ivf_s.append(run["candidate_generation_seconds"])
            speedup.append(run["speedup_vs_exact"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(sizes, exact_s, "o-", color="#c0392b", label="Exact MaxSim (NumPy)")
    ax1.plot(sizes, ivf_s, "s-", color="#27ae60", label="IVF-on-PDX candidate gen")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("corpus size (documents)")
    ax1.set_ylabel("time for 50 queries (s)")
    ax1.set_title("Candidate generation vs exact MaxSim")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.plot(sizes, speedup, "D-", color="#2c3e50")
    for x, y in zip(sizes, speedup):
        ax2.annotate(f"{y:.1f}x", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    ax2.set_xscale("log")
    ax2.set_xlabel("corpus size (documents)")
    ax2.set_ylabel("speedup vs exact (x)")
    ax2.set_title("IVF speedup grows with corpus size")
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
        ceiling = summary["ceiling_speedup_vs_naive_full_scan"]
        ax.plot(dims, live, style, color=color, marker="o", label=f"{label} ({ceiling:.2f}x)")
    ax.set_xlabel("dimensions scanned (of 128)")
    ax.set_ylabel("fraction of documents still live")
    ax.set_title(
        "MaxSim multi-vector BOND pruning (full SciFact, k=10)\n"
        "legend shows ceiling speedup = 1 / work-ratio"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="lower-left = more pruning")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_bond_pruning.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_scaling()
    figure_baselines()
    figure_pruning()
    print(f"Wrote figures to {FIGURES}")
    for png in sorted(FIGURES.glob("*.png")):
        print(f"  {png.name}")


if __name__ == "__main__":
    main()
