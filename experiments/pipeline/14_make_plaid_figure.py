"""Plot the controlled PLAID comparison and post-hoc full-score sensitivity."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = PROJECT_ROOT / "results" / "final"
OUTPUT = PROJECT_ROOT / "docs" / "figures" / "fig9_controlled_plaid.png"
IVF_PATTERN = re.compile(r"^faiss_ivf_c(.+)$")
COLORS = {
    "ivf": "#187a8c",
    "plaid": "#6b4c9a",
    "exact": "#b23a48",
}


def load(name: str) -> dict:
    result = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    if result.get("schema_version") != "controlled_ivf_plaid_v2":
        raise SystemExit(f"Unexpected schema in {name}: {result.get('schema_version')}")
    return result


def sorted_ivf_arms(result: dict) -> list[tuple[str, dict]]:
    values = []
    for name, arm in result["approximate_arms"].items():
        match = IVF_PATTERN.match(name)
        if not match:
            continue
        budget = match.group(1)
        order = float("inf") if budget == "full" else int(budget)
        values.append((name, arm, order))
    return [(name, arm) for name, arm, _ in sorted(values, key=lambda item: item[2])]


def budget_label(name: str) -> str:
    budget = IVF_PATTERN.match(name).group(1)
    return "pool" if budget == "full" else budget


def panel(axis, dataset: str, primary: dict, sensitivity: dict) -> None:
    ivf = sorted_ivf_arms(primary)
    ivf_times = [arm["timing"]["end_to_end_summary"]["median"] for _, arm in ivf]
    ivf_recalls = [arm["exact_recovery"]["mean_recall@10"] for _, arm in ivf]
    axis.plot(
        ivf_times,
        ivf_recalls,
        color=COLORS["ivf"],
        marker="o",
        linewidth=2,
        markersize=5,
        label="FAISS-IVF + exact rerank",
    )
    for (name, _), x_value, y_value in zip(ivf, ivf_times, ivf_recalls):
        axis.annotate(
            f"C={budget_label(name)}",
            (x_value, y_value),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["ivf"],
        )

    plaid = list(primary["plaid_arms"].items())
    plaid.sort(key=lambda item: item[1]["config"]["n_full_scores"])
    plaid_times = [arm["timing"]["end_to_end_summary"]["median"] for _, arm in plaid]
    plaid_recalls = [arm["exact_recovery"]["mean_recall@10"] for _, arm in plaid]
    axis.plot(
        plaid_times,
        plaid_recalls,
        color=COLORS["plaid"],
        marker="^",
        linestyle="--",
        linewidth=2,
        markersize=7,
        label="PLAID selected on validation",
    )
    for (_, arm), x_value, y_value in zip(plaid, plaid_times, plaid_recalls):
        axis.annotate(
            f"full={arm['config']['n_full_scores']}",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["plaid"],
        )

    _, sensitivity_arm = next(iter(sensitivity["plaid_arms"].items()))
    sensitivity_time = sensitivity_arm["timing"]["end_to_end_summary"]["median"]
    sensitivity_recall = sensitivity_arm["exact_recovery"]["mean_recall@10"]
    sensitivity_budget = sensitivity_arm["config"]["n_full_scores"]
    axis.scatter(
        [sensitivity_time],
        [sensitivity_recall],
        marker="^",
        s=80,
        facecolors="none",
        edgecolors=COLORS["plaid"],
        linewidths=1.8,
        label="PLAID full-score sensitivity (post hoc)",
        zorder=4,
    )
    axis.annotate(
        f"full={sensitivity_budget}",
        (sensitivity_time, sensitivity_recall),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=COLORS["plaid"],
    )

    exact_time = primary["exact_reference"]["timing"]["end_to_end_summary"]["median"]
    axis.scatter(
        [exact_time],
        [1.0],
        marker="*",
        s=150,
        color=COLORS["exact"],
        label="Compiled exact MaxSim",
        zorder=5,
    )
    axis.set_xscale("log")
    axis.set_ylim(0.25, 1.035)
    axis.set_xlabel("Online latency for 40 queries (median seconds, log scale)")
    axis.set_title(dataset)
    axis.grid(axis="both", alpha=0.22)
    if dataset == "SciFact":
        axis.set_ylabel("Recall@10 versus exact MaxSim ranking")


def main() -> None:
    primary = {
        "SciFact": load("scifact_ivf_test_40q_final.json"),
        "NFCorpus": load("nfcorpus_ivf_test_40q_final.json"),
    }
    sensitivity = {
        "SciFact": load("scifact_plaid_fullscore_sensitivity_40q_final.json"),
        "NFCorpus": load("nfcorpus_plaid_fullscore_sensitivity_40q_final.json"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharey=True)
    for axis, dataset in zip(axes, ("SciFact", "NFCorpus")):
        panel(axis, dataset, primary[dataset], sensitivity[dataset])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Controlled PLAID comparison: selected budgets and full-score sensitivity",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.01,
        "40 held-out queries; four pinned physical cores; five interleaved runs. "
        "PLAID full is a configured budget, not an API-observed candidate count; hollow points are post-hoc sensitivity runs.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.80))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
