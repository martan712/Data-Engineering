"""Plot the controlled SciFact-to-NFCorpus transfer evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = PROJECT_ROOT / "results" / "final"
OUTPUT = PROJECT_ROOT / "docs" / "figures" / "fig8_controlled_transfer.png"
DATASETS = ("SciFact", "NFCorpus")
COLORS = {"SciFact": "#187a8c", "NFCorpus": "#d17a22"}


def load(name: str, schema: str) -> dict:
    result = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    if result.get("schema_version") != schema:
        raise SystemExit(f"Unexpected schema in {name}: {result.get('schema_version')}")
    return result


def latency_ratio(result: dict, arm_name: str) -> float:
    exact = result["exact_reference"]["timing"]["end_to_end_summary"]["median"]
    bond = result["bond_arms"][arm_name]["timing"]["end_to_end_summary"]["median"]
    return bond / exact


def main() -> None:
    ivf = {
        "SciFact": load(
            "scifact_ivf_test_40q_final.json",
            "controlled_ivf_plaid_v2",
        ),
        "NFCorpus": load(
            "nfcorpus_ivf_test_40q_final.json",
            "controlled_ivf_plaid_v2",
        ),
    }
    bond = {
        "SciFact": load(
            "scifact_bond_raw_test_40q_final.json",
            "controlled_bond_pilot_v1",
        ),
        "NFCorpus": load(
            "nfcorpus_bond_raw_test_40q_final.json",
            "controlled_bond_pilot_v1",
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))

    budgets = ("c50", "c200", "cfull")
    budget_labels = ("C=50", "C=200", "pool")
    x_values = np.arange(len(budgets))
    for dataset in DATASETS:
        recalls = [
            ivf[dataset]["approximate_arms"][f"pdx_ivf_{budget}"]["exact_recovery"]
            ["mean_recall@10"]
            for budget in budgets
        ]
        axes[0].plot(
            x_values,
            recalls,
            marker="o",
            linewidth=2,
            color=COLORS[dataset],
            label=dataset,
        )
    axes[0].set_xticks(x_values, budget_labels)
    axes[0].set_ylim(0.8, 1.01)
    axes[0].set_ylabel("Recall@10 vs exact MaxSim")
    axes[0].set_title("IVF candidate recovery")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")

    x_datasets = np.arange(len(DATASETS))
    width = 0.34
    component_ratios = [
        bond[name]["bond_arms"]["bond_seed500"]["work"]["component_product_ratio"]
        for name in DATASETS
    ]
    prune_fractions = [
        bond[name]["bond_arms"]["bond_seed500"]["work"]["pruned_document_fraction"]
        for name in DATASETS
    ]
    component_bars = axes[1].bar(
        x_datasets - width / 2,
        component_ratios,
        width,
        color="#2878b5",
        label="component products",
    )
    prune_bars = axes[1].bar(
        x_datasets + width / 2,
        prune_fractions,
        width,
        color="#d55e00",
        label="document pairs pruned",
    )
    axes[1].set_xticks(x_datasets, DATASETS)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_ylabel("Fraction of exhaustive total")
    axes[1].set_title("Raw-order BOND work")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    for bars in (component_bars, prune_bars):
        for bar in bars:
            axes[1].annotate(
                f"{100 * bar.get_height():.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ratios = [latency_ratio(bond[name], "bond_seed500") for name in DATASETS]
    ratio_bars = axes[2].bar(
        x_datasets,
        ratios,
        width=0.58,
        color=[COLORS[name] for name in DATASETS],
    )
    axes[2].axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    axes[2].set_xticks(x_datasets, DATASETS)
    axes[2].set_ylim(0.0, max(ratios) * 1.18)
    axes[2].set_ylabel("BOND / same-run exact median")
    axes[2].set_title("Raw-order BOND latency")
    axes[2].grid(axis="y", alpha=0.25)
    for bar, ratio in zip(ratio_bars, ratios):
        axes[2].annotate(
            f"{ratio:.1f}x",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.suptitle(
        "Controlled transfer check: candidate recovery varies, BOND conclusion persists",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "40 held-out queries per dataset; L=100, nprobe=8; BOND prefix seeds=500; "
        "same WSL stack, four pinned physical cores, and online boundary.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
