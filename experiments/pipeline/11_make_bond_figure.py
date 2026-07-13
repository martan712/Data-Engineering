"""Generate the controlled BOND mechanism/latency figure from tracked JSON."""

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
OUTPUT = PROJECT_ROOT / "docs" / "figures" / "fig7_controlled_bond.png"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def arm(data: dict, name: str, label: str) -> dict:
    value = data["bond_arms"][name]
    exact_median = data["exact_reference"]["timing"]["end_to_end_summary"]["median"]
    bond_median = value["timing"]["end_to_end_summary"]["median"]
    return {
        "label": label,
        "exact_median": exact_median,
        "bond_median": bond_median,
        "latency_ratio": bond_median / exact_median,
        "work_ratio": value["work"]["component_product_ratio"],
        "pruned_fraction": value["work"]["pruned_document_fraction"],
        "exact": value["exact_recovery"]["rankings_identical"],
    }


def main() -> None:
    raw = load("scifact_bond_raw_test_40q_final.json")
    pca = load("scifact_bond_pca_test_40q_final.json")
    rows = [
        arm(raw, "bond_seed500", "Raw\nprefix-500"),
        arm(raw, "bond_oracle_topk_seeds", "Raw\nfree oracle"),
        arm(pca, "bond_seed500", "PCA\nprefix-500"),
        arm(pca, "bond_oracle_topk_seeds", "PCA\nfree oracle"),
    ]
    if not all(row["exact"] for row in rows):
        raise SystemExit("Refusing to plot a BOND arm that differs from exact top-k")

    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    colors = ["#2878B5", "#70A0C8", "#D55E00", "#E69F73"]

    fig, (ax_work, ax_time) = plt.subplots(1, 2, figsize=(12, 4.8))

    width = 0.36
    product_bars = ax_work.bar(
        x - width / 2,
        [row["work_ratio"] for row in rows],
        width,
        color=colors,
        label="component products evaluated",
    )
    prune_bars = ax_work.bar(
        x + width / 2,
        [row["pruned_fraction"] for row in rows],
        width,
        color=colors,
        alpha=0.42,
        hatch="//",
        label="query-document pairs pruned",
    )
    ax_work.set_xticks(x, labels)
    ax_work.set_ylim(0.0, 1.08)
    ax_work.set_ylabel("fraction of exhaustive total")
    ax_work.set_title("Pruned documents are not saved arithmetic")
    ax_work.grid(axis="y", alpha=0.25)
    ax_work.legend(loc="lower left", fontsize=8)
    for bars in (product_bars, prune_bars):
        for bar in bars:
            ax_work.annotate(
                f"{100 * bar.get_height():.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    time_bars = ax_time.bar(
        x,
        [row["latency_ratio"] for row in rows],
        color=colors,
        width=0.62,
    )
    ax_time.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    ax_time.set_ylim(0.0, max(row["latency_ratio"] for row in rows) * 1.18)
    ax_time.set_xticks(x, labels)
    ax_time.set_ylabel("BOND median latency / same-run exact median")
    ax_time.set_title("All exact-safe BOND arms remain slower")
    ax_time.grid(axis="y", alpha=0.25)
    for bar, row in zip(time_bars, rows):
        ax_time.annotate(
            f"{row['latency_ratio']:.1f}x\n({row['bond_median']:.1f}s)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    fig.suptitle(
        "Exact-safe BOND-MaxSim on held-out SciFact: PCA improves pruning, not latency\n"
        "5,183 documents; 40 queries; 4 pinned physical cores; median of 5 interleaved runs; oracle seeds are free",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
