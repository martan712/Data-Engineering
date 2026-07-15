"""Render E01 from a validated saved accounting envelope."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bondmaxsim.results import load_result


def render_bound_slack(input_path: Path, output_path: Path) -> Path:
    envelope = load_result(input_path)
    if envelope.experiment_id != "stage3-e01-bound-slack":
        raise ValueError("E01 renderer requires a stage3-e01-bound-slack artifact")
    colors = {"natural": "steelblue", "bond": "darkorange", "pca": "forestgreen"}
    figure, axis = plt.subplots(figsize=(7, 4))
    for row in envelope.payload["rows"]:
        order = row["dimension_order"]
        axis.plot(row["prefix_grid"], row["slack_mean"], label=f"{order} (mean)", color=colors.get(order))
        axis.fill_between(row["prefix_grid"], row["slack_p10"], row["slack_p90"], alpha=0.15, color=colors.get(order))
    dimension = max(max(row["prefix_grid"]) for row in envelope.payload["rows"])
    axis.axvline(dimension / 4, color="gray", linestyle=":", linewidth=0.8, label="D/4")
    axis.axvline(dimension / 2, color="gray", linestyle="--", linewidth=0.8, label="D/2")
    axis.axhline(1.0, color="black", linewidth=0.6)
    axis.set(xlabel="Prefix dimensions scanned", ylabel="UB / exact score", title=f"Bound slack — {envelope.dataset_id}")
    axis.legend(fontsize=7, ncol=2)
    axis.set_ylim(bottom=0.9)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
