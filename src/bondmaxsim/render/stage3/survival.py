"""Render E02 survival curves from a validated saved envelope."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bondmaxsim.results import load_result


def render_survival(input_path: Path, output_path: Path) -> Path:
    envelope = load_result(input_path)
    if envelope.experiment_id != "stage3-e02-survival":
        raise ValueError("E02 renderer requires a stage3-e02-survival artifact")
    rows = [row for row in envelope.payload["rows"] if row.get("row_type") != "numpy-crosscheck"]
    colors = {"self_bound": "#2a78d6", "oracle": "#1baf7a", "seed": "#eb6834"}
    styles = {"natural": "-", "bond": "--", "pca": ":"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for axis, key, title in zip(axes, ("doc_survival", "token_survival"), ("Document survival", "Token survival")):
        for row in rows:
            policy, order = row["threshold_policy"], row["dimension_order"]
            axis.plot(row["fetch_boundary_dims"], row[key], color=colors.get(policy), linestyle=styles.get(order, "-"), label=f"{policy} / {order}")
        axis.set(xlabel="dimensions scanned", title=title, ylim=(0, 1.02))
        axis.grid(color="#e9e8e2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("fraction live")
    axes[1].legend(fontsize=7)
    figure.suptitle(f"E02 survival curves — {envelope.dataset_id}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
