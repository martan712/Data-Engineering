"""Render e02 cost components from a validated saved envelope."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bondmaxsim.results import load_result


def render_seeded_tau(input_path: Path, output_path: Path) -> Path:
    envelope = load_result(input_path)
    if envelope.payload_kind != "timing_comparison" or "stage4-e02" not in envelope.experiment_id:
        raise ValueError("e02 renderer requires a migrated e02 timing envelope")
    session = envelope.payload["sessions"][0]
    if not session.get("complete"):
        raise ValueError("e02 renderer refuses incomplete sessions")
    sample_size = int(envelope.protocol["workload"]["sample_size"])
    arms = [arm for arm in session["arm_ids"] if arm != "dense-fused"]
    values = [session["summaries"][arm]["median_ns"] / sample_size / 1e6 for arm in arms]
    figure, axis = plt.subplots(figsize=(max(8, len(arms) * 1.05), 4.4))
    axis.bar(range(len(arms)), values, color="#2e8b57")
    axis.set_xticks(range(len(arms)), arms, rotation=55, ha="right", fontsize=7)
    axis.set_ylabel("directly observed median ms / query")
    axis.set_title("seed-only, kernel-only, and end-to-end observations (not summed)")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
