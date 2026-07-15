"""Render e01 comparisons from validated saved envelopes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bondmaxsim.results import load_result


def render_fixed_candidates(input_path: Path, output_path: Path) -> Path:
    envelope = load_result(input_path)
    if envelope.payload_kind != "timing_comparison" or "stage4-e01" not in envelope.experiment_id:
        raise ValueError("e01 renderer requires a migrated e01 timing envelope")
    session = envelope.payload["sessions"][0]
    if not session.get("complete"):
        raise ValueError("e01 renderer refuses incomplete sessions")
    sample_size = int(envelope.protocol["workload"]["sample_size"])
    arms = [arm for arm in session["arm_ids"] if arm != "dense-fused"]
    latency = [session["summaries"][arm]["median_ns"] / sample_size / 1e6 for arm in arms]
    dense = session["summaries"]["dense-fused"]["median_ns"] / sample_size / 1e6
    figure, axis = plt.subplots(figsize=(max(6, len(arms) * 1.2), 4.2))
    axis.bar(range(len(arms)), latency, color="#2a78d6")
    axis.axhline(dense, color="#5c4a9e", linestyle="--", label="dense fused")
    axis.set_xticks(range(len(arms)), arms, rotation=45, ha="right")
    axis.set_ylabel("median ms / query")
    axis.set_title(envelope.method_configuration["output_semantics"].replace("_", " "))
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
