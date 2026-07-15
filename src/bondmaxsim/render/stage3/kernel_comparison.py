"""Render E03/R12c latency summaries from validated timing envelopes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bondmaxsim.results import load_result


def render_kernel_comparison(input_path: Path, output_path: Path) -> Path:
    envelope = load_result(input_path)
    if not (
        envelope.experiment_id.startswith("stage3-e03-kernel-comparison-")
        or envelope.experiment_id.startswith("stage3-r12c-exact-safe-interleaved-")
    ):
        raise ValueError("renderer requires an E03 or R12c timing artifact")
    session = envelope.payload["sessions"][0]
    if not session.get("complete"):
        raise ValueError("renderer refuses incomplete timing sessions")
    sample_size = int(envelope.protocol["workload"]["sample_size"])
    arm_ids = session["arm_ids"]
    latency = [session["summaries"][arm_id]["median_ns"] / sample_size / 1e6 for arm_id in arm_ids]
    x = np.arange(len(arm_ids))
    figure, axis = plt.subplots(figsize=(max(7, len(arm_ids) * 0.75), 4.5))
    axis.bar(x, latency, color=["#5c4a9e" if arm_id == "dense-fused" else "#2a78d6" for arm_id in arm_ids])
    axis.set_xticks(x)
    axis.set_xticklabels(arm_ids, rotation=50, ha="right", fontsize=8)
    axis.set_ylabel("median ms / query")
    axis.set_title(f"{envelope.experiment_id} — {envelope.dataset_id}")
    axis.grid(axis="y", color="#e9e8e2", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
