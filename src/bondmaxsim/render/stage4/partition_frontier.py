"""Render e03 latency/recall frontiers from a validated saved envelope."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bondmaxsim.results import load_result


def render_partition_frontier(input_path: Path, output_path: Path) -> Path:
    envelope = load_result(input_path)
    if envelope.payload_kind != "timing_comparison" or "stage4-e03" not in envelope.experiment_id:
        raise ValueError("e03 renderer requires a migrated e03 timing envelope")
    session = envelope.payload["sessions"][0]
    if not session.get("complete"):
        raise ValueError("e03 renderer refuses incomplete sessions")
    sample_size = int(envelope.protocol["workload"]["sample_size"])
    validations: dict[str, float] = {}
    for row in session["validation_results"]:
        if row["phase"] == "measured" and row["arm_id"] not in validations:
            validations[row["arm_id"]] = row["result"]["recall_vs_oracle_set_mean"]
    configurations = {arm["arm_id"]: arm for arm in envelope.method_configuration["arms"]}
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    for scanner, color in (("brute", "#2a78d6"), ("bond", "#2e8b57")):
        arms = [
            arm for arm in session["arm_ids"]
            if arm.startswith(f"partition-{scanner}-")
        ]
        arms.sort(key=lambda arm: configurations[arm]["parameters"]["nprobe"])
        recall = [validations[arm] for arm in arms]
        latency = [session["summaries"][arm]["median_ns"] / sample_size / 1e6 for arm in arms]
        axis.plot(recall, latency, "o-", color=color, label=scanner)
        for arm, x, y in zip(arms, recall, latency):
            axis.annotate(f"p{configurations[arm]['parameters']['nprobe']}", (x, y), xytext=(4, 4), textcoords="offset points")
    axis.set_xlabel("mean recall vs exact top-k")
    axis.set_ylabel("median ms / query")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
