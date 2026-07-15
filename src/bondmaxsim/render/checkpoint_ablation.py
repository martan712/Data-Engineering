"""Render checkpoint-ablation figures from validated saved envelopes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bondmaxsim.results import load_result


def render_checkpoint_ablation(input_path: Path, output_path: Path) -> Path:
    envelope = load_result(input_path)
    if envelope.payload_kind != "timing_comparison":
        raise ValueError("e08 renderer requires a timing_comparison envelope")
    session = envelope.payload["sessions"][0]
    if not session.get("complete"):
        raise ValueError("e08 renderer refuses incomplete timing sessions")
    sample_size = int(envelope.protocol["workload"]["sample_size"])
    arm_configs = {
        arm["arm_id"]: arm for arm in envelope.method_configuration["arms"]
    }
    accounting = {}
    for metadata in session["result_metadata"]:
        if metadata["phase"] != "measured" or metadata["arm_id"] in accounting:
            continue
        rows = metadata["pruning_accounting"]
        accounting[metadata["arm_id"]] = {
            "cells_scanned_pct": float(
                np.mean([row["cells_scanned_pct"] for row in rows])
            ),
            "pruned_docs_pct": float(
                np.mean([row["pruned_docs_pct"] for row in rows])
            ),
        }

    arm_ids = [arm_id for arm_id in session["arm_ids"] if arm_id != "dense-fused"]
    labels = [
        "/".join(
            [
                arm_configs[arm_id]["parameters"]["dimension_order"],
                ",".join(
                    str(value)
                    for value in arm_configs[arm_id]["parameters"]["checkpoints"]
                ),
            ]
        )
        for arm_id in arm_ids
    ]
    latency_ms = [
        session["summaries"][arm_id]["median_ns"] / sample_size / 1e6
        for arm_id in arm_ids
    ]
    dense_ms = session["summaries"]["dense-fused"]["median_ns"] / sample_size / 1e6
    cells_pct = [accounting[arm_id]["cells_scanned_pct"] for arm_id in arm_ids]

    x = np.arange(len(arm_ids))
    fig, (latency_axis, accounting_axis) = plt.subplots(1, 2, figsize=(13, 4.5))
    latency_axis.bar(x, latency_ms, color="#2a78d6")
    latency_axis.axhline(
        dense_ms, color="#5c4a9e", linestyle="--", label="dense fused"
    )
    latency_axis.set_ylabel("median ms / query")
    latency_axis.legend(frameon=False)
    accounting_axis.bar(x, cells_pct, color="#eb6834")
    accounting_axis.set_ylabel("mean padded cells scanned %")
    for axis in (latency_axis, accounting_axis):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        axis.grid(axis="y", color="#e9e8e2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"e08 checkpoint ablation — {envelope.dataset_id} "
        f"({envelope.protocol['workload']['workload_id']})"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
