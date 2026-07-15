"""Render Stage 5 IR tables and figures exclusively from validated artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from bondmaxsim.results import load_result


def _rows_and_latency(path: Path) -> tuple[str, list[dict[str, Any]]]:
    envelope = load_result(path)
    if envelope.payload_kind != "timing_comparison":
        raise ValueError("Stage 5 rendering requires timing-comparison artifacts")
    sessions = envelope.payload.get("sessions", [])
    if not sessions or not all(session.get("complete") for session in sessions):
        raise ValueError("Stage 5 rendering refuses incomplete artifacts")
    quality = envelope.payload.get("quality", {}).get("rows")
    if not isinstance(quality, list) or not quality:
        raise ValueError("Stage 5 artifact has no quality rows")
    query_count = envelope.protocol.get("workload", {}).get("sample_size")
    if not isinstance(query_count, int) or query_count <= 0:
        raise ValueError("Stage 5 artifact has no positive workload sample size")
    summaries = sessions[-1].get("summaries", {})
    rows = []
    for row in quality:
        summary = summaries.get(row["arm_id"])
        if summary is None:
            raise ValueError(f"missing timing summary for {row['arm_id']}")
        rows.append(
            {
                **row,
                "best_observed_ms_per_query": summary["best_observed_ns"] / query_count / 1e6,
                "median_ms_per_query": summary["median_ns"] / query_count / 1e6,
            }
        )
    return envelope.dataset_id, rows


def latex_rows(paths: Iterable[Path]) -> str:
    """Return deterministic tabular rows; surrounding LaTeX stays in the paper."""
    blocks = []
    for path in paths:
        dataset, rows = _rows_and_latency(path)
        rendered = []
        for row in rows:
            oracle = row.get("recall_vs_oracle_set")
            oracle_text = "--" if oracle is None else f"{oracle:.3f}"
            rendered.append(
                f" & {row['arm_id']} & {row['ndcg_at_10']:.3f} & "
                f"{row['recall_at_100']:.3f} & {row['mrr_at_10']:.3f} & "
                f"{oracle_text} & {row['best_observed_ms_per_query']:.1f} \\\\"
            )
        blocks.append(
            f"\\multirow{{{len(rendered)}}}{{*}}{{{dataset}}}\n" + "\n".join(rendered)
        )
    return "\n\\midrule\n".join(blocks) + "\n"


def render_quality_latency(path: Path, output_path: Path) -> Path:
    """Render a quality/latency diagnostic after reopening the saved artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset, rows = _rows_and_latency(path)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for column, (metric, label) in enumerate(
        (("ndcg_at_10", "nDCG@10"), ("recall_at_100", "recall@100"))
    ):
        axis = axes[0][column]
        for row in rows:
            latency = row["best_observed_ms_per_query"]
            axis.scatter(latency, row[metric], s=35)
            axis.annotate(row["arm_id"], (latency, row[metric]), xytext=(3, 3), textcoords="offset points", fontsize=6)
        axis.set_xscale("log")
        axis.set_xlabel("best observed ms/query")
        axis.set_ylabel(label)
        axis.grid(color="#e9e8e2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Stage 5 IR quality/latency — {dataset}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path

