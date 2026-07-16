"""Generate controlled-pilot figures from a tracked result JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PROJECT_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARM_PATTERN = re.compile(r"^(faiss|pdx)_ivf_c(.+)$")
COLORS = {"faiss": "#187a8c", "pdx": "#d17a22", "exact": "#b23a48"}
MARKERS = {"faiss": "o", "pdx": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/final/scifact_ivf_test_40q_final.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    if result.get("schema_version") not in {
        "controlled_ivf_pilot_v1",
        "controlled_ivf_plaid_v2",
    }:
        raise SystemExit(f"Unsupported result schema: {result.get('schema_version')}")
    return result


def sorted_arms(result: dict, engine: str) -> list[tuple[str, dict]]:
    arms = []
    for name, value in result["approximate_arms"].items():
        match = ARM_PATTERN.match(name)
        if not match or match.group(1) != engine:
            continue
        budget = match.group(2)
        sort_value = float("inf") if budget == "full" else int(budget)
        arms.append((name, value, sort_value))
    return [(name, value) for name, value, _ in sorted(arms, key=lambda item: item[2])]


def budget_label(name: str, arm: dict) -> str:
    budget = ARM_PATTERN.match(name).group(2)
    if budget == "full":
        return f"pool ({arm['work']['mean_selected_candidates']:.0f})"
    return f"C={budget}"


def quality_latency_figure(result: dict, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for engine, label in (("faiss", "FAISS-IVF + exact rerank"), ("pdx", "PDX-IVF + exact rerank")):
        arms = sorted_arms(result, engine)
        times = [arm["timing"]["end_to_end_summary"]["median"] for _, arm in arms]
        recalls = [arm["exact_recovery"]["mean_recall@10"] for _, arm in arms]
        axis.plot(
            times,
            recalls,
            color=COLORS[engine],
            marker=MARKERS[engine],
            linewidth=2,
            markersize=6,
            label=label,
        )
        for (name, arm), x_value, y_value in zip(arms, times, recalls):
            axis.annotate(
                budget_label(name, arm),
                (x_value, y_value),
                xytext=(5, 5 if engine == "faiss" else -12),
                textcoords="offset points",
                fontsize=8,
                color=COLORS[engine],
            )

    exact_time = result["exact_reference"]["timing"]["end_to_end_summary"]["median"]
    axis.scatter(
        [exact_time],
        [1.0],
        marker="*",
        s=150,
        color=COLORS["exact"],
        label="Compiled exact MaxSim",
        zorder=4,
    )
    axis.set_xlabel("Online latency for 40 queries (median seconds)")
    axis.set_ylabel("Recall@10 versus exact MaxSim ranking")
    axis.set_ylim(0.84, 1.012)
    axis.grid(axis="both", alpha=0.22)
    axis.legend(loc="lower right", frameon=False)
    axis.set_title("SciFact held-out quality-latency frontier (controlled clean run)")
    fig.text(
        0.5,
        0.01,
        "5,183 documents; L=100; nprobe=8; 4 pinned physical cores; median of 5 interleaved runs. Online boundary only.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "fig5_controlled_quality_latency.png", dpi=180)
    plt.close(fig)


def stage_breakdown_figure(result: dict, output_dir: Path) -> None:
    selected_names = [
        "faiss_ivf_c50",
        "pdx_ivf_c50",
        "faiss_ivf_c200",
        "pdx_ivf_c200",
        "faiss_ivf_cfull",
        "pdx_ivf_cfull",
    ]
    labels = ["FAISS\nC=50", "PDX\nC=50", "FAISS\nC=200", "PDX\nC=200", "FAISS\npool", "PDX\npool"]
    search = []
    aggregation = []
    rerank = []
    total = []
    for name in selected_names:
        arm = result["approximate_arms"][name]
        stages = arm["timing"]["stage_summaries"]
        search.append(stages["search"]["median"])
        aggregation.append(stages["aggregate"]["median"] + stages["select"]["median"])
        rerank.append(stages["exact_rerank_topk"]["median"])
        total.append(arm["timing"]["end_to_end_summary"]["median"])

    x_values = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.bar(x_values, search, color="#187a8c", label="Token search")
    axis.bar(x_values, aggregation, bottom=search, color="#d9a441", label="Aggregation + selection")
    lower = np.asarray(search) + np.asarray(aggregation)
    axis.bar(x_values, rerank, bottom=lower, color="#7a5c99", label="Exact rerank + top-k")
    axis.scatter(x_values, total, marker="_", s=180, linewidth=2, color="#222222", label="Measured end-to-end median")
    axis.set_xticks(x_values, labels)
    axis.set_ylabel("Seconds for 40 queries")
    axis.set_title("Where online time is spent (controlled clean run)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    fig.text(
        0.5,
        0.01,
        "Stage medians are diagnostic and need not sum exactly to the outer-timer median.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "fig6_controlled_stage_breakdown.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = load_result(input_path)
    quality_latency_figure(result, output_dir)
    stage_breakdown_figure(result, output_dir)
    print(f"Saved: {output_dir / 'fig5_controlled_quality_latency.png'}")
    print(f"Saved: {output_dir / 'fig6_controlled_stage_breakdown.png'}")


if __name__ == "__main__":
    main()
