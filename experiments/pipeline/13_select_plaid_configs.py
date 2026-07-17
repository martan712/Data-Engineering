"""Freeze PLAID release configurations from validation-query results only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PROJECT_ROOT  # noqa: E402
from benchmarking import sha256_file  # noqa: E402
from plaid_benchmark import select_validation_configs  # noqa: E402
from utils_colbert import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recall-targets", nargs="+", type=float, default=[0.85, 0.95])
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    if any(target <= 0.0 or target > 1.0 for target in args.recall_targets):
        raise SystemExit("recall targets must be in (0, 1]")

    with input_path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    if result.get("schema_version") != "controlled_ivf_plaid_v2":
        raise SystemExit(f"Unsupported validation schema: {result.get('schema_version')}")
    dataset = result["dataset"]
    if dataset["query_start"] != 0 or dataset["queries"] > 10:
        raise SystemExit(
            "PLAID configurations must be frozen from the first ten validation queries only"
        )
    if not result.get("plaid_arms"):
        raise SystemExit("Validation result contains no PLAID arms")

    selection = select_validation_configs(
        result["plaid_arms"],
        recall_targets=args.recall_targets,
    )
    output = {
        "schema_version": "plaid_validation_selection_v1",
        "selection_policy": (
            "fastest validation arm meeting each declared exact-recall@10 target, "
            "plus the highest-recall arm; ties are deterministic"
        ),
        "validation_result": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "git": result["metadata"]["git"],
            "dataset": dataset,
        },
        **selection,
    }
    save_json(output_path, output)
    for item in output["selected"]:
        config = item["config"]
        print(
            f"{item['name']}: nprobe={config['n_ivf_probe']} "
            f"full_scores={config['n_full_scores']} "
            f"validationR@10={item['validation_exact_recall@10']:.4f} "
            f"median={item['validation_median_seconds']:.6f}s"
        )
    print(f"Saved frozen selection: {output_path}")


if __name__ == "__main__":
    main()
