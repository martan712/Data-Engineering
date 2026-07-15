"""Stage 2 S02 versioned native exact-agreement audit driver."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage3.stage2_audit import run_exact_agreement_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "json")
    arguments = parser.parse_args()
    datasets = ["synthetic-small-v1"] if arguments.fixture else (arguments.dataset or ["scifact"])
    for dataset in datasets:
        result = run_exact_agreement_audit(dataset, fixture=arguments.fixture, output_dir=arguments.output_dir)
        print(result.output_path)


if __name__ == "__main__":
    main()
