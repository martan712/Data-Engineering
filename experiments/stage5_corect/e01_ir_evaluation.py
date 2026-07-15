"""Stage 5 e01 IR evaluation declarative driver.

Fixture smoke run (diagnostic only):
    uv run python -m experiments.stage5_corect.e01_ir_evaluation --fixture

Full qrels-test workload:
    uv run python -m experiments.stage5_corect.e01_ir_evaluation \
        --dataset scifact --threads 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage5.ir_evaluation import (
    DATASETS,
    Stage5IREvaluationConfig,
    run_ir_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="scifact")
    parser.add_argument("--threads", type=int, choices=(0, 1), default=1)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "json"
    )
    arguments = parser.parse_args()
    config = (
        Stage5IREvaluationConfig.fixture_config()
        if arguments.fixture
        else Stage5IREvaluationConfig(
            dataset=arguments.dataset, n_threads=arguments.threads
        )
    )
    if arguments.fixture and arguments.threads != 1:
        config = replace(config, n_threads=arguments.threads)
    result = run_ir_evaluation(
        config,
        output_dir=arguments.output_dir,
        session_id=arguments.session_id,
    )
    print(result.output_path)


if __name__ == "__main__":
    main()
