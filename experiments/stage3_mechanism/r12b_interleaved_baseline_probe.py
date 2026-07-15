"""Stage 3 R12b shared-timing baseline-artifact diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage3.kernel_comparison import (
    BaselineProbeConfig,
    run_baseline_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="arguana")
    parser.add_argument("--queries", type=int, default=120)
    parser.add_argument("--threads", type=int, choices=(0, 1), default=0)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "json",
    )
    arguments = parser.parse_args()
    config = (
        BaselineProbeConfig.fixture_config()
        if arguments.fixture
        else BaselineProbeConfig(
            dataset=arguments.dataset,
            query_count=arguments.queries,
            n_threads=arguments.threads,
        )
    )
    if arguments.fixture and arguments.threads != config.n_threads:
        config = replace(config, n_threads=arguments.threads)
    result = run_baseline_probe(
        config,
        output_dir=arguments.output_dir,
        session_id=arguments.session_id,
    )
    print(result.output_path)


if __name__ == "__main__":
    main()
