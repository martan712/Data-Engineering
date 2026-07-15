"""Stage 3 E01 bound-slack accounting driver."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage3.bound_slack import BoundSlackConfig, run_bound_slack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--session-id", default="stage3-e01-accounting")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "json")
    arguments = parser.parse_args()
    config = BoundSlackConfig.fixture_config() if arguments.fixture else BoundSlackConfig(dataset=arguments.dataset)
    result = run_bound_slack(config, output_dir=arguments.output_dir, session_id=arguments.session_id)
    print(result.output_path)


if __name__ == "__main__":
    main()
