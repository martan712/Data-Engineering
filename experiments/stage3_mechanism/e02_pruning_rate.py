"""Stage 3 E02 native survival-accounting driver."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage3.survival import SurvivalConfig, run_survival


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--session-id", default="stage3-e02-accounting")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "json")
    arguments = parser.parse_args()
    config = SurvivalConfig.fixture_config() if arguments.fixture else SurvivalConfig(dataset=arguments.dataset)
    result = run_survival(config, output_dir=arguments.output_dir, session_id=arguments.session_id)
    print(result.output_path)


if __name__ == "__main__":
    main()
