"""Stage 5 e03 standalone BM25 quality and diagnostic timing driver."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage5.bm25 import Stage5BM25Config, run_bm25
from bondmaxsim.experiments.stage5.ir_evaluation import DATASETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="scifact")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "json"
    )
    arguments = parser.parse_args()
    config = (
        Stage5BM25Config.fixture_config()
        if arguments.fixture
        else Stage5BM25Config(dataset=arguments.dataset)
    )
    result = run_bm25(
        config,
        output_dir=arguments.output_dir,
        session_id=arguments.session_id,
    )
    print(result.output_path)


if __name__ == "__main__":
    main()
