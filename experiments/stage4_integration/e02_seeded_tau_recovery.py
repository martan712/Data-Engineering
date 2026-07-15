"""CLI for migrated Stage 4 e02 seeded-threshold recovery."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage4.seeded_tau import SeededTauConfig, run_seeded_tau_recovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "json")
    arguments = parser.parse_args()
    config = SeededTauConfig.fixture_config() if arguments.fixture else SeededTauConfig(
        dataset=arguments.dataset, n_threads=arguments.threads
    )
    if arguments.fixture and arguments.threads != 1:
        config = replace(config, n_threads=arguments.threads)
    result = run_seeded_tau_recovery(
        config,
        output_dir=arguments.output_dir,
        session_id=arguments.session_id,
    )
    print(result.output_path)


if __name__ == "__main__":
    main()
