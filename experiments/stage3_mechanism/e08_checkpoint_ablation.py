"""Stage 3 e08 checkpoint ablation on the shared experiment framework.

This migrated driver declares only experiment selection and protocol inputs.
Preparation, timing, correctness, accounting, provenance, and serialization
are owned by ``bondmaxsim.experiments.checkpoint_ablation``. Rendering is a
separate validated-artifact consumer.

Examples
--------
Fixture smoke run (not performance evidence):
    uv run python -m experiments.stage3_mechanism.e08_checkpoint_ablation --fixture

SciFact one-thread session:
    uv run python -m experiments.stage3_mechanism.e08_checkpoint_ablation \
        --dataset scifact --threads 1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.checkpoint_ablation import (
    CheckpointAblationConfig,
    run_checkpoint_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "json",
    )
    arguments = parser.parse_args()

    config = (
        CheckpointAblationConfig.fixture_config()
        if arguments.fixture
        else CheckpointAblationConfig(
            dataset=arguments.dataset,
            n_threads=arguments.threads,
        )
    )
    if arguments.fixture and arguments.threads != 1:
        config = replace(config, n_threads=arguments.threads)
    result = run_checkpoint_ablation(
        config,
        output_dir=arguments.output_dir,
        session_id=arguments.session_id,
    )
    print(result.output_path)


if __name__ == "__main__":
    main()
