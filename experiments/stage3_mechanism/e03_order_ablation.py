"""Stage 3 E03 counterbalanced fused-kernel comparison driver."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage3.kernel_comparison import KernelComparisonConfig, run_kernel_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "json")
    arguments = parser.parse_args()
    config = KernelComparisonConfig.fixture_config() if arguments.fixture else KernelComparisonConfig(dataset=arguments.dataset, n_threads=arguments.threads)
    if arguments.fixture and arguments.threads != config.n_threads:
        config = replace(config, n_threads=arguments.threads)
    result = run_kernel_comparison(config, output_dir=arguments.output_dir, session_id=arguments.session_id)
    print(result.output_path)


if __name__ == "__main__":
    main()
