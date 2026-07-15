"""Stage 5 e02 paired analysis over a saved, validated e01 artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.stage5.significance import significance_from_ir_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arm", action="append", dest="arms")
    parser.add_argument("--n-permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "json"
    )
    arguments = parser.parse_args()
    result = significance_from_ir_artifact(
        arguments.input,
        output_dir=arguments.output_dir,
        candidate_arm_ids=arguments.arms,
        n_permutations=arguments.n_permutations,
        seed=arguments.seed,
    )
    print(result.output_path)


if __name__ == "__main__":
    main()
