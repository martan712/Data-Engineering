"""Command-line entry point for paper asset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.render.paper.generator import (
    StalePaperOutputError,
    check_generated_outputs,
    generate_paper_assets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if a generated output is missing or stale; never write files",
    )
    args = parser.parse_args()
    try:
        paths = (
            check_generated_outputs(workspace=args.workspace)
            if args.check
            else generate_paper_assets(workspace=args.workspace)
        )
    except StalePaperOutputError as error:
        parser.exit(1, f"{error}\n")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
