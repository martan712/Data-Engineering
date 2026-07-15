"""CLI for the validated E02 artifact renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.render.stage3.survival import render_survival


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(render_survival(arguments.input, arguments.output))


if __name__ == "__main__":
    main()
