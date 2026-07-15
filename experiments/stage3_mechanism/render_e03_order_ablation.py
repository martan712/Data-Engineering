"""CLI for the validated E03 artifact renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.render.stage3.kernel_comparison import render_kernel_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(render_kernel_comparison(arguments.input, arguments.output))


if __name__ == "__main__":
    main()
