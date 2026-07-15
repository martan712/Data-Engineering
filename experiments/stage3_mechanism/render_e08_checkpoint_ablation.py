"""CLI wrapper for the validated e08 artifact renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.render.checkpoint_ablation import render_checkpoint_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(render_checkpoint_ablation(arguments.input, arguments.output))


if __name__ == "__main__":
    main()
