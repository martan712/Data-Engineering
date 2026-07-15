"""Render Stage 5 table rows or a diagnostic plot from validated artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from bondmaxsim.render.stage5.ir import latex_rows, render_quality_latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--figure", type=Path)
    arguments = parser.parse_args()
    if arguments.figure is not None:
        if len(arguments.artifacts) != 1:
            parser.error("--figure accepts exactly one Stage 5 artifact")
        print(render_quality_latency(arguments.artifacts[0], arguments.figure))
    else:
        print(latex_rows(arguments.artifacts), end="")


if __name__ == "__main__":
    main()
