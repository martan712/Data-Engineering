"""Shared path helpers for scripts under experiments/."""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent


def setup_imports() -> None:
    """Ensure `utils_colbert` and sibling modules are importable."""
    experiments = str(EXPERIMENTS_DIR)
    if experiments not in sys.path:
        sys.path.insert(0, experiments)
