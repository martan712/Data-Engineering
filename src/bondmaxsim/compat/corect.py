"""Pinned CoRECT checkout verification and import isolation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bondmaxsim.config import EXTERN_DIR

CORECT_DIR = EXTERN_DIR / "CoRECT"
CORECT_PINNED_COMMIT = "fedf8bb296f8a1f01c76df28fd9b80596646f015"


@dataclass(frozen=True)
class CorectCapabilities:
    revision: str
    standard_metric_function: str = "corect.utils.evaluate_results"
    relevance_composition: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "revision": self.revision,
            "standard_metric_function": self.standard_metric_function,
            "relevance_composition": self.relevance_composition,
        }


def inspect_corect_checkout(checkout: Path = CORECT_DIR) -> CorectCapabilities:
    """Fail closed unless the configured checkout is the accepted revision."""
    utility = checkout / "src" / "corect" / "utils.py"
    if not utility.is_file():
        raise RuntimeError(f"CoRECT evaluate_results source is missing: {utility}")
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if revision != CORECT_PINNED_COMMIT:
        raise RuntimeError(
            f"CoRECT checkout must be {CORECT_PINNED_COMMIT}, found {revision}"
        )
    return CorectCapabilities(revision=revision)


def load_corect_evaluate_results(
    checkout: Path = CORECT_DIR,
) -> Callable[..., Any]:
    """Load the pinned function while containing CoRECT's circular-import shim."""
    inspect_corect_checkout(checkout)
    source = (checkout / "src").resolve()
    source_text = str(source)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        # This order is required by pinned CoRECT: utils imports the abstract
        # wrapper back through the package initializer.
        import corect.model_wrappers  # noqa: F401
        from corect import utils

        module_path = Path(utils.__file__).resolve()
        if source not in module_path.parents:
            raise RuntimeError(
                f"imported CoRECT from {module_path}, outside pinned checkout {source}"
            )
        evaluate_results = getattr(utils, "evaluate_results", None)
        if not callable(evaluate_results):
            raise RuntimeError("pinned CoRECT has no callable evaluate_results")
        return evaluate_results
    finally:
        if inserted:
            sys.path.remove(source_text)
