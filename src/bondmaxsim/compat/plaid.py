"""Feature and observability checks for the pinned PyLate PLAID adapter."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

PYLATE_PINNED_VERSION = "1.6.0"


@dataclass(frozen=True)
class PlaidCapabilities:
    version: str
    public_search_settings: bool
    actual_full_score_count: bool
    settings_source: str
    count_source: str

    def require_system_cap(self, comparison_scope: str) -> None:
        if comparison_scope != "system_cap":
            raise ValueError(
                "PLAID actual full-score work is unavailable; only system_cap "
                "comparison wording is permitted"
            )

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "version": self.version,
            "public_search_settings": self.public_search_settings,
            "actual_full_score_count": self.actual_full_score_count,
            "settings_source": self.settings_source,
            "count_source": self.count_source,
        }


def inspect_plaid_capabilities(
    plaid_class: type[Any] | None = None,
    *,
    installed_version: str | None = None,
) -> PlaidCapabilities:
    """Verify the pinned public constructor/call surface used by the wrapper."""
    if installed_version is None:
        try:
            installed_version = version("pylate")
        except PackageNotFoundError as exc:
            raise RuntimeError("PyLate is not installed") from exc
    if installed_version != PYLATE_PINNED_VERSION:
        raise RuntimeError(
            f"PyLate {PYLATE_PINNED_VERSION} is required, found {installed_version}"
        )

    if plaid_class is None:
        from pylate.indexes import PLAID

        plaid_class = PLAID
    parameters = inspect.signature(plaid_class).parameters
    required = {"n_ivf_probe", "n_full_scores", "index_folder", "index_name"}
    missing = sorted(required - set(parameters))
    if missing or not callable(getattr(plaid_class, "__call__", None)):
        detail = ", ".join(missing) if missing else "call operation"
        raise RuntimeError(f"pinned PyLate PLAID public API is missing {detail}")

    return PlaidCapabilities(
        version=installed_version,
        public_search_settings=True,
        actual_full_score_count=False,
        settings_source=(
            "PyLate PLAID public constructor parameters; index reattached outside timing"
        ),
        count_source=(
            "PyLate 1.6.0 returns ranked results but exposes no actual per-query "
            "full-score count"
        ),
    )
