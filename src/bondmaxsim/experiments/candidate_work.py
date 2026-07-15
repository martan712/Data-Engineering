"""Typed, per-query candidate-work observations.

Configured controls and observed work deliberately use the same small value /
quality / source envelope, while remaining separate named fields.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from statistics import mean, median
from typing import Iterable, Literal

CountQuality = Literal["exact", "estimated", "upper_bound", "unavailable"]
_QUALITIES = {"exact", "estimated", "upper_bound", "unavailable"}


@dataclass(frozen=True)
class CountObservation:
    value: int | None
    quality: CountQuality
    source: str

    def __post_init__(self) -> None:
        if self.quality not in _QUALITIES:
            raise ValueError(f"invalid count quality {self.quality!r}")
        if not self.source:
            raise ValueError("count source must be non-empty")
        if self.quality == "unavailable":
            if self.value is not None:
                raise ValueError("unavailable counts must have value=None")
        elif isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0:
            raise ValueError("available counts must be non-negative integers")

    @classmethod
    def exact(cls, value: int, source: str) -> "CountObservation":
        return cls(value=int(value), quality="exact", source=source)

    @classmethod
    def unavailable(cls, source: str) -> "CountObservation":
        return cls(value=None, quality="unavailable", source=source)

    def to_dict(self) -> dict[str, int | str | None]:
        return {"value": self.value, "quality": self.quality, "source": self.source}


_DEFAULT_UNAVAILABLE = "not instrumented by this arm"


@dataclass(frozen=True)
class CandidateWorkObservation:
    configured_candidate_cap: CountObservation
    configured_full_score_cap: CountObservation
    unique_candidates_generated: CountObservation
    documents_admitted_to_scoring: CountObservation
    documents_fully_scored: CountObservation
    documents_probed: CountObservation
    partitions_probed: CountObservation
    token_hits_inspected: CountObservation

    @classmethod
    def empty(cls, source: str = _DEFAULT_UNAVAILABLE) -> "CandidateWorkObservation":
        missing = CountObservation.unavailable(source)
        return cls(**{field.name: missing for field in fields(cls)})

    def to_dict(self) -> dict[str, dict[str, int | str | None]]:
        return {
            field.name: getattr(self, field.name).to_dict()
            for field in fields(self)
        }

    def require_equal_work_reranking(self) -> None:
        """Reject equal-work wording unless actual full-score work is exact."""
        if self.documents_fully_scored.quality != "exact":
            raise ValueError(
                "equal_work_reranking requires an exact documents_fully_scored count"
            )


def summarize_count_observations(
    observations: Iterable[CountObservation],
) -> dict[str, int | float | None]:
    """Summarize retained observations without converting missing work to zero."""
    rows = list(observations)
    values = [row.value for row in rows if row.value is not None]
    return {
        "count": len(rows),
        "min": min(values) if values else None,
        "median": float(median(values)) if values else None,
        "mean": float(mean(values)) if values else None,
        "max": max(values) if values else None,
        "unavailable": sum(row.quality == "unavailable" for row in rows),
    }


def summarize_candidate_work(
    observations: Iterable[CandidateWorkObservation],
) -> dict[str, dict[str, int | float | None]]:
    rows = list(observations)
    return {
        field.name: summarize_count_observations(
            getattr(row, field.name) for row in rows
        )
        for field in fields(CandidateWorkObservation)
    }
