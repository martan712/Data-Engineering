"""Counterbalanced timing with raw observations and explicit scope boundaries."""

from __future__ import annotations

import contextvars
import hashlib
import math
import statistics
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator

from bondmaxsim.experiments.arms import ArmSpec

PROTOCOL_PRESETS = {
    "fixture": (1, 3),
    "mechanism": (2, 9),
    "full_system": (1, 5),
    "confirmation": (1, 9),
}


class TimingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimingProtocol:
    protocol_class: str
    warmup_rounds: int
    measured_rounds: int

    def __post_init__(self) -> None:
        if self.protocol_class not in PROTOCOL_PRESETS:
            raise ValueError(f"unknown timing protocol {self.protocol_class!r}")
        if self.warmup_rounds < 0 or self.measured_rounds <= 0:
            raise ValueError("timing rounds must be non-negative with measurements > 0")

    @classmethod
    def preset(cls, protocol_class: str) -> "TimingProtocol":
        warmups, measured = PROTOCOL_PRESETS[protocol_class]
        return cls(protocol_class, warmups, measured)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScopeEvent:
    scope_id: str
    phase: str
    started_ns: int
    ended_ns: int
    measured: bool


_ACTIVE_MEASURED_SCOPE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bondmaxsim_active_measured_scope", default=None
)


@dataclass
class ScopeLedger:
    clock_ns: Callable[[], int] = time.perf_counter_ns
    events: list[ScopeEvent] = field(default_factory=list)

    @contextmanager
    def scope(self, scope_id: str, phase: str, *, measured: bool = False) -> Iterator[None]:
        active = _ACTIVE_MEASURED_SCOPE.get()
        if measured and active is not None:
            raise TimingError(f"measured scope {scope_id!r} nested inside {active!r}")
        if not measured and active is not None:
            raise TimingError(
                f"excluded scope {scope_id!r} entered inside measured scope {active!r}"
            )
        token = _ACTIVE_MEASURED_SCOPE.set(scope_id) if measured else None
        started = self.clock_ns()
        try:
            yield
        finally:
            ended = self.clock_ns()
            if token is not None:
                _ACTIVE_MEASURED_SCOPE.reset(token)
            self.events.append(ScopeEvent(scope_id, phase, started, ended, measured))

    def assert_outside_measured(self, operation: str) -> None:
        active = _ACTIVE_MEASURED_SCOPE.get()
        if active is not None:
            raise TimingError(f"{operation} cannot run inside measured scope {active!r}")


@dataclass(frozen=True)
class TimingObservation:
    session_id: str
    round_index: int
    phase: str
    arm_id: str
    arm_position: int
    started_at_utc: str
    elapsed_ns: int
    timer_scope_id: str
    completed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _linear_percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_values(values: Iterable[int]) -> dict[str, Any]:
    measured = [int(value) for value in values]
    if not measured or any(value < 0 for value in measured):
        raise TimingError("timing summaries need non-negative observations")
    q25 = _linear_percentile(measured, 0.25)
    q75 = _linear_percentile(measured, 0.75)
    return {
        "count": len(measured),
        "values_ns": measured,
        "best_observed_ns": min(measured),
        "median_ns": float(statistics.median(measured)),
        "minimum_ns": min(measured),
        "maximum_ns": max(measured),
        "q25_ns": q25,
        "q75_ns": q75,
        "iqr_ns": q75 - q25,
    }


def session_offset(session_id: str, arm_count: int) -> int:
    if not session_id or arm_count <= 0:
        raise ValueError("session_id and arm_count must be non-empty")
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % arm_count


def new_session_id(prefix: str = "session") -> str:
    """Return a fresh stable-format process/session identifier."""
    if not prefix or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in prefix):
        raise ValueError("session prefix must be lowercase alphanumeric with hyphens")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz").lower()
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:12]}"


def cyclic_order(arms: tuple[ArmSpec, ...], session_id: str, round_index: int) -> tuple[ArmSpec, ...]:
    start = (session_offset(session_id, len(arms)) + round_index) % len(arms)
    return arms[start:] + arms[:start]


@dataclass
class TimingSession:
    session_id: str
    protocol: TimingProtocol
    arm_ids: tuple[str, ...]
    observations: list[TimingObservation]
    scope_events: list[ScopeEvent]
    validation_results: list[dict[str, Any]]
    complete: bool
    environment: dict[str, Any]

    def measured(self) -> list[TimingObservation]:
        return [
            row for row in self.observations
            if row.phase == "measured" and row.completed
        ]

    def summaries(self) -> dict[str, dict[str, Any]]:
        if not self.complete:
            raise TimingError("incomplete timing sessions have no complete summary")
        return {
            arm_id: summarize_values(
                row.elapsed_ns for row in self.measured() if row.arm_id == arm_id
            )
            for arm_id in self.arm_ids
        }

    def paired_comparisons(self, baseline_arm_id: str) -> dict[str, dict[str, Any]]:
        if not self.complete:
            raise TimingError("incomplete timing sessions cannot be compared")
        if baseline_arm_id not in self.arm_ids:
            raise TimingError("baseline arm is not present")
        rounds: dict[int, dict[str, int]] = {}
        for row in self.measured():
            rounds.setdefault(row.round_index, {})[row.arm_id] = row.elapsed_ns
        result = {}
        for candidate in self.arm_ids:
            if candidate == baseline_arm_id:
                continue
            ratios, margins = [], []
            for round_index in sorted(rounds):
                values = rounds[round_index]
                if baseline_arm_id not in values or candidate not in values:
                    raise TimingError("paired comparison encountered incomplete round")
                baseline = values[baseline_arm_id]
                if baseline <= 0:
                    raise TimingError("baseline elapsed time must be positive")
                ratios.append(values[candidate] / baseline)
                margins.append(100.0 * (baseline - values[candidate]) / baseline)
            result[candidate] = {
                "baseline_arm_id": baseline_arm_id,
                "paired_ratios": ratios,
                "paired_margins_pct": margins,
                "median_paired_ratio": float(statistics.median(ratios)),
                "median_paired_margin_pct": float(statistics.median(margins)),
            }
        return result

    def to_dict(self, baseline_arm_id: str | None = None) -> dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "protocol": self.protocol.to_dict(),
            "arm_ids": list(self.arm_ids),
            "complete": self.complete,
            "observations": [row.to_dict() for row in self.observations],
            "scope_events": [asdict(event) for event in self.scope_events],
            "validation_results": self.validation_results,
            "environment": self.environment,
        }
        if self.complete:
            result["summaries"] = self.summaries()
            if baseline_arm_id is not None:
                result["paired_comparisons"] = self.paired_comparisons(baseline_arm_id)
        return result


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def run_timing_session(
    arms: tuple[ArmSpec, ...],
    protocol: TimingProtocol,
    *,
    session_id: str,
    environment: dict[str, Any],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    utc_timestamp: Callable[[], str] = _utc_timestamp,
    ledger: ScopeLedger | None = None,
) -> TimingSession:
    """Execute counterbalanced rounds; validation always occurs after timing."""
    if not arms:
        raise ValueError("at least one arm is required")
    if len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("arm IDs must be unique")
    ledger = ledger or ScopeLedger(clock_ns)
    observations: list[TimingObservation] = []
    validation_results: list[dict[str, Any]] = []
    complete = True

    for phase, rounds in (("warmup", protocol.warmup_rounds), ("measured", protocol.measured_rounds)):
        for round_index in range(rounds):
            ordered = cyclic_order(arms, session_id, round_index)
            round_failed = False
            for position, arm in enumerate(ordered):
                started_at = utc_timestamp()
                started_ns = clock_ns()
                result = None
                error = None
                try:
                    with ledger.scope(arm.timer_scope_id, phase, measured=True):
                        result = arm.operation()
                except BaseException as caught:
                    error = f"{type(caught).__name__}: {caught}"
                ended_ns = clock_ns()
                completed = error is None
                observations.append(
                    TimingObservation(
                        session_id=session_id,
                        round_index=round_index,
                        phase=phase,
                        arm_id=arm.arm_id,
                        arm_position=position,
                        started_at_utc=started_at,
                        elapsed_ns=max(0, ended_ns - started_ns),
                        timer_scope_id=arm.timer_scope_id,
                        completed=completed,
                        error=error,
                    )
                )
                if completed and arm.validator is not None:
                    try:
                        with ledger.scope("result-validation", phase, measured=False):
                            validation_result = arm.validator(result)
                        if validation_result is not None and hasattr(validation_result, "to_dict"):
                            validation_results.append(
                                {
                                    "phase": phase,
                                    "round_index": round_index,
                                    "arm_id": arm.arm_id,
                                    "result": validation_result.to_dict(),
                                }
                            )
                        if (
                            arm.exactness == "exact_safe"
                            and validation_result is not None
                            and hasattr(validation_result, "exact_gate_passed")
                            and not validation_result.exact_gate_passed
                        ):
                            raise TimingError(
                                f"exact-safe arm {arm.arm_id!r} failed correctness gate"
                            )
                    except BaseException as caught:
                        observations[-1] = TimingObservation(
                            **{
                                **observations[-1].to_dict(),
                                "completed": False,
                                "error": f"validation {type(caught).__name__}: {caught}",
                            }
                        )
                        completed = False
                if not completed:
                    complete = False
                    round_failed = True
                    break
            if round_failed:
                break
        if not complete:
            break

    expected = protocol.measured_rounds * len(arms)
    if len([row for row in observations if row.phase == "measured" and row.completed]) != expected:
        complete = False
    return TimingSession(
        session_id=session_id,
        protocol=protocol,
        arm_ids=tuple(arm.arm_id for arm in arms),
        observations=observations,
        scope_events=list(ledger.events),
        validation_results=validation_results,
        complete=complete,
        environment=environment,
    )


def classify_paired_sessions(
    sessions: Iterable[TimingSession],
    *,
    candidate_arm_id: str,
    baseline_arm_id: str,
) -> dict[str, Any]:
    """Apply the accepted 3% tie band and fresh-session confirmation rule."""
    session_list = list(sessions)
    if not session_list:
        raise TimingError("at least one timing session is required")
    if len({session.session_id for session in session_list}) != len(session_list):
        raise TimingError("paired confirmation requires fresh session IDs")
    per_session = []
    pooled = []
    for session in session_list:
        comparison = session.paired_comparisons(baseline_arm_id).get(candidate_arm_id)
        if comparison is None:
            raise TimingError("candidate arm is absent from a timing session")
        margin = comparison["median_paired_margin_pct"]
        per_session.append(margin)
        pooled.extend(comparison["paired_margins_pct"])
    pooled_median = float(statistics.median(pooled))
    if -3.0 <= pooled_median <= 3.0:
        decision = "tie"
    elif pooled_median > 3.0 and all(margin > 0 for margin in per_session):
        decision = "win"
        if pooled_median < 10.0 and (
            len(session_list) < 3
            or any(session.protocol.protocol_class != "confirmation" for session in session_list)
        ):
            decision = "insufficient_confirmation"
    elif pooled_median < -3.0 and all(margin < 0 for margin in per_session):
        decision = "loss"
    else:
        decision = "inconsistent"
    return {
        "candidate_arm_id": candidate_arm_id,
        "baseline_arm_id": baseline_arm_id,
        "session_ids": [session.session_id for session in session_list],
        "per_session_median_paired_margin_pct": per_session,
        "pooled_median_paired_margin_pct": pooled_median,
        "equivalence_band_pct": [-3.0, 3.0],
        "decision": decision,
    }
