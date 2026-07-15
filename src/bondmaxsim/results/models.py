"""Typed versioned result envelope and semantic payload validation."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

CURRENT_SCHEMA_NAME = "bondmaxsim.result-envelope"
CURRENT_SCHEMA_VERSION = "1.0.0"
PAYLOAD_KINDS = frozenset(
    {
        "timing_comparison",
        "pruning_accounting",
        "candidate_frontier",
        "ir_quality",
        "paired_significance",
        "validation_audit",
    }
)
_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PROVENANCE_KEYS = frozenset(
    {
        "git_revision",
        "git_dirty",
        "lockfile_sha256",
        "configuration_sha256",
        "data_sha256",
        "index_sha256",
        "native_binary_sha256",
        "command",
        "input_artifact_ids",
    }
)


class ResultValidationError(ValueError):
    """A result violates the accepted envelope or payload semantics."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResultValidationError(f"{name} must be finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0:
        raise ResultValidationError(f"{name} must be non-negative")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if not 0 <= result <= 1:
        raise ResultValidationError(f"{name} must be in [0, 1]")
    return result


def _percentage(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if not 0 <= result <= 100:
        raise ResultValidationError(f"{name} must be in [0, 100]")
    return result


def _rows(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ResultValidationError(f"payload.{key} must be a non-empty list")
    if not all(isinstance(row, Mapping) for row in value):
        raise ResultValidationError(f"payload.{key} entries must be objects")
    return value


def _validate_count_observation(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ResultValidationError(f"{name} must be a count observation")
    count = value.get("value")
    quality = value.get("quality")
    if quality not in {"exact", "estimated", "upper_bound", "unavailable"}:
        raise ResultValidationError(f"{name}.quality is invalid")
    if quality == "unavailable":
        if count is not None:
            raise ResultValidationError(f"{name}.value must be null when unavailable")
    elif isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ResultValidationError(f"{name}.value must be a non-negative integer")
    if not isinstance(value.get("source"), str) or not value["source"]:
        raise ResultValidationError(f"{name}.source must be non-empty")


def _validate_timing(payload: Mapping[str, Any]) -> None:
    sessions = _rows(payload, "sessions")
    for session_index, session in enumerate(sessions):
        observations = session.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ResultValidationError("timing session observations must be non-empty")
        measured_by_round: dict[int, list[Mapping[str, Any]]] = {}
        expected_arms = set(session.get("arm_ids", []))
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise ResultValidationError("timing observations must be objects")
            elapsed = observation.get("elapsed_ns")
            if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
                raise ResultValidationError("elapsed_ns must be a non-negative integer")
            if observation.get("phase") not in {"warmup", "measured"}:
                raise ResultValidationError("timing phase is invalid")
            if observation.get("phase") == "measured":
                measured_by_round.setdefault(int(observation.get("round_index", -1)), []).append(
                    observation
                )
        if session.get("complete"):
            for round_observations in measured_by_round.values():
                completed = {
                    row.get("arm_id") for row in round_observations if row.get("completed") is True
                }
                if expected_arms and completed != expected_arms:
                    raise ResultValidationError(
                        f"timing session {session_index} has an incomplete measured round"
                    )


def _validate_pruning(payload: Mapping[str, Any]) -> None:
    for row in _rows(payload, "rows"):
        for key, value in row.items():
            if value is None:
                continue
            if key.endswith("_pct"):
                _percentage(value, key)
            elif key in {"cells_scanned", "documents_pruned", "tokens_pruned", "query_count"}:
                _nonnegative(value, key)


def _validate_candidate(payload: Mapping[str, Any]) -> None:
    scopes = {"system_cap", "equal_work_reranking", "probe_frontier", "uncapped_reference"}
    for point in _rows(payload, "points"):
        if point.get("comparison_scope") not in scopes:
            raise ResultValidationError("candidate comparison_scope is invalid")
        work = point.get("candidate_work")
        if not isinstance(work, Mapping):
            raise ResultValidationError("candidate_work must be an object")
        for name, observation in work.items():
            _validate_count_observation(observation, f"candidate_work.{name}")
        if "recall" in point and point["recall"] is not None:
            _unit_interval(point["recall"], "recall")


def _validate_ir(payload: Mapping[str, Any]) -> None:
    for row in _rows(payload, "rows"):
        for name in ("ndcg_at_10", "recall_at_100", "mrr_at_10", "recall_vs_oracle_set"):
            if name in row and row[name] is not None:
                _unit_interval(row[name], name)
        for name in ("latency_ms", "qps"):
            if name in row and row[name] is not None:
                _nonnegative(row[name], name)


def _validate_significance(payload: Mapping[str, Any]) -> None:
    for comparison in _rows(payload, "comparisons"):
        if "p_value" in comparison:
            _unit_interval(comparison["p_value"], "p_value")
        n_queries = comparison.get("n_queries")
        if isinstance(n_queries, bool) or not isinstance(n_queries, int) or n_queries <= 0:
            raise ResultValidationError("paired significance requires n_queries > 0")


def _validate_audit(payload: Mapping[str, Any]) -> None:
    checks = _rows(payload, "checks")
    statuses = []
    for check in checks:
        if check.get("status") not in {"pass", "fail", "not_run"}:
            raise ResultValidationError("validation check status is invalid")
        statuses.append(check["status"])
    if payload.get("passed") is True and "fail" in statuses:
        raise ResultValidationError("validation audit cannot pass with failed checks")


_PAYLOAD_VALIDATORS = {
    "timing_comparison": _validate_timing,
    "pruning_accounting": _validate_pruning,
    "candidate_frontier": _validate_candidate,
    "ir_quality": _validate_ir,
    "paired_significance": _validate_significance,
    "validation_audit": _validate_audit,
}


@dataclass(frozen=True)
class TimingComparisonPayload:
    sessions: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = {"sessions": [dict(session) for session in self.sessions]}
        _validate_timing(value)
        return value


@dataclass(frozen=True)
class PruningAccountingPayload:
    rows: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = {"rows": [dict(row) for row in self.rows]}
        _validate_pruning(value)
        return value


@dataclass(frozen=True)
class CandidateFrontierPayload:
    points: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = {"points": [dict(point) for point in self.points]}
        _validate_candidate(value)
        return value


@dataclass(frozen=True)
class IRQualityPayload:
    rows: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = {"rows": [dict(row) for row in self.rows]}
        _validate_ir(value)
        return value


@dataclass(frozen=True)
class PairedSignificancePayload:
    comparisons: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = {"comparisons": [dict(row) for row in self.comparisons]}
        _validate_significance(value)
        return value


@dataclass(frozen=True)
class ValidationAuditPayload:
    checks: tuple[Mapping[str, Any], ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        value = {"checks": [dict(check) for check in self.checks], "passed": self.passed}
        _validate_audit(value)
        return value


@dataclass(frozen=True)
class ExperimentResultEnvelope:
    schema_name: str
    schema_version: str
    artifact_id: str
    experiment_id: str
    created_at_utc: str
    dataset_id: str
    workload_id: str
    method_configuration: Mapping[str, Any]
    protocol: Mapping[str, Any]
    environment: Mapping[str, Any]
    provenance: Mapping[str, Any]
    payload_kind: str
    payload: Mapping[str, Any]

    def validate(self) -> "ExperimentResultEnvelope":
        if self.schema_name != CURRENT_SCHEMA_NAME:
            raise ResultValidationError(f"unsupported schema_name {self.schema_name!r}")
        try:
            major, minor, patch = (int(part) for part in self.schema_version.split("."))
        except (ValueError, TypeError):
            raise ResultValidationError("schema_version must be semantic x.y.z") from None
        if (major, minor, patch) != (1, 0, 0):
            raise ResultValidationError(f"unsupported schema_version {self.schema_version!r}")
        for name, value in (
            ("artifact_id", self.artifact_id),
            ("experiment_id", self.experiment_id),
            ("dataset_id", self.dataset_id),
            ("workload_id", self.workload_id),
        ):
            if not isinstance(value, str) or not _STABLE_ID.fullmatch(value):
                raise ResultValidationError(f"{name} is not a stable lowercase ID")
        try:
            parsed = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            raise ResultValidationError("created_at_utc must be ISO-8601") from None
        if parsed.tzinfo is None:
            raise ResultValidationError("created_at_utc must include a timezone")
        for name, value in (
            ("method_configuration", self.method_configuration),
            ("protocol", self.protocol),
            ("environment", self.environment),
            ("provenance", self.provenance),
            ("payload", self.payload),
        ):
            if not isinstance(value, Mapping):
                raise ResultValidationError(f"{name} must be an object")
        missing_provenance = _PROVENANCE_KEYS - set(self.provenance)
        if missing_provenance:
            raise ResultValidationError(
                f"provenance is missing {sorted(missing_provenance)}"
            )
        if not isinstance(self.provenance.get("git_dirty"), bool):
            raise ResultValidationError("provenance.git_dirty must be boolean")
        if not isinstance(self.provenance.get("input_artifact_ids"), list):
            raise ResultValidationError("provenance.input_artifact_ids must be a list")
        if self.payload_kind not in PAYLOAD_KINDS:
            raise ResultValidationError(f"unsupported payload_kind {self.payload_kind!r}")
        if self.payload.get("historical_migration"):
            if not isinstance(self.payload.get("source_data"), Mapping):
                raise ResultValidationError("historical payload must retain source_data")
        else:
            _PAYLOAD_VALIDATORS[self.payload_kind](self.payload)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentResultEnvelope":
        try:
            envelope = cls(**value)
        except TypeError as error:
            raise ResultValidationError(f"invalid result envelope fields: {error}") from error
        return envelope.validate()

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        experiment_id: str,
        dataset_id: str,
        workload_id: str,
        method_configuration: Mapping[str, Any],
        protocol: Mapping[str, Any],
        environment: Mapping[str, Any],
        provenance: Mapping[str, Any],
        payload_kind: str,
        payload: Mapping[str, Any],
        created_at_utc: str | None = None,
    ) -> "ExperimentResultEnvelope":
        return cls(
            schema_name=CURRENT_SCHEMA_NAME,
            schema_version=CURRENT_SCHEMA_VERSION,
            artifact_id=artifact_id,
            experiment_id=experiment_id,
            created_at_utc=created_at_utc or utc_now(),
            dataset_id=dataset_id,
            workload_id=workload_id,
            method_configuration=dict(method_configuration),
            protocol=dict(protocol),
            environment=dict(environment),
            provenance=dict(provenance),
            payload_kind=payload_kind,
            payload=dict(payload),
        ).validate()


def default_provenance(*, command: str, git_revision: str = "unknown") -> dict[str, Any]:
    """Create an explicit provenance block; unknown hashes remain visible."""
    return {
        "git_revision": git_revision,
        "git_dirty": False,
        "lockfile_sha256": None,
        "configuration_sha256": None,
        "data_sha256": None,
        "index_sha256": None,
        "native_binary_sha256": None,
        "command": command,
        "input_artifact_ids": [],
    }
