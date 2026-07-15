"""Fail-closed validation of result provenance against frozen inputs."""

from __future__ import annotations

import re
from typing import Any, Mapping

from bondmaxsim.results.models import ExperimentResultEnvelope, ResultValidationError


FROZEN_INPUT_HASH_FIELDS = (
    "configuration_sha256",
    "data_sha256",
    "index_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_expected_hashes(
    expected: Mapping[str, Any],
) -> dict[str, str | None]:
    if not isinstance(expected, Mapping):
        raise ResultValidationError("frozen input hashes must be an object")
    expected_fields = set(expected)
    required_fields = set(FROZEN_INPUT_HASH_FIELDS)
    if expected_fields != required_fields:
        missing = sorted(required_fields - expected_fields)
        extra = sorted(expected_fields - required_fields)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ResultValidationError(
            "frozen input hashes must contain exactly configuration, data, and index "
            f"hashes ({'; '.join(details)})"
        )

    normalized: dict[str, str | None] = {}
    for field in FROZEN_INPUT_HASH_FIELDS:
        value = expected[field]
        if value is None and field == "index_sha256":
            normalized[field] = None
            continue
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ResultValidationError(
                f"frozen {field} must be a lowercase SHA-256 hex digest"
            )
        normalized[field] = value
    return normalized


def validate_frozen_inputs(
    envelope: ExperimentResultEnvelope,
    expected: Mapping[str, Any],
) -> ExperimentResultEnvelope:
    """Refuse a result whose data/configuration/index provenance is not frozen."""
    frozen = _validate_expected_hashes(expected)
    if envelope.payload.get("historical_migration"):
        raise ResultValidationError(
            "historical result cannot satisfy frozen input hashes; rerun it under the "
            "versioned result envelope"
        )

    mismatches = []
    for field, expected_value in frozen.items():
        actual_value = envelope.provenance.get(field)
        if actual_value != expected_value:
            mismatches.append(field)
    if mismatches:
        raise ResultValidationError(
            "result provenance does not match frozen inputs: "
            + ", ".join(sorted(mismatches))
        )
    return envelope
