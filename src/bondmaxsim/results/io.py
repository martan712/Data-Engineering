"""Deterministic, atomic result serialization."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from bondmaxsim.results.models import ExperimentResultEnvelope

_COMPONENT = re.compile(r"[^a-z0-9._-]+")


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return the accepted deterministic UTF-8 JSON representation."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> Path:
    """Write canonical JSON through a sibling temporary file and atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_envelope(
    path: Path | str, envelope: ExperimentResultEnvelope
) -> Path:
    return atomic_write_json(path, envelope.validate().to_dict())


def _component(value: str) -> str:
    normalized = _COMPONENT.sub("-", value.lower()).strip("-._")
    if not normalized:
        raise ValueError("result filename component is empty")
    return normalized


def deterministic_result_name(
    experiment_id: str,
    dataset_id: str,
    workload_id: str,
    *,
    suffix: str = ".json",
) -> str:
    """Build a stable name without timestamps or mutable status words."""
    return "__".join(
        (_component(experiment_id), _component(dataset_id), _component(workload_id))
    ) + suffix
