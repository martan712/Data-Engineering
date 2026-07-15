"""Artifact catalog and paper-evidence relationship validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from bondmaxsim.results.models import ResultValidationError

CATALOG_STATUSES = {"current", "superseded", "historical", "diagnostic", "invalid"}
_POINTER = re.compile(r"^(?:/(?:[^~/]|~0|~1)*)*$")


def _read_json_subset(path: Path | str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultValidationError(f"cannot read catalog {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ResultValidationError("catalog root must be an object")
    return value


def validate_catalog(path: Path | str, *, workspace: Path | None = None) -> Mapping[str, Any]:
    catalog = _read_json_subset(path)
    entries = catalog.get("artifacts")
    if not isinstance(entries, list):
        raise ResultValidationError("catalog.artifacts must be a list")
    by_id = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("artifact_id"), str):
            raise ResultValidationError("catalog entries require artifact_id")
        artifact_id = entry["artifact_id"]
        if artifact_id in by_id:
            raise ResultValidationError(f"duplicate artifact_id {artifact_id}")
        if entry.get("status") not in CATALOG_STATUSES:
            raise ResultValidationError(f"invalid catalog status for {artifact_id}")
        if workspace is not None and entry.get("path") and not (workspace / entry["path"]).exists():
            raise ResultValidationError(f"catalog path does not exist: {entry['path']}")
        by_id[artifact_id] = entry

    graph = {artifact_id: [] for artifact_id in by_id}
    for artifact_id, entry in by_id.items():
        supersession = entry.get("supersession")
        if supersession:
            successor = supersession.get("successor")
            if successor not in by_id:
                raise ResultValidationError(f"unknown successor {successor}")
            scope = supersession.get("scope")
            pointers = [scope] if isinstance(scope, str) else scope
            if not isinstance(pointers, list) or not pointers or not all(
                isinstance(pointer, str) and _POINTER.fullmatch(pointer)
                for pointer in pointers
            ):
                raise ResultValidationError(f"invalid supersession scope for {artifact_id}")
            graph[artifact_id].append(successor)

    visiting, visited = set(), set()
    def visit(node: str) -> None:
        if node in visiting:
            raise ResultValidationError("catalog supersession graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for successor in graph[node]:
            visit(successor)
        visiting.remove(node)
        visited.add(node)
    for artifact_id in graph:
        visit(artifact_id)
    return catalog


def validate_evidence_manifest(
    path: Path | str, catalog: Mapping[str, Any]
) -> Mapping[str, Any]:
    manifest = _read_json_subset(path)
    evidence = manifest.get("evidence")
    if not isinstance(evidence, list):
        raise ResultValidationError("manifest.evidence must be a list")
    artifacts = {entry["artifact_id"]: entry for entry in catalog.get("artifacts", [])}
    seen = set()
    for entry in evidence:
        evidence_id = entry.get("evidence_id") if isinstance(entry, Mapping) else None
        if not isinstance(evidence_id, str) or evidence_id in seen:
            raise ResultValidationError("evidence IDs must be unique strings")
        seen.add(evidence_id)
        references = entry.get("references")
        if not isinstance(references, list) or not references:
            raise ResultValidationError(f"evidence {evidence_id} needs references")
        for reference in references:
            artifact = artifacts.get(reference.get("artifact_id"))
            pointer = reference.get("pointer")
            if artifact is None:
                raise ResultValidationError(f"evidence {evidence_id} references unknown artifact")
            if not isinstance(pointer, str) or not _POINTER.fullmatch(pointer):
                raise ResultValidationError(f"evidence {evidence_id} has invalid JSON pointer")
            if artifact.get("status") in {"invalid", "superseded"} and entry.get("status") == "current":
                raise ResultValidationError(
                    f"current evidence {evidence_id} references {artifact['status']} evidence"
                )
    return manifest
