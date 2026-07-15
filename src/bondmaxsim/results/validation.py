"""Envelope discovery and compatibility-safe repeated-session merging."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from bondmaxsim.results.migrations import load_result
from bondmaxsim.results.models import ExperimentResultEnvelope, ResultValidationError, utc_now


def discover_results(paths: Iterable[Path]) -> list[ExperimentResultEnvelope]:
    envelopes = []
    for root in paths:
        for path in sorted(root.rglob("*.json")):
            envelopes.append(load_result(path))
    return envelopes


def merge_incompatibilities(
    left: ExperimentResultEnvelope, right: ExperimentResultEnvelope
) -> list[str]:
    mismatches = []
    if left.schema_version.split(".")[0] != right.schema_version.split(".")[0]:
        mismatches.append("schema_major")
    for name in ("workload_id", "dataset_id", "payload_kind"):
        if getattr(left, name) != getattr(right, name):
            mismatches.append(name)
    if left.method_configuration != right.method_configuration:
        mismatches.append("method_configuration")
    if left.protocol != right.protocol:
        mismatches.append("protocol")
    for name in ("data_sha256", "index_sha256"):
        if left.provenance.get(name) != right.provenance.get(name):
            mismatches.append(name)
    if left.protocol.get("thread_mode") != right.protocol.get("thread_mode"):
        mismatches.append("thread_mode")
    return sorted(set(mismatches))


def merge_timing_sessions(
    left: ExperimentResultEnvelope,
    right: ExperimentResultEnvelope,
    *,
    artifact_id: str,
    allow_incompatible: bool = False,
) -> ExperimentResultEnvelope:
    """Merge raw timing sessions, refusing silent cross-protocol coercion."""
    if left.payload_kind != "timing_comparison" or right.payload_kind != "timing_comparison":
        raise ResultValidationError("only timing_comparison envelopes can be merged")
    mismatches = merge_incompatibilities(left, right)
    if mismatches and not allow_incompatible:
        raise ResultValidationError(f"incompatible result merge: {mismatches}")
    sessions = list(left.payload.get("sessions", [])) + list(right.payload.get("sessions", []))
    provenance = dict(left.provenance)
    provenance["input_artifact_ids"] = [left.artifact_id, right.artifact_id]
    if mismatches:
        provenance["merge_overrides"] = mismatches
    return replace(
        left,
        artifact_id=artifact_id,
        created_at_utc=utc_now(),
        provenance=provenance,
        payload={"sessions": sessions},
    ).validate()
