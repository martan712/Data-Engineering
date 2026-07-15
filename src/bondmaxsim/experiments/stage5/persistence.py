"""Stage 5 session-preserving timing artifact persistence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from bondmaxsim.results import load_result
from bondmaxsim.results.io import atomic_write_envelope
from bondmaxsim.results.models import ExperimentResultEnvelope, ResultValidationError
from bondmaxsim.results.validation import merge_incompatibilities


_FROZEN_PROVENANCE = (
    "git_revision",
    "lockfile_sha256",
    "configuration_sha256",
    "data_sha256",
    "index_sha256",
    "native_binary_sha256",
)


def write_or_append_timing_sessions(
    path: Path,
    envelope: ExperimentResultEnvelope,
) -> ExperimentResultEnvelope:
    """Append compatible fresh sessions and never discard prior observations."""
    if not path.exists():
        atomic_write_envelope(path, envelope)
        return envelope
    previous = load_result(path)
    mismatches = merge_incompatibilities(previous, envelope)
    mismatches.extend(
        f"provenance.{name}"
        for name in _FROZEN_PROVENANCE
        if previous.provenance.get(name) != envelope.provenance.get(name)
    )
    previous_extras = {key: value for key, value in previous.payload.items() if key != "sessions"}
    current_extras = {key: value for key, value in envelope.payload.items() if key != "sessions"}
    if previous_extras != current_extras:
        mismatches.append("payload_non_timing_fields")
    if mismatches:
        raise ResultValidationError(
            f"refusing to append incompatible Stage 5 sessions: {sorted(set(mismatches))}"
        )
    sessions = [*previous.payload["sessions"], *envelope.payload["sessions"]]
    session_ids = [session.get("session_id") for session in sessions]
    if len(set(session_ids)) != len(session_ids):
        raise ResultValidationError("refusing duplicate Stage 5 timing session IDs")
    merged = replace(
        envelope,
        payload={**current_extras, "sessions": sessions},
    ).validate()
    atomic_write_envelope(path, merged)
    return merged

