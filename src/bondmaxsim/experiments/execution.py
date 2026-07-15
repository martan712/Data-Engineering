"""Execute declarative timing experiments into validated result envelopes."""

from __future__ import annotations

from typing import Any, Callable

from bondmaxsim.experiments.arms import ExperimentSpec
from bondmaxsim.experiments.environment import capture_session_environment
from bondmaxsim.experiments.timing import ScopeLedger, TimingProtocol, TimingSession, run_timing_session
from bondmaxsim.results.models import ExperimentResultEnvelope, default_provenance


def execute_timing_experiment(
    spec: ExperimentSpec,
    protocol: TimingProtocol,
    *,
    session_id: str,
    build_profile: str | None = None,
    machine_quiescent: bool | None = None,
    setup: Callable[[], Any] | None = None,
    environment: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    clock_ns=None,
    utc_timestamp=None,
) -> tuple[ExperimentResultEnvelope, TimingSession]:
    """Run setup outside timing, execute arms, then build a versioned envelope."""
    ledger = ScopeLedger(clock_ns) if clock_ns is not None else ScopeLedger()
    if setup is not None:
        with ledger.scope("experiment-setup", "setup", measured=False):
            setup()
    snapshot = environment or capture_session_environment(
        session_id,
        build_profile=build_profile,
        machine_quiescent=machine_quiescent,
    )
    kwargs = {
        "session_id": session_id,
        "environment": snapshot,
        "ledger": ledger,
    }
    if clock_ns is not None:
        kwargs["clock_ns"] = clock_ns
    if utc_timestamp is not None:
        kwargs["utc_timestamp"] = utc_timestamp
    session = run_timing_session(spec.arms, protocol, **kwargs)

    provenance_block = provenance or default_provenance(
        command=spec.command,
        git_revision=snapshot.get("code", {}).get("git_revision") or "unknown",
    )
    provenance_block = dict(provenance_block)
    provenance_block["git_dirty"] = bool(snapshot.get("code", {}).get("git_dirty"))
    provenance_block["lockfile_sha256"] = snapshot.get("code", {}).get("uv_lock_sha256")
    provenance_block["native_binary_sha256"] = snapshot.get("toolchain", {}).get(
        "native_binary_sha256"
    )
    protocol_block = {
        **protocol.to_dict(),
        "thread_mode": snapshot.get("threads", {}),
    }
    envelope = ExperimentResultEnvelope.create(
        artifact_id=spec.artifact_id,
        experiment_id=spec.experiment_id,
        dataset_id=spec.dataset_id,
        workload_id=spec.workload_id,
        method_configuration=spec.method_configuration(),
        protocol=protocol_block,
        environment=snapshot,
        provenance=provenance_block,
        payload_kind="timing_comparison",
        payload={
            "sessions": [session.to_dict(spec.baseline_arm_id)],
        },
    )
    return envelope, session
