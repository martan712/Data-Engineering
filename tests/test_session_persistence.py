from __future__ import annotations

from pathlib import Path

import pytest

from bondmaxsim.experiments.persistence import write_or_append_timing_sessions
from bondmaxsim.experiments.stage4.partition_frontier import (
    PartitionFrontierConfig,
    run_partition_frontier,
)
from bondmaxsim.results import load_result
from bondmaxsim.results.models import (
    ExperimentResultEnvelope,
    ResultValidationError,
    default_provenance,
)


def _environment(session_id: str) -> dict:
    return {
        "session": {"session_id": session_id},
        "threads": {"OMP_NUM_THREADS": "1"},
        "code": {
            "git_revision": "fixture-revision",
            "git_dirty": False,
            "uv_lock_sha256": "fixture-lock",
        },
        "toolchain": {"native_binary_sha256": {"fused": "fixture-native"}},
    }


def _timing_envelope(session_id: str, *, dataset_id: str = "synthetic-small-v1"):
    session = {
        "session_id": session_id,
        "observations": [{"elapsed_ns": 1, "phase": "measured"}],
    }
    return ExperimentResultEnvelope.create(
        artifact_id="fixture-timing",
        experiment_id="fixture-timing",
        dataset_id=dataset_id,
        workload_id=dataset_id,
        method_configuration={"arms": []},
        protocol={"timing_evidence": True},
        environment=_environment(session_id),
        provenance=default_provenance(command="fixture --fixture"),
        payload_kind="timing_comparison",
        payload={"sessions": [session]},
    )


def test_repeated_stage4_sessions_accumulate_into_one_artifact(tmp_path: Path):
    first = run_partition_frontier(
        PartitionFrontierConfig.fixture_config(),
        output_dir=tmp_path,
        session_id="fixture-e03-a",
        environment=_environment("fixture-e03-a"),
    )
    assert len(first.envelope.payload["sessions"]) == 1

    second = run_partition_frontier(
        PartitionFrontierConfig.fixture_config(),
        output_dir=tmp_path,
        session_id="fixture-e03-b",
        environment=_environment("fixture-e03-b"),
    )

    # Both invocations resolve to the same deterministic artifact.
    assert second.output_path == first.output_path
    saved = load_result(second.output_path)
    session_ids = [session["session_id"] for session in saved.payload["sessions"]]
    assert session_ids == ["fixture-e03-a", "fixture-e03-b"]


def test_repeated_session_id_is_refused(tmp_path: Path):
    run_partition_frontier(
        PartitionFrontierConfig.fixture_config(),
        output_dir=tmp_path,
        session_id="fixture-e03-dup",
        environment=_environment("fixture-e03-dup"),
    )
    with pytest.raises(ResultValidationError, match="duplicate timing session"):
        run_partition_frontier(
            PartitionFrontierConfig.fixture_config(),
            output_dir=tmp_path,
            session_id="fixture-e03-dup",
            environment=_environment("fixture-e03-dup"),
        )


def test_incompatible_sessions_are_never_merged(tmp_path: Path):
    path = tmp_path / "timing.json"
    write_or_append_timing_sessions(path, _timing_envelope("s1", dataset_id="scifact"))
    with pytest.raises(ResultValidationError, match="incompatible timing sessions"):
        write_or_append_timing_sessions(
            path, _timing_envelope("s2", dataset_id="nfcorpus")
        )
    # The prior artifact is left intact after the refused merge.
    preserved = load_result(path)
    assert [s["session_id"] for s in preserved.payload["sessions"]] == ["s1"]
