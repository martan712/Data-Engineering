from __future__ import annotations

from collections import Counter

import numpy as np

from bondmaxsim.experiments import (
    ArmSpec,
    CandidateWorkObservation,
    CountObservation,
    ExperimentSpec,
    TimingProtocol,
    WorkloadMetadata,
    execute_timing_experiment,
)
from bondmaxsim.oracle.agreement import validate_boundary_tie_equivalence
from bondmaxsim.results import (
    atomic_write_envelope,
    deterministic_result_name,
    load_result,
)
from bondmaxsim.results.models import default_provenance


class _Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        return self.value

    def advance(self, value: int) -> None:
        self.value += value


def _candidate_work(count: int) -> CandidateWorkObservation:
    unavailable = CountObservation.unavailable("fixture has no partition probes")
    return CandidateWorkObservation(
        configured_candidate_cap=CountObservation.exact(count, "fixture cap"),
        configured_full_score_cap=CountObservation.exact(count, "fixture cap"),
        unique_candidates_generated=CountObservation.exact(count, "fixture IDs"),
        documents_admitted_to_scoring=CountObservation.exact(count, "fixture IDs"),
        documents_fully_scored=CountObservation.exact(count, "fixture exact scorer"),
        documents_probed=unavailable,
        partitions_probed=unavailable,
        token_hits_inspected=unavailable,
    )


def test_synthetic_experiment_exercises_the_complete_stage3_flow(tmp_path):
    queries = [
        np.ones((2, 3), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32),
    ]
    workload = WorkloadMetadata.from_queries(
        dataset="fixture",
        regime="qrels_test",
        query_ids=["q1", "q2"],
        queries=queries,
        encoding_configuration={"model": "fixture", "normalization": "unit"},
        selection="all fixture queries in source order",
        source_revision="fixture-v1",
        source_split="test",
    )
    oracle_ids = np.array([0, 1], dtype=np.int64)
    oracle_scores = np.array([2.0, 1.0], dtype=np.float32)
    exact_scores = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    work = [_candidate_work(2), _candidate_work(2)]
    clock = _Clock()

    def arm(arm_id: str, duration: int, exactness: str) -> ArmSpec:
        def operation():
            clock.advance(duration)
            return {
                "ids": oracle_ids.copy(),
                "scores": oracle_scores.copy(),
                "candidate_work": work,
            }

        def validator(result):
            return validate_boundary_tie_equivalence(
                result["ids"],
                oracle_ids,
                oracle_scores,
                k=2,
                num_documents=3,
                exact_scores_by_id=exact_scores,
                returned_scores=result["scores"],
            )

        return ArmSpec(
            arm_id=arm_id,
            method_family="fixture",
            operation=operation,
            exactness=exactness,
            comparison_scope="equal_work_reranking",
            validator=validator,
            metadata_extractor=lambda result: {
                "candidate_work": [row.to_dict() for row in result["candidate_work"]]
            },
        )

    arms = (
        arm("baseline", 100, "reference"),
        arm("candidate", 80, "exact_safe"),
    )
    spec = ExperimentSpec(
        experiment_id="fixture-stage3-flow",
        artifact_id="fixture-stage3-flow-result",
        dataset_id="fixture",
        workload_id=workload.workload_id,
        arms=arms,
        command="pytest stage3 flow",
        baseline_arm_id="baseline",
        workload_metadata=workload.to_dict(),
    )
    environment = {
        "session": {"session_id": "fixture-stage3-session"},
        "threads": {"OMP_NUM_THREADS": "1"},
        "code": {
            "git_revision": "fixture-revision",
            "git_dirty": False,
            "uv_lock_sha256": "fixture-lock",
        },
        "toolchain": {"native_binary_sha256": {"fixture": "fixture-native"}},
    }
    envelope, session = execute_timing_experiment(
        spec,
        TimingProtocol.preset("fixture"),
        session_id="fixture-stage3-session",
        environment=environment,
        provenance=default_provenance(
            command=spec.command, git_revision="fixture-revision"
        ),
        clock_ns=clock,
        utc_timestamp=lambda: "2026-07-15T12:00:00Z",
    )

    assert session.complete
    assert len(session.observations) == 8
    assert len(session.validation_results) == 8
    assert len(session.result_metadata) == 8
    positions = Counter(
        (row.arm_id, row.arm_position) for row in session.measured()
    )
    assert {positions[("baseline", 0)], positions[("baseline", 1)]} == {1, 2}
    assert positions[("baseline", 0)] == positions[("candidate", 1)]
    assert positions[("baseline", 1)] == positions[("candidate", 0)]
    assert envelope.protocol["workload"]["token_counts"] == [2, 3]
    serialized_work = envelope.payload["sessions"][0]["result_metadata"][0]
    assert serialized_work["candidate_work"][0]["documents_fully_scored"]["value"] == 2

    output = tmp_path / deterministic_result_name(
        spec.experiment_id, spec.dataset_id, spec.workload_id
    )
    atomic_write_envelope(output, envelope)
    assert load_result(output) == envelope
