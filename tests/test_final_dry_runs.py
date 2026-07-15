from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from bondmaxsim.experiments.candidate_work import (
    CandidateWorkObservation,
    CountObservation,
)
from bondmaxsim.experiments.timing import session_offset
from bondmaxsim.release.dry_runs import (
    DEFAULT_DRY_RUNS,
    UNUSABLE_FINAL_DRIVER_CLIS,
    CandidateCountExpectation,
    DryRunDeclaration,
    DryRunError,
    ExpectedDryRunOutput,
    ProcessOutcome,
    execute_dry_runs,
)
from bondmaxsim.results.io import atomic_write_envelope, atomic_write_json
from bondmaxsim.results.models import ExperimentResultEnvelope, default_provenance


def _environment(session_id: str) -> dict:
    return {
        "session": {"session_id": session_id},
        "code": {"git_revision": "fixture", "git_dirty": False},
        "threads": {"OMP_NUM_THREADS": "1"},
        "toolchain": {"native_binary_sha256": {"fused": "fixture"}},
    }


def _provenance() -> dict:
    value = default_provenance(command="fixture --fixture")
    value["configuration_sha256"] = "a" * 64
    return value


def _candidate_work(*, quality: str = "exact") -> dict:
    unavailable = CountObservation.unavailable("fixture not applicable")
    observed = (
        CountObservation.exact(2, "fixture partition count")
        if quality == "exact"
        else unavailable
    )
    return CandidateWorkObservation(
        configured_candidate_cap=unavailable,
        configured_full_score_cap=unavailable,
        unique_candidates_generated=unavailable,
        documents_admitted_to_scoring=unavailable,
        documents_fully_scored=unavailable,
        documents_probed=observed,
        partitions_probed=observed,
        token_hits_inspected=unavailable,
    ).to_dict()


def _timing_envelope(
    experiment_id: str,
    session_id: str,
    *,
    candidate_quality: str = "exact",
) -> ExperimentResultEnvelope:
    arm_ids = ["dense", "partition"]
    initial = session_offset(session_id, len(arm_ids))
    observations = []
    scope_events = [
        {
            "scope_id": "experiment-setup",
            "phase": "setup",
            "started_ns": 0,
            "ended_ns": 1,
            "measured": False,
        }
    ]
    tick = 2
    for phase, rounds in (("warmup", 1), ("measured", 2)):
        for round_index in range(rounds):
            start = (initial + round_index) % len(arm_ids)
            order = arm_ids[start:] + arm_ids[:start]
            for position, arm_id in enumerate(order):
                scope = "native-pass" if arm_id == "dense" else "partition-pass"
                observations.append(
                    {
                        "session_id": session_id,
                        "round_index": round_index,
                        "phase": phase,
                        "arm_id": arm_id,
                        "arm_position": position,
                        "started_at_utc": "2026-07-15T12:00:00Z",
                        "elapsed_ns": 10,
                        "timer_scope_id": scope,
                        "completed": True,
                        "error": None,
                    }
                )
                scope_events.append(
                    {
                        "scope_id": scope,
                        "phase": phase,
                        "started_ns": tick,
                        "ended_ns": tick + 1,
                        "measured": True,
                    }
                )
                tick += 2
    environment = _environment(session_id)
    session = {
        "session_id": session_id,
        "protocol": {
            "protocol_class": "fixture",
            "warmup_rounds": 1,
            "measured_rounds": 2,
        },
        "arm_ids": arm_ids,
        "complete": True,
        "observations": observations,
        "scope_events": scope_events,
        "validation_results": [],
        "result_metadata": [
            {
                "phase": "measured",
                "round_index": 0,
                "arm_id": "partition",
                "candidate_work": [_candidate_work(quality=candidate_quality)],
            }
        ],
        "environment": environment,
        "summaries": {},
        "paired_comparisons": {},
    }
    return ExperimentResultEnvelope.create(
        artifact_id=f"{experiment_id}-fixture",
        experiment_id=experiment_id,
        dataset_id="synthetic-small-v1",
        workload_id="synthetic-small-v1",
        method_configuration={
            "arms": [
                {
                    "arm_id": "dense",
                    "comparison_scope": "uncapped_reference",
                },
                {
                    "arm_id": "partition",
                    "comparison_scope": "probe_frontier",
                },
            ]
        },
        protocol={"protocol_class": "fixture"},
        environment=environment,
        provenance=_provenance(),
        payload_kind="timing_comparison",
        payload={"sessions": [session]},
        created_at_utc="2026-07-15T12:00:00Z",
    )


def _audit_envelope(experiment_id: str, session_id: str) -> ExperimentResultEnvelope:
    return ExperimentResultEnvelope.create(
        artifact_id=f"{experiment_id}-fixture",
        experiment_id=experiment_id,
        dataset_id="synthetic-small-v1",
        workload_id="synthetic-small-v1",
        method_configuration={"fixture": True},
        protocol={"timing_evidence": False},
        environment=_environment(session_id),
        provenance=_provenance(),
        payload_kind="validation_audit",
        payload={"checks": [{"check_id": "fixture", "status": "pass"}], "passed": True},
        created_at_utc="2026-07-15T12:00:00Z",
    )


_CANDIDATE_EXPECTATION = CandidateCountExpectation(
    "probe_frontier", "documents_probed", ("exact",)
)


def _timing_declaration(run_id: str = "timing") -> DryRunDeclaration:
    return DryRunDeclaration(
        run_id,
        (
            "fake",
            "--fixture",
            "--session-id",
            "{session_id}",
            "--output-dir",
            "{output_dir}",
        ),
        (
            ExpectedDryRunOutput(
                "fixture-timing",
                "timing_comparison",
                True,
                (_CANDIDATE_EXPECTATION,),
            ),
        ),
    )


def _output_dir(command: tuple[str, ...]) -> Path:
    return Path(command[command.index("--output-dir") + 1])


def _session_id(command: tuple[str, ...]) -> str:
    return command[command.index("--session-id") + 1]


def test_runner_executes_serially_resolves_dependency_and_validates_exact_outputs(
    tmp_path: Path,
):
    declarations = (
        _timing_declaration("timing"),
        DryRunDeclaration(
            "analysis",
            (
                "fake-analysis",
                "--input",
                "{input}",
                "--output-dir",
                "{output_dir}",
            ),
            (ExpectedDryRunOutput("fixture-audit", "validation_audit"),),
            mode="derived_fixture",
            input_from="timing",
        ),
    )
    calls = []

    def runner(command, cwd):
        assert cwd == tmp_path
        calls.append(command)
        output = _output_dir(command)
        if command[0] == "fake":
            atomic_write_envelope(
                output / "timing.json",
                _timing_envelope("fixture-timing", _session_id(command)),
            )
        else:
            dependency = Path(command[command.index("--input") + 1])
            assert dependency == tmp_path / "outputs/timing/timing.json"
            atomic_write_envelope(
                output / "audit.json",
                _audit_envelope("fixture-audit", "analysis-session"),
            )
        return ProcessOutcome(0)

    suite = execute_dry_runs(
        declarations,
        output_root=tmp_path / "outputs",
        runner=runner,
        cwd=tmp_path,
    )
    assert [row.run_id for row in suite.runs] == ["timing", "analysis"]
    assert len(calls) == 2
    assert not list((tmp_path / "outputs").rglob("*.tmp"))


def _mutating_runner(mutator):
    def runner(command, cwd):
        del cwd
        envelope = _timing_envelope("fixture-timing", _session_id(command))
        document = deepcopy(envelope.to_dict())
        mutator(document, _output_dir(command))
        atomic_write_json(_output_dir(command) / "result.json", document)
        return ProcessOutcome(0)

    return runner


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda value, _: value["payload"]["sessions"][0]["observations"].pop(0),
            "raw timing repetitions",
        ),
        (
            lambda value, _: value["payload"]["sessions"][0]["observations"][0].update(
                arm_position=1
            ),
            "arm order|arm positions",
        ),
        (
            lambda value, _: value["payload"]["sessions"][0]["scope_events"][0].update(
                measured=True
            ),
            "setup is not excluded",
        ),
        (
            lambda value, _: value["payload"]["sessions"][0]["result_metadata"][0][
                "candidate_work"
            ][0]["documents_probed"].update(quality="unavailable", value=None),
            "observability mismatch",
        ),
    ],
)
def test_validator_rejects_incomplete_execution_evidence(tmp_path, mutator, match):
    with pytest.raises(DryRunError, match=match):
        execute_dry_runs(
            (_timing_declaration(),),
            output_root=tmp_path / "outputs",
            runner=_mutating_runner(mutator),
        )


def test_runner_rejects_failed_process_before_accepting_outputs(tmp_path: Path):
    with pytest.raises(DryRunError, match="command failed with 7"):
        execute_dry_runs(
            (_timing_declaration(),),
            output_root=tmp_path / "outputs",
            runner=lambda *_: ProcessOutcome(7, stderr="fixture failure"),
        )


@pytest.mark.parametrize("artifact", ["extra.json", ".result.json.123.tmp", "notes.txt"])
def test_runner_rejects_extra_or_non_atomic_outputs(tmp_path: Path, artifact: str):
    def runner(command, cwd):
        del cwd
        output = _output_dir(command)
        atomic_write_envelope(
            output / "result.json",
            _timing_envelope("fixture-timing", _session_id(command)),
        )
        (output / artifact).write_text("{}", encoding="utf-8")
        return ProcessOutcome(0)

    with pytest.raises(DryRunError, match="temporary|non-JSON|expected 1 outputs"):
        execute_dry_runs(
            (_timing_declaration(),),
            output_root=tmp_path / "outputs",
            runner=runner,
        )


def test_non_fixture_one_query_requires_data_hash(tmp_path: Path):
    declaration = DryRunDeclaration(
        "one-query",
        (
            "fake",
            "--query-count",
            "1",
            "--session-id",
            "{session_id}",
            "--output-dir",
            "{output_dir}",
        ),
        (ExpectedDryRunOutput("fixture-timing", "timing_comparison", True),),
        mode="one_query",
    )

    def runner(command, cwd):
        del cwd
        atomic_write_envelope(
            _output_dir(command) / "result.json",
            _timing_envelope("fixture-timing", _session_id(command)),
        )
        return ProcessOutcome(0)

    with pytest.raises(DryRunError, match="non-fixture dry run has no data hash"):
        execute_dry_runs(
            (declaration,), output_root=tmp_path / "outputs", runner=runner
        )


def test_output_root_must_be_empty(tmp_path: Path):
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "old.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DryRunError, match="must be empty"):
        execute_dry_runs(
            (_timing_declaration(),),
            output_root=output,
            runner=lambda *_: pytest.fail("dirty root executed a process"),
        )


def test_default_release_matrix_is_unique_and_records_unusable_legacy_drivers():
    assert len(DEFAULT_DRY_RUNS) == 13
    assert len({row.run_id for row in DEFAULT_DRY_RUNS}) == len(DEFAULT_DRY_RUNS)
    assert all(row.outputs for row in DEFAULT_DRY_RUNS)
    assert "experiments.stage2_testbed.s03_two_mode_smoke" in UNUSABLE_FINAL_DRIVER_CLIS
    assert {
        "experiments.stage3_mechanism.e04_exact_safe_pruning",
        "experiments.stage3_mechanism.e05_approximate_recall_sweep",
        "experiments.stage3_mechanism.e06_threshold_policy_ablation",
        "experiments.stage3_mechanism.e07_cache_layout_sensitivity",
        "experiments.stage3_mechanism.e09_bound_tightness_ablation",
    }.issubset(UNUSABLE_FINAL_DRIVER_CLIS)
