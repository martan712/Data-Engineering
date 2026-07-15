from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from bondmaxsim.experiments.arms import ArmSpec, ExperimentSpec
from bondmaxsim.experiments.execution import execute_timing_experiment
from bondmaxsim.experiments.timing import (
    ScopeLedger,
    TimingError,
    TimingProtocol,
    classify_paired_sessions,
    cyclic_order,
    new_session_id,
    run_timing_session,
    summarize_values,
)
from bondmaxsim.results.models import default_provenance
from bondmaxsim.oracle.agreement import validate_boundary_tie_equivalence


class FakeClock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value

    def advance(self, nanoseconds):
        self.value += nanoseconds


def _arm(arm_id, clock, duration, *, validator=None, fail=False):
    def operation():
        clock.advance(duration)
        if fail:
            raise RuntimeError(f"{arm_id} failed")
        return arm_id

    return ArmSpec(
        arm_id=arm_id,
        method_family="fake",
        operation=operation,
        validator=validator,
        exactness="reference" if arm_id == "baseline" else "exact_safe",
        comparison_scope="uncapped_reference",
    )


def _environment(session_id="session-a"):
    return {
        "session": {"session_id": session_id},
        "threads": {"OMP_NUM_THREADS": "1"},
        "code": {"git_revision": "abc", "git_dirty": False, "uv_lock_sha256": "lock"},
        "toolchain": {"native_binary_sha256": {}},
    }


def test_protocol_presets_match_contract():
    assert TimingProtocol.preset("fixture") == TimingProtocol("fixture", 1, 3)
    assert TimingProtocol.preset("mechanism") == TimingProtocol("mechanism", 2, 9)
    assert TimingProtocol.preset("full_system") == TimingProtocol("full_system", 1, 5)
    assert TimingProtocol.preset("confirmation") == TimingProtocol("confirmation", 1, 9)


def test_fresh_session_ids_are_distinct_and_stable_format():
    first, second = new_session_id("fixture"), new_session_id("fixture")
    assert first != second
    assert first.startswith("fixture-") and first == first.lower()


def test_cyclic_order_is_deterministic_and_balanced():
    clock = FakeClock()
    arms = tuple(_arm(name, clock, 1) for name in ("a", "b", "c"))
    first = [tuple(arm.arm_id for arm in cyclic_order(arms, "stable-session", r)) for r in range(3)]
    second = [tuple(arm.arm_id for arm in cyclic_order(arms, "stable-session", r)) for r in range(3)]
    assert first == second
    assert len(set(first)) == 3
    positions = defaultdict(Counter)
    for order in first:
        for position, arm_id in enumerate(order):
            positions[arm_id][position] += 1
    assert all(counts == Counter({0: 1, 1: 1, 2: 1}) for counts in positions.values())


def test_session_retains_warmups_measurements_positions_and_summaries():
    clock = FakeClock()
    arms = (
        _arm("baseline", clock, 100),
        _arm("candidate", clock, 75),
    )
    session = run_timing_session(
        arms,
        TimingProtocol.preset("fixture"),
        session_id="session-a",
        environment=_environment(),
        clock_ns=clock,
        utc_timestamp=lambda: "2026-07-15T12:00:00Z",
    )
    assert session.complete
    assert len(session.observations) == (1 + 3) * 2
    assert Counter(row.phase for row in session.observations) == {"warmup": 2, "measured": 6}
    assert Counter(row.arm_id for row in session.measured()) == {"baseline": 3, "candidate": 3}
    summaries = session.summaries()
    assert summaries["baseline"]["values_ns"] == [100, 100, 100]
    assert summaries["candidate"]["median_ns"] == 75
    paired = session.paired_comparisons("baseline")["candidate"]
    assert paired["paired_ratios"] == [0.75, 0.75, 0.75]
    assert paired["median_paired_margin_pct"] == 25.0


def test_small_margin_requires_three_fresh_confirmation_sessions():
    sessions = []
    for index in range(3):
        clock = FakeClock()
        session = run_timing_session(
            (_arm("baseline", clock, 100), _arm("candidate", clock, 95)),
            TimingProtocol.preset("confirmation"),
            session_id=f"confirmation-{index}",
            environment=_environment(),
            clock_ns=clock,
        )
        sessions.append(session)
    assert classify_paired_sessions(
        sessions[:2], candidate_arm_id="candidate", baseline_arm_id="baseline"
    )["decision"] == "insufficient_confirmation"
    assert classify_paired_sessions(
        sessions, candidate_arm_id="candidate", baseline_arm_id="baseline"
    )["decision"] == "win"


def test_three_percent_equivalence_band_is_a_tie():
    clock = FakeClock()
    session = run_timing_session(
        (_arm("baseline", clock, 100), _arm("candidate", clock, 98)),
        TimingProtocol.preset("fixture"),
        session_id="tie-session",
        environment=_environment(),
        clock_ns=clock,
    )
    decision = classify_paired_sessions(
        [session], candidate_arm_id="candidate", baseline_arm_id="baseline"
    )
    assert decision["decision"] == "tie"


def test_summary_uses_linear_quartiles_and_keeps_all_values():
    summary = summarize_values([10, 20, 30, 100])
    assert summary["values_ns"] == [10, 20, 30, 100]
    assert summary["best_observed_ns"] == 10
    assert summary["median_ns"] == 25
    assert summary["q25_ns"] == pytest.approx(17.5)
    assert summary["q75_ns"] == pytest.approx(47.5)
    assert summary["iqr_ns"] == pytest.approx(30.0)


def test_validation_runs_outside_measured_scope():
    clock = FakeClock()
    checked = []

    def validator(result):
        ScopeLedger(clock).assert_outside_measured("correctness validation")
        clock.advance(1_000_000)
        checked.append(result)

    session = run_timing_session(
        (_arm("a", clock, 5, validator=validator),),
        TimingProtocol.preset("fixture"),
        session_id="scope-session",
        environment=_environment(),
        clock_ns=clock,
    )
    assert session.complete
    assert all(row.elapsed_ns == 5 for row in session.observations)
    assert len(checked) == 4
    assert all(event.scope_id == "result-validation" and not event.measured
               for event in session.scope_events if event.scope_id == "result-validation")


def test_repaired_agreement_result_is_recorded_and_blocks_exact_safe_failure():
    clock = FakeClock()

    def validator(_):
        return validate_boundary_tie_equivalence(
            [1, 2, 99], [1, 2, 3], [10.0, 9.0, 8.0],
            k=3, num_documents=100, exact_scores_by_id={99: 7.0},
        )

    session = run_timing_session(
        (_arm("candidate", clock, 5, validator=validator),),
        TimingProtocol.preset("fixture"),
        session_id="agreement-session",
        environment=_environment(),
        clock_ns=clock,
    )
    assert not session.complete
    assert session.validation_results[0]["result"]["boundary_tie_equivalent"] is False
    assert "correctness gate" in session.observations[-1].error


def test_excluded_scope_cannot_enter_measured_scope():
    clock = FakeClock()
    ledger = ScopeLedger(clock)
    with ledger.scope("scoring", "measured", measured=True):
        with pytest.raises(TimingError, match="inside measured scope"):
            with ledger.scope("serialization", "measured", measured=False):
                pass


def test_failure_retains_diagnostic_observation_and_blocks_summary():
    clock = FakeClock()
    session = run_timing_session(
        (_arm("a", clock, 10), _arm("broken", clock, 3, fail=True)),
        TimingProtocol.preset("fixture"),
        session_id="broken-session",
        environment=_environment(),
        clock_ns=clock,
    )
    assert not session.complete
    assert any(not row.completed and "RuntimeError" in row.error for row in session.observations)
    with pytest.raises(TimingError, match="incomplete"):
        session.summaries()
    payload = session.to_dict("a")
    assert payload["complete"] is False
    assert "summaries" not in payload


def test_declarative_execution_excludes_setup_and_serializes_raw_session():
    clock = FakeClock()
    setup_calls = []
    validation_calls = []
    arms = (
        _arm("baseline", clock, 100, validator=lambda result: validation_calls.append(result)),
        _arm("candidate", clock, 80, validator=lambda result: validation_calls.append(result)),
    )
    spec = ExperimentSpec(
        experiment_id="fixture-timing",
        artifact_id="fixture-timing-result",
        dataset_id="fixture",
        workload_id="fixture-workload-v1",
        arms=arms,
        command="pytest fixture timing",
        baseline_arm_id="baseline",
        workload_metadata={
            "workload_id": "fixture-workload-v1",
            "dataset": "fixture",
            "sample_size": 2,
        },
    )
    envelope, session = execute_timing_experiment(
        spec,
        TimingProtocol.preset("fixture"),
        session_id="execution-session",
        setup=lambda: (clock.advance(9_000_000), setup_calls.append(True)),
        environment=_environment("execution-session"),
        provenance=default_provenance(command=spec.command, git_revision="abc"),
        clock_ns=clock,
        utc_timestamp=lambda: "2026-07-15T12:00:00Z",
    )
    assert setup_calls == [True]
    assert len(validation_calls) == 8
    assert session.summaries()["baseline"]["median_ns"] == 100
    assert envelope.payload_kind == "timing_comparison"
    assert envelope.protocol["workload"]["sample_size"] == 2
    serialized_session = envelope.payload["sessions"][0]
    assert serialized_session["complete"] is True
    assert len(serialized_session["observations"]) == 8
    assert any(event["scope_id"] == "experiment-setup" for event in serialized_session["scope_events"])


def test_arm_and_experiment_specs_reject_ambiguous_configuration():
    clock = FakeClock()
    with pytest.raises(ValueError, match="stable lowercase"):
        _arm("Bad Arm", clock, 1)
    arm = _arm("a", clock, 1)
    with pytest.raises(ValueError, match="unique"):
        ExperimentSpec(
            experiment_id="fixture",
            artifact_id="fixture-result",
            dataset_id="fixture",
            workload_id="fixture-workload",
            arms=(arm, arm),
            command="pytest",
        )
    with pytest.raises(ValueError, match="workload metadata ID"):
        ExperimentSpec(
            experiment_id="fixture",
            artifact_id="fixture-result",
            dataset_id="fixture",
            workload_id="fixture-workload",
            arms=(arm,),
            command="pytest",
            workload_metadata={
                "workload_id": "another-workload",
                "dataset": "fixture",
            },
        )
