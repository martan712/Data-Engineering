"""Fail-closed, serial dry-run execution for final experiment CLIs."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.timing import session_offset
from bondmaxsim.results import load_result
from bondmaxsim.results.models import (
    CURRENT_SCHEMA_NAME,
    CURRENT_SCHEMA_VERSION,
    ExperimentResultEnvelope,
)


_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_SCOPES = {
    "equal_work_reranking",
    "system_cap",
    "probe_frontier",
}


class DryRunError(RuntimeError):
    """A final experiment dry run failed its execution or evidence contract."""


@dataclass(frozen=True)
class CandidateCountExpectation:
    comparison_scope: str
    field: str
    allowed_qualities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.comparison_scope not in _CANDIDATE_SCOPES:
            raise ValueError(f"invalid candidate scope {self.comparison_scope!r}")
        if not self.field or not self.allowed_qualities:
            raise ValueError("candidate expectation requires a field and qualities")


@dataclass(frozen=True)
class ExpectedDryRunOutput:
    experiment_id: str
    payload_kind: str
    timing: bool = False
    candidate_counts: tuple[CandidateCountExpectation, ...] = ()
    session_suffix: str = ""


@dataclass(frozen=True)
class DryRunDeclaration:
    run_id: str
    command: tuple[str, ...]
    outputs: tuple[ExpectedDryRunOutput, ...]
    mode: str = "fixture"
    input_from: str | None = None

    def __post_init__(self) -> None:
        if not _STABLE_ID.fullmatch(self.run_id):
            raise ValueError(f"invalid dry-run ID {self.run_id!r}")
        if self.mode not in {"fixture", "one_query", "derived_fixture"}:
            raise ValueError(f"invalid dry-run mode {self.mode!r}")
        if not self.command or "{output_dir}" not in self.command:
            raise ValueError("dry-run command must contain a standalone {output_dir}")
        if not self.outputs:
            raise ValueError("dry-run declaration must expect at least one output")
        if len({row.experiment_id for row in self.outputs}) != len(self.outputs):
            raise ValueError("expected experiment IDs must be unique within a run")
        if any(row.timing for row in self.outputs) and "{session_id}" not in self.command:
            raise ValueError("timing dry run must contain a standalone {session_id}")
        if self.mode == "fixture" and "--fixture" not in self.command:
            raise ValueError("fixture dry run must pass --fixture")
        if self.input_from is None and "{input}" in self.command:
            raise ValueError("{input} requires input_from")
        if self.input_from is not None and "{input}" not in self.command:
            raise ValueError("input_from requires a standalone {input}")


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class ProcessRunner(Protocol):
    def __call__(self, command: tuple[str, ...], cwd: Path) -> ProcessOutcome: ...


@dataclass(frozen=True)
class ValidatedDryRunOutput:
    path: Path
    envelope: ExperimentResultEnvelope


@dataclass(frozen=True)
class DryRunOutcome:
    run_id: str
    command: tuple[str, ...]
    outputs: tuple[ValidatedDryRunOutput, ...]


@dataclass(frozen=True)
class DryRunSuite:
    output_root: Path
    runs: tuple[DryRunOutcome, ...]


def _default_runner(command: tuple[str, ...], cwd: Path) -> ProcessOutcome:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return ProcessOutcome(completed.returncode, completed.stdout, completed.stderr)


def _expand_command(
    declaration: DryRunDeclaration,
    *,
    output_dir: Path,
    session_id: str,
    dependency: Path | None,
) -> tuple[str, ...]:
    replacements = {
        "{output_dir}": str(output_dir),
        "{session_id}": session_id,
    }
    if dependency is not None:
        replacements["{input}"] = str(dependency)
    expanded = tuple(replacements.get(value, value) for value in declaration.command)
    unresolved = [value for value in expanded if value.startswith("{") and value.endswith("}")]
    if unresolved:
        raise DryRunError(
            f"{declaration.run_id}: unresolved command placeholders {unresolved}"
        )
    return expanded


def _validate_environment_and_provenance(
    envelope: ExperimentResultEnvelope,
    *,
    declaration: DryRunDeclaration,
    expected_session_id: str | None,
) -> None:
    environment = envelope.environment
    session = environment.get("session")
    if not isinstance(session, Mapping) or not isinstance(session.get("session_id"), str):
        raise DryRunError(f"{declaration.run_id}: environment session identity is missing")
    if not session["session_id"]:
        raise DryRunError(f"{declaration.run_id}: environment session identity is empty")
    if expected_session_id is not None and session["session_id"] != expected_session_id:
        raise DryRunError(f"{declaration.run_id}: environment session ID mismatch")
    for section in ("code", "threads", "toolchain"):
        if not isinstance(environment.get(section), Mapping):
            raise DryRunError(f"{declaration.run_id}: environment.{section} is missing")

    provenance = envelope.provenance
    configuration = provenance.get("configuration_sha256")
    if not isinstance(configuration, str) or not _SHA256.fullmatch(configuration):
        raise DryRunError(f"{declaration.run_id}: experiment configuration hash is missing")
    data_hash = provenance.get("data_sha256")
    if data_hash is not None and (
        not isinstance(data_hash, str) or not _SHA256.fullmatch(data_hash)
    ):
        raise DryRunError(f"{declaration.run_id}: data provenance hash is malformed")
    if data_hash is None and declaration.mode not in {"fixture", "derived_fixture"}:
        raise DryRunError(f"{declaration.run_id}: non-fixture dry run has no data hash")
    index_hash = provenance.get("index_sha256")
    if index_hash is not None and (
        not isinstance(index_hash, str) or not _SHA256.fullmatch(index_hash)
    ):
        raise DryRunError(f"{declaration.run_id}: index provenance hash is malformed")
    if not isinstance(provenance.get("command"), str) or not provenance["command"]:
        raise DryRunError(f"{declaration.run_id}: producing command provenance is missing")


def _ordered_arms(observations: Sequence[Mapping[str, object]]) -> list[str]:
    try:
        return [
            str(row["arm_id"])
            for row in sorted(observations, key=lambda row: int(row["arm_position"]))
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise DryRunError(f"malformed arm-position metadata: {error}") from error


def _validate_timing(
    envelope: ExperimentResultEnvelope,
    *,
    declaration: DryRunDeclaration,
    expected_session_id: str,
) -> None:
    sessions = envelope.payload.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 1:
        raise DryRunError(f"{declaration.run_id}: dry run must retain exactly one session")
    session = sessions[0]
    if not isinstance(session, Mapping) or session.get("complete") is not True:
        raise DryRunError(f"{declaration.run_id}: timing session is incomplete")
    if session.get("session_id") != expected_session_id:
        raise DryRunError(f"{declaration.run_id}: payload session ID mismatch")
    if session.get("environment") != envelope.environment:
        raise DryRunError(f"{declaration.run_id}: session environment snapshot drifted")
    arm_ids = session.get("arm_ids")
    protocol = session.get("protocol")
    observations = session.get("observations")
    if (
        not isinstance(arm_ids, list)
        or not arm_ids
        or len(set(arm_ids)) != len(arm_ids)
        or not isinstance(protocol, Mapping)
        or not isinstance(observations, list)
    ):
        raise DryRunError(f"{declaration.run_id}: malformed timing session metadata")
    warmups = protocol.get("warmup_rounds")
    measured = protocol.get("measured_rounds")
    if (
        isinstance(warmups, bool)
        or not isinstance(warmups, int)
        or warmups < 0
        or isinstance(measured, bool)
        or not isinstance(measured, int)
        or measured <= 0
    ):
        raise DryRunError(f"{declaration.run_id}: invalid timing repetition contract")
    expected_count = len(arm_ids) * (warmups + measured)
    if len(observations) != expected_count:
        raise DryRunError(f"{declaration.run_id}: raw timing repetitions are incomplete")
    if any(
        not isinstance(row, Mapping)
        or row.get("completed") is not True
        or row.get("error") is not None
        or row.get("session_id") != expected_session_id
        for row in observations
    ):
        raise DryRunError(f"{declaration.run_id}: failed or malformed raw timing observation")

    original = [str(value) for value in arm_ids]
    initial = session_offset(expected_session_id, len(original))
    for phase, rounds in (("warmup", warmups), ("measured", measured)):
        for round_index in range(rounds):
            rows = [
                row
                for row in observations
                if row.get("phase") == phase and row.get("round_index") == round_index
            ]
            if len(rows) != len(original):
                raise DryRunError(f"{declaration.run_id}: incomplete {phase} round {round_index}")
            start = (initial + round_index) % len(original)
            expected_order = original[start:] + original[:start]
            if _ordered_arms(rows) != expected_order:
                raise DryRunError(f"{declaration.run_id}: arm order is not counterbalanced")
            if sorted(int(row["arm_position"]) for row in rows) != list(range(len(original))):
                raise DryRunError(f"{declaration.run_id}: arm positions are incomplete")

    scope_events = session.get("scope_events")
    if not isinstance(scope_events, list) or not scope_events:
        raise DryRunError(f"{declaration.run_id}: timer-boundary ledger is missing")
    if any(
        not isinstance(event, Mapping)
        or not isinstance(event.get("started_ns"), int)
        or not isinstance(event.get("ended_ns"), int)
        or event["ended_ns"] < event["started_ns"]
        for event in scope_events
    ):
        raise DryRunError(f"{declaration.run_id}: malformed timer-boundary event")
    setup = [event for event in scope_events if event.get("scope_id") == "experiment-setup"]
    if len(setup) != 1 or setup[0].get("measured") is not False:
        raise DryRunError(f"{declaration.run_id}: setup is not excluded from timing")
    measured_events = [event for event in scope_events if event.get("measured") is True]
    if len(measured_events) != len(observations):
        raise DryRunError(f"{declaration.run_id}: measured scope/observation count mismatch")
    observed_scopes = Counter(
        (str(row.get("phase")), str(row.get("timer_scope_id"))) for row in observations
    )
    ledger_scopes = Counter(
        (str(row.get("phase")), str(row.get("scope_id"))) for row in measured_events
    )
    if observed_scopes != ledger_scopes:
        raise DryRunError(f"{declaration.run_id}: timer scopes do not match observations")
    if any(
        event.get("measured") is not False
        for event in scope_events
        if event.get("scope_id") in {"result-validation", "result-metadata"}
    ):
        raise DryRunError(f"{declaration.run_id}: validation or metadata entered timing")


def _validate_candidate_counts(
    envelope: ExperimentResultEnvelope,
    expected: tuple[CandidateCountExpectation, ...],
    *,
    run_id: str,
) -> None:
    if not expected:
        return
    arms = envelope.method_configuration.get("arms")
    sessions = envelope.payload.get("sessions")
    if not isinstance(arms, list) or not isinstance(sessions, list) or len(sessions) != 1:
        raise DryRunError(f"{run_id}: candidate arm/session metadata is missing")
    metadata = sessions[0].get("result_metadata")
    if not isinstance(metadata, list):
        raise DryRunError(f"{run_id}: candidate result metadata is missing")
    for requirement in expected:
        candidate_arm_ids = {
            str(arm.get("arm_id"))
            for arm in arms
            if isinstance(arm, Mapping)
            and arm.get("comparison_scope") == requirement.comparison_scope
        }
        if not candidate_arm_ids:
            raise DryRunError(
                f"{run_id}: no arm declares candidate scope {requirement.comparison_scope}"
            )
        for arm_id in candidate_arm_ids:
            rows = [
                row
                for row in metadata
                if isinstance(row, Mapping) and row.get("arm_id") == arm_id
            ]
            if not rows:
                raise DryRunError(f"{run_id}: {arm_id} has no candidate metadata")
            for row in rows:
                work = row.get("candidate_work")
                if not isinstance(work, list) or not work:
                    raise DryRunError(f"{run_id}: {arm_id} has no per-query candidate counts")
                for query_work in work:
                    observation = (
                        query_work.get(requirement.field)
                        if isinstance(query_work, Mapping)
                        else None
                    )
                    if (
                        not isinstance(observation, Mapping)
                        or observation.get("quality") not in requirement.allowed_qualities
                    ):
                        raise DryRunError(
                            f"{run_id}: {arm_id}.{requirement.field} observability mismatch"
                        )


def validate_dry_run_output(
    path: Path,
    expected: ExpectedDryRunOutput,
    *,
    declaration: DryRunDeclaration,
    session_id: str,
) -> ValidatedDryRunOutput:
    envelope = load_result(path)
    if (
        envelope.schema_name != CURRENT_SCHEMA_NAME
        or envelope.schema_version != CURRENT_SCHEMA_VERSION
        or envelope.payload.get("historical_migration")
    ):
        raise DryRunError(f"{declaration.run_id}: output is not the current result schema")
    if envelope.experiment_id != expected.experiment_id:
        raise DryRunError(f"{declaration.run_id}: unexpected experiment ID")
    if envelope.payload_kind != expected.payload_kind:
        raise DryRunError(f"{declaration.run_id}: unexpected payload kind")
    expected_session = (
        f"{session_id}{expected.session_suffix}"
        if "{session_id}" in declaration.command
        else None
    )
    _validate_environment_and_provenance(
        envelope,
        declaration=declaration,
        expected_session_id=expected_session,
    )
    if expected.timing:
        _validate_timing(
            envelope,
            declaration=declaration,
            expected_session_id=str(expected_session),
        )
    _validate_candidate_counts(
        envelope,
        expected.candidate_counts,
        run_id=declaration.run_id,
    )
    return ValidatedDryRunOutput(path, envelope)


def execute_dry_runs(
    declarations: Sequence[DryRunDeclaration],
    *,
    output_root: Path,
    runner: ProcessRunner = _default_runner,
    cwd: Path = REPO_ROOT,
) -> DryRunSuite:
    """Execute declarations synchronously and validate every exact output set."""
    declared = tuple(declarations)
    if not declared:
        raise DryRunError("at least one dry run must be declared")
    run_ids = [row.run_id for row in declared]
    if len(set(run_ids)) != len(run_ids):
        raise DryRunError("dry-run IDs must be unique")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise DryRunError("dry-run output root must be empty")

    outcomes: list[DryRunOutcome] = []
    outputs_by_run: dict[str, tuple[ValidatedDryRunOutput, ...]] = {}
    for declaration in declared:
        dependency = None
        if declaration.input_from is not None:
            prior = outputs_by_run.get(declaration.input_from)
            if prior is None:
                raise DryRunError(
                    f"{declaration.run_id}: dependency {declaration.input_from!r} has not run"
                )
            if len(prior) != 1:
                raise DryRunError(
                    f"{declaration.run_id}: dependency must produce exactly one artifact"
                )
            dependency = prior[0].path
        run_dir = output_root / declaration.run_id
        run_dir.mkdir()
        session_id = f"dry-run-{declaration.run_id}"
        command = _expand_command(
            declaration,
            output_dir=run_dir,
            session_id=session_id,
            dependency=dependency,
        )
        process = runner(command, Path(cwd))
        if process.returncode != 0:
            raise DryRunError(
                f"{declaration.run_id}: command failed with {process.returncode}: "
                f"{process.stderr.strip()}"
            )
        temporary = [path for path in run_dir.rglob("*") if path.name.endswith(".tmp")]
        if temporary:
            raise DryRunError(f"{declaration.run_id}: atomic-output temporary files remain")
        files = sorted(path for path in run_dir.rglob("*") if path.is_file())
        if any(path.suffix != ".json" for path in files):
            raise DryRunError(f"{declaration.run_id}: undeclared non-JSON output exists")
        if len(files) != len(declaration.outputs):
            raise DryRunError(
                f"{declaration.run_id}: expected {len(declaration.outputs)} outputs, "
                f"found {len(files)}"
            )
        actual_by_experiment: dict[str, Path] = {}
        for path in files:
            envelope = load_result(path)
            if envelope.experiment_id in actual_by_experiment:
                raise DryRunError(
                    f"{declaration.run_id}: duplicate output for {envelope.experiment_id}"
                )
            actual_by_experiment[envelope.experiment_id] = path
        if set(actual_by_experiment) != {
            expected.experiment_id for expected in declaration.outputs
        }:
            raise DryRunError(f"{declaration.run_id}: declared output set does not match")
        validated = tuple(
            validate_dry_run_output(
                actual_by_experiment[expected.experiment_id],
                expected,
                declaration=declaration,
                session_id=session_id,
            )
            for expected in declaration.outputs
        )
        outputs_by_run[declaration.run_id] = validated
        outcomes.append(DryRunOutcome(declaration.run_id, command, validated))
    return DryRunSuite(output_root, tuple(outcomes))


def _command(module: str, *, session: bool = False) -> tuple[str, ...]:
    values = ["uv", "run", "--no-sync", "python", "-m", module, "--fixture"]
    if session:
        values.extend(("--session-id", "{session_id}"))
    values.extend(("--output-dir", "{output_dir}"))
    return tuple(values)


_FULLY_SCORED_EXACT = CandidateCountExpectation(
    "equal_work_reranking", "documents_fully_scored", ("exact",)
)
_FULL_SCORE_CAP_EXACT = CandidateCountExpectation(
    "system_cap", "configured_full_score_cap", ("exact",)
)
_FULLY_SCORED_UNAVAILABLE = CandidateCountExpectation(
    "system_cap", "documents_fully_scored", ("unavailable",)
)
_SYSTEM_FULLY_SCORED_EXACT = CandidateCountExpectation(
    "system_cap", "documents_fully_scored", ("exact",)
)
_DOCUMENTS_PROBED_EXACT = CandidateCountExpectation(
    "probe_frontier", "documents_probed", ("exact",)
)
_PARTITIONS_PROBED_EXACT = CandidateCountExpectation(
    "probe_frontier", "partitions_probed", ("exact",)
)


DEFAULT_DRY_RUNS: tuple[DryRunDeclaration, ...] = (
    DryRunDeclaration(
        "stage2-normalization",
        _command("experiments.stage2_testbed.s01_normalization_guard"),
        (ExpectedDryRunOutput("stage2-s01-normalization-audit", "validation_audit"),),
    ),
    DryRunDeclaration(
        "stage2-exact-agreement",
        _command("experiments.stage2_testbed.s02_exact_agreement"),
        (ExpectedDryRunOutput("stage2-s02-exact-agreement-audit", "validation_audit"),),
    ),
    DryRunDeclaration(
        "stage3-bound-slack",
        _command("experiments.stage3_mechanism.e01_bound_slack", session=True),
        (ExpectedDryRunOutput("stage3-e01-bound-slack", "pruning_accounting"),),
    ),
    DryRunDeclaration(
        "stage3-pruning-rate",
        _command("experiments.stage3_mechanism.e02_pruning_rate", session=True),
        (ExpectedDryRunOutput("stage3-e02-survival", "pruning_accounting"),),
    ),
    DryRunDeclaration(
        "stage3-order-ablation",
        _command("experiments.stage3_mechanism.e03_order_ablation", session=True),
        (ExpectedDryRunOutput("stage3-e03-kernel-comparison-1t", "timing_comparison", True),),
    ),
    DryRunDeclaration(
        "stage3-checkpoint-ablation",
        _command("experiments.stage3_mechanism.e08_checkpoint_ablation", session=True),
        (ExpectedDryRunOutput("stage3-e08-checkpoint-ablation-1t", "timing_comparison", True),),
    ),
    DryRunDeclaration(
        "stage3-r12c",
        _command("experiments.stage3_mechanism.r12c_interleaved_exact_safe", session=True),
        (ExpectedDryRunOutput("stage3-r12c-exact-safe-interleaved-mt", "timing_comparison", True),),
    ),
    DryRunDeclaration(
        "stage4-fixed-candidates",
        _command("experiments.stage4_integration.e01_fixed_candidate_arms", session=True),
        (
            ExpectedDryRunOutput(
                "stage4-e01-fixed-candidates-equal-work-1t",
                "timing_comparison",
                True,
                (_FULLY_SCORED_EXACT,),
                "-equal",
            ),
            ExpectedDryRunOutput(
                "stage4-e01-fixed-candidates-system-cap-1t",
                "timing_comparison",
                True,
                (_FULL_SCORE_CAP_EXACT, _FULLY_SCORED_UNAVAILABLE),
                "-system",
            ),
        ),
    ),
    DryRunDeclaration(
        "stage4-seeded-tau",
        _command("experiments.stage4_integration.e02_seeded_tau_recovery", session=True),
        (
            ExpectedDryRunOutput(
                "stage4-e02-seeded-tau-recovery-1t",
                "timing_comparison",
                True,
                (_SYSTEM_FULLY_SCORED_EXACT,),
            ),
        ),
    ),
    DryRunDeclaration(
        "stage4-partition-frontier",
        _command("experiments.stage4_integration.e03_partitioned_fused_scan", session=True),
        (
            ExpectedDryRunOutput(
                "stage4-e03-partition-frontier-1t",
                "timing_comparison",
                True,
                (_DOCUMENTS_PROBED_EXACT, _PARTITIONS_PROBED_EXACT),
            ),
        ),
    ),
    DryRunDeclaration(
        "stage5-ir-evaluation",
        _command("experiments.stage5_corect.e01_ir_evaluation", session=True),
        (
            ExpectedDryRunOutput(
                "stage5-e01-ir-evaluation-1t",
                "timing_comparison",
                True,
                (_DOCUMENTS_PROBED_EXACT, _PARTITIONS_PROBED_EXACT),
            ),
        ),
    ),
    DryRunDeclaration(
        "stage5-significance",
        (
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "experiments.stage5_corect.e02_paired_significance",
            "--input",
            "{input}",
            "--n-permutations",
            "100",
            "--output-dir",
            "{output_dir}",
        ),
        (ExpectedDryRunOutput("stage5-e02-paired-significance", "paired_significance"),),
        mode="derived_fixture",
        input_from="stage5-ir-evaluation",
    ),
    DryRunDeclaration(
        "stage5-bm25",
        _command("experiments.stage5_corect.e03_bm25_baseline", session=True),
        (ExpectedDryRunOutput("stage5-e03-bm25-standalone", "timing_comparison", True),),
    ),
)


UNUSABLE_FINAL_DRIVER_CLIS: Mapping[str, str] = {
    "experiments.stage2_testbed.s03_two_mode_smoke": (
        "no fixture/output-dir CLI and writes the legacy ResultRecord schema"
    ),
    "experiments.stage3_mechanism.e04_exact_safe_pruning": (
        "legacy full-data script with no fixture or isolated output CLI"
    ),
    "experiments.stage3_mechanism.e05_approximate_recall_sweep": (
        "legacy full-data script with no fixture or isolated output CLI"
    ),
    "experiments.stage3_mechanism.e06_threshold_policy_ablation": (
        "legacy full-data script with no fixture or isolated output CLI"
    ),
    "experiments.stage3_mechanism.e07_cache_layout_sensitivity": (
        "legacy full-data script with no fixture or isolated output CLI"
    ),
    "experiments.stage3_mechanism.e09_bound_tightness_ablation": (
        "legacy full-data script with no fixture or isolated output CLI"
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    suite = execute_dry_runs(DEFAULT_DRY_RUNS, output_root=arguments.output_root)
    print(f"validated {len(suite.runs)} serial dry runs under {suite.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
