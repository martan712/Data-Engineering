"""Fail-closed validation for the versioned Stage 8 final-run manifest."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from bondmaxsim.config import REPO_ROOT

FINAL_DATASETS = ("scifact", "nfcorpus", "arguana", "scidocs")
_MANIFEST_VERSION = "1.0.0"
_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_THREAD_MODES = {"one-thread", "all-core", "not-applicable", "standalone"}
_PHASES = {"8A", "8B", "8C", "8D", "8E"}
_DRY_RUN_KINDS = {"fixture", "one-query"}
_TOP_LEVEL_FIELDS = {
    "manifest_version",
    "manifest_id",
    "required_datasets",
    "execution_policy",
    "runs",
}
_RUN_FIELDS = {
    "run_id",
    "phase",
    "run_order",
    "experiment_id",
    "dataset",
    "thread_mode",
    "session_count",
    "warmups_per_session",
    "repetitions_per_session",
    "analysis_parameters",
    "workload",
    "index_backend",
    "outputs",
    "paper_consumers",
    "command_template",
    "dry_run_kind",
    "dry_run_command",
    "coverage_group",
    "requires_all_datasets",
}
_ANALYSIS_FIELDS = {"n_permutations", "seed"}
_WORKLOAD_FIELDS = {"workload_id", "regime", "query_count"}
_OUTPUT_FIELDS = {"artifact_id", "path"}
_POLICY_FIELDS = {
    "strictly_serial",
    "stage7_freeze_required",
    "output_root",
}


class FinalRunManifestError(ValueError):
    """Raised when a final-run manifest is incomplete or internally unsafe."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalRunManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError) as error:
        raise FinalRunManifestError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise FinalRunManifestError(f"{label} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FinalRunManifestError(
            f"{label} has invalid fields (missing={missing}, extra={extra})"
        )


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalRunManifestError(f"{label} must be a non-empty string")
    return value


def _stable_id(value: Any, *, label: str) -> str:
    selected = _nonempty_string(value, label=label)
    if not _ID.fullmatch(selected):
        raise FinalRunManifestError(f"{label} must be a stable lowercase ID")
    return selected


def _integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalRunManifestError(f"{label} must be an integer >= {minimum}")
    return value


def _relative_json_path(value: Any, *, label: str, output_root: str) -> str:
    selected = _nonempty_string(value, label=label)
    path = PurePosixPath(selected)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
        raise FinalRunManifestError(f"{label} must be a safe relative JSON path")
    root = PurePosixPath(output_root)
    if path.parts[: len(root.parts)] != root.parts:
        raise FinalRunManifestError(f"{label} must be below {output_root}")
    return selected


def _evidence_ids(path: Path) -> set[str]:
    document = _read_json(path, label="paper evidence manifest")
    entries = document.get("evidence")
    if not isinstance(entries, list):
        raise FinalRunManifestError("paper evidence manifest needs an evidence list")
    ids: set[str] = set()
    for entry in entries:
        evidence_id = entry.get("evidence_id") if isinstance(entry, Mapping) else None
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in ids:
            raise FinalRunManifestError("paper evidence IDs must be unique strings")
        ids.add(evidence_id)
    return ids


def _validate_workload(run: Mapping[str, Any], *, label: str) -> None:
    workload = run["workload"]
    if not isinstance(workload, Mapping):
        raise FinalRunManifestError(f"{label}.workload must be an object")
    _exact_fields(workload, _WORKLOAD_FIELDS, label=f"{label}.workload")
    workload_id = _stable_id(workload["workload_id"], label=f"{label}.workload_id")
    regime = workload["regime"]
    if regime not in {"mechanism", "qrels_test"}:
        raise FinalRunManifestError(f"{label}.workload.regime is invalid")
    query_count = workload["query_count"]
    if regime == "mechanism":
        _integer(query_count, label=f"{label}.workload.query_count", minimum=1)
        expected = f"beir-{run['dataset']}-mechanism-seed42-n50-v1"
        if query_count != 50 or workload_id != expected:
            raise FinalRunManifestError(f"{label} must use frozen mechanism workload {expected}")
    else:
        if query_count != "all-qrels-test":
            raise FinalRunManifestError(
                f"{label}.workload.query_count must be 'all-qrels-test'"
            )
        expected = f"beir-{run['dataset']}-qrels-test-v1"
        if workload_id != expected:
            raise FinalRunManifestError(f"{label} must use frozen qrels workload {expected}")


def validate_final_run_manifest(
    manifest: Mapping[str, Any],
    *,
    evidence_manifest_path: Path | None = None,
) -> Mapping[str, Any]:
    """Validate schema, ordering, coverage, outputs, and paper-evidence links."""

    if not isinstance(manifest, Mapping):
        raise FinalRunManifestError("final-run manifest must be an object")
    _exact_fields(manifest, _TOP_LEVEL_FIELDS, label="manifest")
    if manifest["manifest_version"] != _MANIFEST_VERSION:
        raise FinalRunManifestError(
            f"manifest_version must be {_MANIFEST_VERSION!r}"
        )
    _stable_id(manifest["manifest_id"], label="manifest.manifest_id")
    datasets = manifest["required_datasets"]
    if not isinstance(datasets, list) or tuple(datasets) != FINAL_DATASETS:
        raise FinalRunManifestError(
            f"required_datasets must be exactly {list(FINAL_DATASETS)!r} in order"
        )
    policy = manifest["execution_policy"]
    if not isinstance(policy, Mapping):
        raise FinalRunManifestError("execution_policy must be an object")
    _exact_fields(policy, _POLICY_FIELDS, label="execution_policy")
    if policy["strictly_serial"] is not True or policy["stage7_freeze_required"] is not True:
        raise FinalRunManifestError("Stage 8 must be serial and gated by the Stage 7 freeze")
    output_root = _nonempty_string(policy["output_root"], label="execution_policy.output_root")
    root_path = PurePosixPath(output_root)
    if root_path.is_absolute() or ".." in root_path.parts:
        raise FinalRunManifestError("execution_policy.output_root must be relative and safe")

    evidence_path = evidence_manifest_path or REPO_ROOT / "artifacts" / "paper_evidence.yaml"
    known_consumers = _evidence_ids(evidence_path)
    runs = manifest["runs"]
    if not isinstance(runs, list) or not runs:
        raise FinalRunManifestError("manifest.runs must be a non-empty list")

    run_ids: set[str] = set()
    output_ids: set[str] = set()
    output_paths: set[str] = set()
    orders: list[int] = []
    coverage: dict[str, set[str]] = defaultdict(set)
    coverage_flags: dict[str, bool] = {}
    coverage_shapes: set[tuple[str, str, str]] = set()

    for index, run in enumerate(runs):
        label = f"runs[{index}]"
        if not isinstance(run, Mapping):
            raise FinalRunManifestError(f"{label} must be an object")
        actual_fields = set(run)
        missing_fields = (_RUN_FIELDS - {"analysis_parameters"}) - actual_fields
        extra_fields = actual_fields - _RUN_FIELDS
        if missing_fields or extra_fields:
            raise FinalRunManifestError(
                f"{label} has invalid fields "
                f"(missing={sorted(missing_fields)}, extra={sorted(extra_fields)})"
            )
        run_id = _stable_id(run["run_id"], label=f"{label}.run_id")
        if run_id in run_ids:
            raise FinalRunManifestError(f"duplicate run_id {run_id}")
        run_ids.add(run_id)
        if run["phase"] not in _PHASES:
            raise FinalRunManifestError(f"{label}.phase is invalid")
        order = _integer(run["run_order"], label=f"{label}.run_order", minimum=1)
        orders.append(order)
        experiment_id = _stable_id(
            run["experiment_id"], label=f"{label}.experiment_id"
        )
        if run["dataset"] not in FINAL_DATASETS:
            raise FinalRunManifestError(f"{label}.dataset is not a final dataset")
        if run["thread_mode"] not in _THREAD_MODES:
            raise FinalRunManifestError(f"{label}.thread_mode is invalid")
        _integer(run["session_count"], label=f"{label}.session_count", minimum=1)
        _integer(run["warmups_per_session"], label=f"{label}.warmups_per_session", minimum=0)
        repetitions = _integer(
            run["repetitions_per_session"],
            label=f"{label}.repetitions_per_session",
            minimum=1,
        )
        analysis_parameters = run.get("analysis_parameters")
        if experiment_id == "stage5-e02-paired-significance":
            if repetitions != 1:
                raise FinalRunManifestError(
                    f"{label} paired significance must be one analysis invocation"
                )
            if not isinstance(analysis_parameters, Mapping):
                raise FinalRunManifestError(
                    f"{label}.analysis_parameters must be an object"
                )
            _exact_fields(
                analysis_parameters,
                _ANALYSIS_FIELDS,
                label=f"{label}.analysis_parameters",
            )
            _integer(
                analysis_parameters["n_permutations"],
                label=f"{label}.analysis_parameters.n_permutations",
                minimum=1,
            )
            _integer(
                analysis_parameters["seed"],
                label=f"{label}.analysis_parameters.seed",
                minimum=0,
            )
        elif analysis_parameters not in (None, {}):
            raise FinalRunManifestError(
                f"{label}.analysis_parameters is only valid for analysis runs"
            )
        _validate_workload(run, label=label)
        backend = run["index_backend"]
        if backend is not None:
            _stable_id(backend, label=f"{label}.index_backend")

        outputs = run["outputs"]
        if not isinstance(outputs, list) or not outputs:
            raise FinalRunManifestError(f"{label}.outputs must be a non-empty list")
        for output_index, output in enumerate(outputs):
            output_label = f"{label}.outputs[{output_index}]"
            if not isinstance(output, Mapping):
                raise FinalRunManifestError(f"{output_label} must be an object")
            _exact_fields(output, _OUTPUT_FIELDS, label=output_label)
            artifact_id = _stable_id(output["artifact_id"], label=f"{output_label}.artifact_id")
            path = _relative_json_path(
                output["path"], label=f"{output_label}.path", output_root=output_root
            )
            if artifact_id in output_ids:
                raise FinalRunManifestError(f"duplicate output artifact_id {artifact_id}")
            if path in output_paths:
                raise FinalRunManifestError(f"duplicate output path {path}")
            output_ids.add(artifact_id)
            output_paths.add(path)

        consumers = run["paper_consumers"]
        if (
            not isinstance(consumers, list)
            or any(not isinstance(value, str) or not value for value in consumers)
            or len(consumers) != len(set(consumers))
        ):
            raise FinalRunManifestError(f"{label}.paper_consumers must contain unique IDs")
        unknown = sorted(set(consumers) - known_consumers)
        if unknown:
            raise FinalRunManifestError(f"{label} links unknown paper consumers {unknown}")

        command = _nonempty_string(run["command_template"], label=f"{label}.command_template")
        if "{dataset}" not in command or "{output_dir}" not in command:
            raise FinalRunManifestError(
                f"{label}.command_template must expose dataset and output_dir placeholders"
            )
        if run["session_count"] > 1 and "{session_id}" not in command:
            raise FinalRunManifestError(
                f"{label}.command_template must expose session_id for repeated sessions"
            )
        if experiment_id == "stage5-e02-paired-significance":
            permutations = analysis_parameters["n_permutations"]
            seed = analysis_parameters["seed"]
            if (
                f"--n-permutations {permutations}" not in command
                or f"--seed {seed}" not in command
            ):
                raise FinalRunManifestError(
                    f"{label}.command_template does not match analysis_parameters"
                )
        dry_kind = run["dry_run_kind"]
        if dry_kind not in _DRY_RUN_KINDS:
            raise FinalRunManifestError(f"{label}.dry_run_kind is invalid")
        dry_command = _nonempty_string(run["dry_run_command"], label=f"{label}.dry_run_command")
        if dry_kind == "fixture" and "--fixture" not in dry_command:
            raise FinalRunManifestError(f"{label} fixture dry run must pass --fixture")
        if dry_kind == "one-query" and "one-query" not in dry_command:
            raise FinalRunManifestError(
                f"{label} one-query dry run must explicitly identify one-query mode"
            )

        group = _stable_id(run["coverage_group"], label=f"{label}.coverage_group")
        all_datasets = run["requires_all_datasets"]
        if not isinstance(all_datasets, bool):
            raise FinalRunManifestError(f"{label}.requires_all_datasets must be boolean")
        previous = coverage_flags.setdefault(group, all_datasets)
        if previous != all_datasets:
            raise FinalRunManifestError(f"coverage group {group} has inconsistent policy")
        shape = (group, run["dataset"], run["thread_mode"])
        if shape in coverage_shapes:
            raise FinalRunManifestError(f"duplicate dataset/thread entry in coverage group {group}")
        coverage_shapes.add(shape)
        coverage[group].add(run["dataset"])

    expected_orders = list(range(1, len(runs) + 1))
    if orders != expected_orders:
        raise FinalRunManifestError("run_order must be unique, contiguous, and match list order")
    required = set(FINAL_DATASETS)
    for group, group_datasets in coverage.items():
        if coverage_flags[group] and group_datasets != required:
            raise FinalRunManifestError(
                f"coverage group {group} must cover exactly {sorted(required)}"
            )
    return manifest


def load_final_run_manifest(
    path: Path | str = REPO_ROOT / "configs" / "final-runs" / "v1.json",
    *,
    evidence_manifest_path: Path | None = None,
) -> Mapping[str, Any]:
    """Read and validate a final-run manifest without executing any experiment."""

    document = _read_json(Path(path), label="final-run manifest")
    return validate_final_run_manifest(
        document, evidence_manifest_path=evidence_manifest_path
    )
