from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.release.final_runs import (
    FINAL_DATASETS,
    FinalRunManifestError,
    load_final_run_manifest,
    validate_final_run_manifest,
)


MANIFEST_PATH = REPO_ROOT / "configs" / "final-runs" / "v1.json"
EVIDENCE_PATH = REPO_ROOT / "artifacts" / "paper_evidence.yaml"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _renumber(document: dict) -> None:
    for order, run in enumerate(document["runs"], start=1):
        run["run_order"] = order


@pytest.mark.artifact
def test_tracked_final_run_manifest_is_complete_and_evidence_linked():
    manifest = load_final_run_manifest(MANIFEST_PATH)

    assert tuple(manifest["required_datasets"]) == FINAL_DATASETS
    assert manifest["execution_policy"] == {
        "strictly_serial": True,
        "stage7_freeze_required": True,
        "output_root": "results/final/v1",
    }
    assert [run["run_order"] for run in manifest["runs"]] == list(
        range(1, len(manifest["runs"]) + 1)
    )
    assert {run["phase"] for run in manifest["runs"]} == {
        "8A",
        "8B",
        "8C",
        "8D",
        "8E",
    }
    assert all(run["outputs"] for run in manifest["runs"])
    assert all(run["dry_run_command"] for run in manifest["runs"])

    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    expected_consumers = {entry["evidence_id"] for entry in evidence["evidence"]}
    linked_consumers = {
        consumer
        for run in manifest["runs"]
        for consumer in run["paper_consumers"]
    }
    assert linked_consumers == expected_consumers


@pytest.mark.artifact
def test_required_coverage_group_rejects_a_missing_dataset():
    manifest = _manifest()
    manifest["runs"] = [
        run
        for run in manifest["runs"]
        if not (
            run["coverage_group"] == "8c-r12c-mt"
            and run["dataset"] == "scidocs"
        )
    ]
    _renumber(manifest)

    with pytest.raises(FinalRunManifestError, match="must cover exactly"):
        validate_final_run_manifest(manifest)


@pytest.mark.artifact
def test_manifest_rejects_duplicate_outputs_and_run_order():
    duplicate = _manifest()
    duplicate["runs"][1]["outputs"][0] = copy.deepcopy(
        duplicate["runs"][0]["outputs"][0]
    )
    with pytest.raises(FinalRunManifestError, match="duplicate output artifact_id"):
        validate_final_run_manifest(duplicate)

    reordered = _manifest()
    reordered["runs"][0]["run_order"] = 2
    with pytest.raises(FinalRunManifestError, match="run_order"):
        validate_final_run_manifest(reordered)


@pytest.mark.artifact
def test_manifest_rejects_unknown_evidence_consumer_and_bad_schema():
    unknown = _manifest()
    unknown["runs"][0]["paper_consumers"] = ["paper.claim.not-registered"]
    with pytest.raises(FinalRunManifestError, match="unknown paper consumers"):
        validate_final_run_manifest(unknown)

    extra_field = _manifest()
    extra_field["runs"][0]["unexpected"] = True
    with pytest.raises(FinalRunManifestError, match="invalid fields"):
        validate_final_run_manifest(extra_field)


@pytest.mark.artifact
def test_manifest_rejects_workload_drift_and_non_fixture_dry_run():
    workload = _manifest()
    workload["runs"][0]["workload"]["query_count"] = 49
    with pytest.raises(FinalRunManifestError, match="frozen mechanism workload"):
        validate_final_run_manifest(workload)

    dry_run = _manifest()
    dry_run["runs"][0]["dry_run_command"] = "uv run python -m experiment"
    with pytest.raises(FinalRunManifestError, match="must pass --fixture"):
        validate_final_run_manifest(dry_run)


@pytest.mark.artifact
def test_significance_separates_analysis_draws_from_run_repetitions():
    manifest = _manifest()
    significance = next(
        run
        for run in manifest["runs"]
        if run["experiment_id"] == "stage5-e02-paired-significance"
    )
    assert significance["repetitions_per_session"] == 1
    assert significance["analysis_parameters"] == {
        "n_permutations": 20_000,
        "seed": 0,
    }

    bad_repetitions = copy.deepcopy(manifest)
    selected = next(
        run
        for run in bad_repetitions["runs"]
        if run["experiment_id"] == "stage5-e02-paired-significance"
    )
    selected["repetitions_per_session"] = 20_000
    with pytest.raises(FinalRunManifestError, match="one analysis invocation"):
        validate_final_run_manifest(bad_repetitions)

    missing_draws = copy.deepcopy(manifest)
    selected = next(
        run
        for run in missing_draws["runs"]
        if run["experiment_id"] == "stage5-e02-paired-significance"
    )
    del selected["analysis_parameters"]["n_permutations"]
    with pytest.raises(FinalRunManifestError, match="invalid fields"):
        validate_final_run_manifest(missing_draws)

    mismatched_command = copy.deepcopy(manifest)
    selected = next(
        run
        for run in mismatched_command["runs"]
        if run["experiment_id"] == "stage5-e02-paired-significance"
    )
    selected["command_template"] = selected["command_template"].replace(
        "--n-permutations 20000", "--n-permutations 100"
    )
    with pytest.raises(FinalRunManifestError, match="does not match"):
        validate_final_run_manifest(mismatched_command)


@pytest.mark.artifact
def test_loader_rejects_duplicate_json_keys(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"manifest_version":"1.0.0","manifest_version":"1.0.0"}',
        encoding="utf-8",
    )

    with pytest.raises(FinalRunManifestError, match="duplicate JSON key"):
        load_final_run_manifest(manifest_path)
