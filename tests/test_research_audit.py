from __future__ import annotations

import json
from pathlib import Path

import pytest

from bondmaxsim.research_audit import (
    NATIVE_BINARIES,
    ResearchAuditError,
    _require_native_binaries,
    audit_generated_paper_outputs,
    audit_governance,
    audit_result_schemas,
    tracked_result_paths,
)


def _historical_result() -> dict[str, object]:
    return {
        "dataset": "fixture",
        "method": "fixture-method",
        "num_docs": 2,
        "num_queries": 1,
        "recall_vs_exact_at_10": 1.0,
        "ms_per_query": None,
    }


def test_schema_discovery_validates_every_declared_tracked_result(tmp_path: Path):
    first = Path("results/json/one.json")
    second = Path("results/archive/two.json")
    for relative in (first, second):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(_historical_result()), encoding="utf-8")

    report = audit_result_schemas(tmp_path, tracked_files=(first, second))

    assert report["result_count"] == 2
    assert report["schemas"] == {"historical.result-record.v1": 2}


def test_schema_discovery_fails_closed_on_zero_results(tmp_path: Path):
    with pytest.raises(ResearchAuditError, match="zero tracked result"):
        tracked_result_paths(tmp_path, tracked_files=(Path("notes.json"),))


def test_schema_audit_reports_bad_tracked_json(tmp_path: Path):
    relative = Path("results/json/bad.json")
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ResearchAuditError, match="bad.json"):
        audit_result_schemas(tmp_path, tracked_files=(relative,))


def test_native_preflight_requires_all_three_shared_libraries(tmp_path: Path):
    for relative in NATIVE_BINARIES[:-1]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")

    with pytest.raises(ResearchAuditError, match=str(NATIVE_BINARIES[-1])):
        _require_native_binaries(tmp_path)


def test_repository_generated_paper_outputs_are_current():
    workspace = Path(__file__).resolve().parents[1]
    report = audit_generated_paper_outputs(workspace)
    assert report["generated_output_count"] > 0
    assert len(report["generated_outputs"]) == report["generated_output_count"]


def test_repository_catalog_and_evidence_manifest_are_populated_and_valid():
    workspace = Path(__file__).resolve().parents[1]
    report = audit_governance(workspace)
    assert report["artifact_count"] > 0
    assert report["evidence_count"] > 0


def test_ci_and_makefile_expose_all_research_tiers():
    workspace = Path(__file__).resolve().parents[1]
    makefile = (workspace / "Makefile").read_text(encoding="utf-8")
    workflow = (workspace / ".github/workflows/research-audit.yml").read_text(
        encoding="utf-8"
    )
    for tier in (
        "test-unit",
        "test-native",
        "test-integration",
        "test-artifact",
        "test-reproduction",
        "test-sanitize",
        "test-sanitizer",
        "test-full",
        "research-audit",
    ):
        assert tier in makefile
    assert "--native-required" in makefile
    assert "make research-audit" in workflow
    assert "make test-sanitize" in workflow
