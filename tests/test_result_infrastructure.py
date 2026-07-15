from __future__ import annotations

import json
from pathlib import Path

import pytest

from bondmaxsim.results.catalog import validate_catalog, validate_evidence_manifest
from bondmaxsim.results.io import (
    atomic_write_envelope,
    canonical_json,
    deterministic_result_name,
)
from bondmaxsim.results.migrations import load_result
from bondmaxsim.results.models import (
    ExperimentResultEnvelope,
    ResultValidationError,
    default_provenance,
)
from bondmaxsim.results.schemas import CURRENT_SCHEMAS, HISTORICAL_READERS
from bondmaxsim.results.validation import merge_timing_sessions

REPO_ROOT = Path(__file__).parents[1]

_CANDIDATE_FIELDS = (
    "configured_candidate_cap",
    "configured_full_score_cap",
    "unique_candidates_generated",
    "documents_admitted_to_scoring",
    "documents_fully_scored",
    "documents_probed",
    "partitions_probed",
    "token_hits_inspected",
)


def _candidate_work(**overrides):
    missing = {"value": None, "quality": "unavailable", "source": "fixture unavailable"}
    result = {name: dict(missing) for name in _CANDIDATE_FIELDS}
    result.update(overrides)
    return result


def _environment():
    return {"snapshot_id": "fixture-session", "thread_mode": "single"}


def _envelope(kind, payload, *, artifact_id="fixture-result", protocol=None):
    return ExperimentResultEnvelope.create(
        artifact_id=artifact_id,
        experiment_id="fixture-experiment",
        dataset_id="fixture",
        workload_id="fixture-workload-v1",
        method_configuration={"arm": "fixture"},
        protocol=protocol or {"class": "fixture", "thread_mode": "single"},
        environment=_environment(),
        provenance=default_provenance(command="pytest fixture"),
        payload_kind=kind,
        payload=payload,
        created_at_utc="2026-07-15T12:00:00Z",
    )


def _observation(round_index, arm_id, position, elapsed=100):
    return {
        "session_id": "session-a",
        "round_index": round_index,
        "phase": "measured",
        "arm_id": arm_id,
        "arm_position": position,
        "started_at_utc": "2026-07-15T12:00:00Z",
        "elapsed_ns": elapsed,
        "timer_scope_id": "score-only",
        "completed": True,
    }


def _timing_payload(session_id="session-a"):
    observations = [_observation(0, "a", 0), _observation(0, "b", 1, 120)]
    for row in observations:
        row["session_id"] = session_id
    return {
        "sessions": [
            {
                "session_id": session_id,
                "arm_ids": ["a", "b"],
                "complete": True,
                "observations": observations,
            }
        ]
    }


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("timing_comparison", _timing_payload()),
        ("pruning_accounting", {"rows": [{"cells_scanned_pct": 12.5, "query_count": 3}]}),
        (
            "candidate_frontier",
            {
                "points": [
                    {
                        "comparison_scope": "system_cap",
                        "recall": 0.9,
                        "candidate_work": _candidate_work(
                            documents_fully_scored={
                                "value": None,
                                "quality": "unavailable",
                                "source": "pinned adapter exposes no count",
                            },
                        ),
                    }
                ]
            },
        ),
        ("ir_quality", {"rows": [{"ndcg_at_10": 0.5, "latency_ms": 1.2}]}),
        ("paired_significance", {"comparisons": [{"p_value": 0.2, "n_queries": 10}]}),
        ("validation_audit", {"checks": [{"status": "pass"}], "passed": True}),
    ],
)
def test_each_payload_kind_has_semantic_validation(kind, payload):
    assert _envelope(kind, payload).payload_kind == kind


@pytest.mark.parametrize(
    "kind,payload,match",
    [
        ("pruning_accounting", {"rows": [{"cells_scanned_pct": 101.0}]}, r"\[0, 100\]"),
        ("ir_quality", {"rows": [{"ndcg_at_10": 1.1}]}, r"\[0, 1\]"),
        ("paired_significance", {"comparisons": [{"p_value": 0.5, "n_queries": 0}]}, "n_queries"),
        ("validation_audit", {"checks": [{"status": "fail"}], "passed": True}, "cannot pass"),
        (
            "candidate_frontier",
            {
                "points": [
                    {
                        "comparison_scope": "equal_work_reranking",
                        "candidate_work": _candidate_work(
                            documents_fully_scored={
                                "value": 10,
                                "quality": "unavailable",
                                "source": "bad",
                            },
                        ),
                    }
                ]
            },
            "must be null",
        ),
    ],
)
def test_semantically_invalid_payloads_fail(kind, payload, match):
    with pytest.raises(ResultValidationError, match=match):
        _envelope(kind, payload)


def test_incomplete_timing_round_cannot_be_complete():
    payload = _timing_payload()
    payload["sessions"][0]["observations"].pop()
    with pytest.raises(ResultValidationError, match="incomplete measured round"):
        _envelope("timing_comparison", payload)


def test_atomic_serialization_is_canonical_and_round_trips(tmp_path):
    envelope = _envelope("timing_comparison", _timing_payload())
    output = tmp_path / "nested" / "result.json"
    atomic_write_envelope(output, envelope)
    assert output.read_text(encoding="utf-8") == canonical_json(envelope.to_dict())
    assert output.read_bytes().endswith(b"\n")
    assert load_result(output) == envelope
    assert list(output.parent.glob(".*.tmp")) == []


def test_deterministic_result_name_has_no_timestamp():
    assert deterministic_result_name(
        "Stage 3 / E02", "SciFact", "beir-scifact-mechanism-seed42-n50-v1"
    ) == "stage-3-e02__scifact__beir-scifact-mechanism-seed42-n50-v1.json"


def test_all_frozen_historical_shapes_load_without_rewrite():
    fixture_dir = REPO_ROOT / "artifacts" / "baseline" / "fixtures"
    fixtures = sorted(fixture_dir.glob("*.json"))
    assert len(fixtures) == 17
    for fixture in fixtures:
        before = fixture.read_bytes()
        envelope = load_result(fixture)
        assert envelope.payload["historical_migration"] is True
        assert envelope.provenance["historical_schema_name"] in HISTORICAL_READERS
        assert envelope.provenance["source_sha256"]
        assert fixture.read_bytes() == before


def test_schema_registry_resources_exist_and_are_json():
    for registration in CURRENT_SCHEMAS.values():
        resource = REPO_ROOT / registration.resource
        assert resource.exists()
        assert isinstance(json.loads(resource.read_text(encoding="utf-8")), dict)


def test_merge_guards_and_override_provenance():
    left = _envelope("timing_comparison", _timing_payload("session-a"), artifact_id="left")
    right = _envelope("timing_comparison", _timing_payload("session-b"), artifact_id="right")
    merged = merge_timing_sessions(left, right, artifact_id="merged")
    assert len(merged.payload["sessions"]) == 2
    assert merged.provenance["input_artifact_ids"] == ["left", "right"]

    incompatible = ExperimentResultEnvelope.create(
        artifact_id="other",
        experiment_id="fixture-experiment",
        dataset_id="fixture",
        workload_id="different-workload",
        method_configuration={"arm": "fixture"},
        protocol={"class": "fixture", "thread_mode": "single"},
        environment=_environment(),
        provenance=default_provenance(command="pytest fixture"),
        payload_kind="timing_comparison",
        payload=_timing_payload("session-c"),
        created_at_utc="2026-07-15T12:00:00Z",
    )
    with pytest.raises(ResultValidationError, match="workload_id"):
        merge_timing_sessions(left, incompatible, artifact_id="bad")
    overridden = merge_timing_sessions(
        left, incompatible, artifact_id="derived", allow_incompatible=True
    )
    assert "workload_id" in overridden.provenance["merge_overrides"]


def test_catalog_and_evidence_manifests_are_populated_and_validate():
    catalog = validate_catalog(REPO_ROOT / "artifacts" / "catalog.yaml", workspace=REPO_ROOT)
    manifest = validate_evidence_manifest(
        REPO_ROOT / "artifacts" / "paper_evidence.yaml", catalog
    )
    assert catalog["catalog_version"] == "1.0.0"
    assert catalog["artifacts"]
    assert manifest["manifest_version"] == "1.0.0"
    assert manifest["evidence"]


def test_catalog_rejects_cycles_and_current_invalid_evidence(tmp_path):
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": "a",
                        "status": "superseded",
                        "supersession": {"successor": "b", "scope": "/"},
                    },
                    {
                        "artifact_id": "b",
                        "status": "superseded",
                        "supersession": {"successor": "a", "scope": "/"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ResultValidationError, match="cycle"):
        validate_catalog(catalog_path)

    valid_catalog_path = tmp_path / "valid.yaml"
    valid_catalog_path.write_text(
        json.dumps({"artifacts": [{"artifact_id": "bad", "status": "invalid"}]}),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.yaml"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        "evidence_id": "claim",
                        "status": "current",
                        "references": [{"artifact_id": "bad", "pointer": "/payload"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    catalog = validate_catalog(valid_catalog_path)
    with pytest.raises(ResultValidationError, match="invalid evidence"):
        validate_evidence_manifest(evidence_path, catalog)
