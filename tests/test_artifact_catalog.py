from __future__ import annotations

import json
from pathlib import Path

import pytest

from bondmaxsim.results.catalog import (
    generate_catalog,
    main,
    pointer_status,
    render_status_inventory,
    validate_catalog,
    validate_evidence_manifest,
)
from bondmaxsim.results.models import ResultValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "artifacts" / "catalog.yaml"


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_catalog_covers_every_result_and_validation_report_with_checksums():
    catalog = validate_catalog(CATALOG_PATH, workspace=REPO_ROOT)
    expected = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "results" / "json").glob("*.json")
    }
    expected.add("artifacts/baseline/agreement-cases.json")
    entries = catalog["artifacts"]

    assert len(entries) == 74
    assert {entry["path"] for entry in entries} == expected
    assert all(len(entry["sha256"]) == 64 for entry in entries)
    assert all(entry["command"].startswith("uv run") for entry in entries)
    assert all(entry["outputs"] and entry["path"] in entry["outputs"] for entry in entries)


def test_catalog_and_human_inventory_are_deterministically_generated():
    generated = generate_catalog(REPO_ROOT)
    assert generated == _catalog()
    assert render_status_inventory(generated) == (
        REPO_ROOT / "docs" / "artifact-status.md"
    ).read_text(encoding="utf-8")
    assert main(["--workspace", str(REPO_ROOT), "--check"]) == 0


def test_partial_e08_e09_latency_supersession_retains_accounting_fields():
    entries = {entry["artifact_id"]: entry for entry in _catalog()["artifacts"]}
    e08 = entries["stage3_mechanism_e08_checkpoint_ablation_nfcorpus"]
    e09 = entries["stage3_mechanism_e09_bound_tightness_ablation_nfcorpus"]

    assert e08["status"] == "historical"
    assert pointer_status(e08, "/arms/0/ms_per_query_1t") == "superseded"
    assert pointer_status(e08, "/arms") == "superseded"
    assert pointer_status(e08, "/arms/0/pruned_docs_pct_fused") == "diagnostic"
    assert {edge["successor"] for edge in e08["supersessions"]} == {
        "stage3_mechanism_r12c_interleaved_exact_safe_1t",
        "stage3_mechanism_r12c_interleaved_exact_safe_mt",
    }

    assert pointer_status(e09, "/dense_arm/ms_per_query") == "superseded"
    assert pointer_status(e09, "/arms/0/pruned_docs_pct") == "diagnostic"
    assert e09["supersessions"][0]["successor"].endswith("_mt")


def test_current_evidence_cannot_select_partial_supersession(tmp_path):
    catalog = validate_catalog(CATALOG_PATH, workspace=REPO_ROOT)
    manifest = tmp_path / "evidence.json"
    manifest.write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        "evidence_id": "bad-latency",
                        "status": "current",
                        "references": [
                            {
                                "artifact_id": "stage3_mechanism_e08_checkpoint_ablation_nfcorpus",
                                "pointer": "/arms/0/ms_per_query_mt",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ResultValidationError, match="superseded"):
        validate_evidence_manifest(manifest, catalog)


def test_checksum_and_relationship_backlinks_are_enforced(tmp_path):
    checksum_catalog = _catalog()
    checksum_catalog["artifacts"][0]["sha256"] = "0" * 64
    checksum_path = tmp_path / "checksum.json"
    checksum_path.write_text(json.dumps(checksum_catalog), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="checksum mismatch"):
        validate_catalog(checksum_path, workspace=REPO_ROOT)

    relationship_catalog = _catalog()
    successor = next(
        entry
        for entry in relationship_catalog["artifacts"]
        if entry["artifact_id"] == "stage3_mechanism_r12c_interleaved_exact_safe_1t"
    )
    successor["supersedes"] = []
    relationship_path = tmp_path / "relationship.json"
    relationship_path.write_text(json.dumps(relationship_catalog), encoding="utf-8")
    with pytest.raises(ResultValidationError, match="lacks supersedes backlink"):
        validate_catalog(relationship_path, workspace=REPO_ROOT)
