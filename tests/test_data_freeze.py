from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bondmaxsim.data.config import DATASETS, load_data_configuration
from bondmaxsim.data.freeze import (
    DataFreezeError,
    create_freeze_bundle,
    frozen_provenance_expectations,
    resolve_frozen_dataset,
    validate_freeze_bundle,
)
from bondmaxsim.data.generation import PublicSource, generate_dataset
from bondmaxsim.data.indexes import build_derived_indexes
from bondmaxsim.data.loader import load_dataset, load_eval_queries


_EXPERIMENT_CONFIGURATION_SHA256 = "e" * 64


class _FixtureEncoder:
    def encode(self, texts, *, is_query, **kwargs):
        del kwargs
        rows = []
        for index, text in enumerate(texts):
            count = 2 + ((len(text) + index + int(is_query)) % 3)
            matrix = np.zeros((count, 128), dtype=np.float32)
            matrix[:, (index + int(is_query)) % 128] = 1.0
            rows.append(matrix)
        return rows


def _source(dataset, frozen):
    del frozen
    return PublicSource(
        tuple(f"{dataset}-d{index}" for index in range(6)),
        tuple(f"{dataset} document {index}" for index in range(6)),
        tuple(f"{dataset}-q{index}" for index in range(5)),
        tuple(f"{dataset} query {index}" for index in range(5)),
        {
            f"{dataset}-q1": {f"{dataset}-d1": 1},
            f"{dataset}-q3": {f"{dataset}-d4": 2},
        },
    )


def _fake_builder(destination, doc_values, doc_starts, spec):
    payload = json.dumps(
        {
            "backend": spec.backend,
            "settings": spec.settings,
            "shape": list(doc_values.shape),
            "starts": doc_starts.tolist(),
        },
        sort_keys=True,
    ).encode("utf-8")
    if spec.backend == "faiss":
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return (destination,)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = destination / "metadata.json"
    index = destination / "fast_plaid_index" / "index.bin"
    index.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_bytes(payload)
    index.write_bytes(payload[::-1])
    return (metadata, index)


def _equivalence(root: Path, configuration_sha256: str) -> Path:
    reports = [
        {
            "schema_name": "bondmaxsim.data-equivalence",
            "schema_version": "1.0.0",
            "dataset": dataset,
            "configuration_sha256": configuration_sha256,
            "identity": {},
            "arrays": {},
            "rankings": {
                "mechanism": {
                    "status": "complete",
                    "query_count": 5,
                    "boundary_tie_complete": True,
                },
                "quality": {
                    "status": "complete",
                    "query_count": 2,
                    "boundary_tie_complete": True,
                },
            },
            "qrels": {
                "status": "complete",
                "semantic_equal": True,
                "metrics_equal": True,
            },
            "representative_accounting": {"status": "complete", "equal": False},
            "bitwise_equivalent": False,
            "audit_complete": True,
            "cross_checks_equal": False,
            "final_eligibility": False,
            "decision": "not_bitwise_equivalent_rerun_all_data_dependent_evidence",
        }
        for dataset in DATASETS
    ]
    path = root / "equivalence.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "bondmaxsim.data-equivalence-suite",
                "schema_version": "1.0.0",
                "selected_datasets": list(DATASETS),
                "scope_complete": True,
                "audits_complete": True,
                "bitwise_equivalent": False,
                "cross_checks_equal": False,
                "final_eligibility": False,
                "reports": reports,
                "decision": "not_bitwise_equivalent_rerun_all_data_dependent_evidence",
            }
        ),
        encoding="utf-8",
    )
    return path


def _use_absent_legacy_quality_option(
    suite: dict, dataset: str = "scidocs"
) -> dict:
    suite["schema_version"] = "1.1.0"
    for report in suite["reports"]:
        report["schema_version"] = "1.1.0"
        if report["dataset"] != dataset:
            continue
        query_count = 2
        report["rankings"]["quality"] = {
            "status": "not_comparable_no_legacy_quality_artifact",
            "query_count": 0,
            "boundary_tie_complete": False,
        }
        report["quality_artifacts"] = {
            "legacy": {
                "expected_path": f"embeddings/{dataset}_test_queries.npz",
                "exists": False,
                "status": "expected_test_query_artifact_absent",
                "fallback_main_query_count": 5,
            },
            "generated": {
                "path": f"embeddings/{dataset}_test_queries.npz",
                "exists": True,
                "authoritative": True,
                "query_count": query_count,
                "qrels_query_count": query_count,
                "full_qrels_coverage": True,
            },
        }
        report["qrels"].update(
            status="complete",
            semantic_equal=True,
            metrics_status="not_comparable_no_legacy_quality_artifact",
            metrics_equal=False,
        )
        report["bitwise_equivalent"] = False
        report["audit_complete"] = True
        report["cross_checks_equal"] = False
        report["final_eligibility"] = False
        report["decision"] = (
            "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
        )
    return suite


def _frozen_fixture(tmp_path: Path):
    root = tmp_path / "data/generated/v1"
    frozen = load_data_configuration()
    for dataset in DATASETS:
        generate_dataset(
            dataset,
            output_root=root,
            frozen=frozen,
            source_loader=_source,
            encoder=_FixtureEncoder(),
            fixture=True,
        )
        build_derived_indexes(
            dataset,
            output_root=root,
            frozen=frozen,
            builders={"faiss": _fake_builder, "plaid": _fake_builder},
            fixture=True,
        )
    equivalence = _equivalence(root, frozen.sha256)
    freeze_path = tmp_path / "artifacts/data/freeze-v1.json"
    bundle = create_freeze_bundle(
        generated_root=root,
        equivalence_path=equivalence,
        output_path=freeze_path,
        frozen=frozen,
        fixture=True,
    )
    return root, freeze_path, frozen, bundle


def test_four_dataset_freeze_binds_data_workloads_indexes_and_decision(tmp_path: Path):
    root, freeze_path, frozen, bundle = _frozen_fixture(tmp_path)
    assert set(bundle["datasets"]) == set(DATASETS)
    assert bundle["configuration"]["sha256"] == frozen.sha256
    assert bundle["equivalence"]["decision"].startswith("not_bitwise_equivalent")
    for dataset, record in bundle["datasets"].items():
        assert len(record["data_sha256"]) == 64
        assert set(record["workloads"]) == {"mechanism", "quality"}
        assert set(record["indexes"]) == {"faiss", "plaid"}
        assert len(record["indexes"]["faiss"]["index_sha256"]) == 64
        assert record["equivalence"]["report_sha256"] == (
            bundle["equivalence"]["reports"][dataset]["report_sha256"]
        )
    assert validate_freeze_bundle(
        freeze_path, generated_root=root, frozen=frozen, fixture=True
    ) == bundle


def test_freeze_exposes_exact_result_provenance_per_backend(tmp_path: Path):
    root, freeze_path, frozen, bundle = _frozen_fixture(tmp_path)
    indexless = frozen_provenance_expectations(
        "scifact",
        configuration_sha256=_EXPERIMENT_CONFIGURATION_SHA256,
        freeze_path=freeze_path,
        generated_root=root,
        frozen=frozen,
        fixture=True,
    )
    faiss = frozen_provenance_expectations(
        "scifact",
        configuration_sha256=_EXPERIMENT_CONFIGURATION_SHA256,
        index_backend="faiss",
        freeze_path=freeze_path,
        generated_root=root,
        frozen=frozen,
        fixture=True,
    )
    assert indexless == {
        "configuration_sha256": _EXPERIMENT_CONFIGURATION_SHA256,
        "data_sha256": bundle["datasets"]["scifact"]["data_sha256"],
        "index_sha256": None,
    }
    assert faiss["index_sha256"] == (
        bundle["datasets"]["scifact"]["indexes"]["faiss"]["index_sha256"]
    )
    with pytest.raises(DataFreezeError, match="not frozen"):
        frozen_provenance_expectations(
            "scifact",
            configuration_sha256=_EXPERIMENT_CONFIGURATION_SHA256,
            index_backend="unknown",
            freeze_path=freeze_path,
            generated_root=root,
            frozen=frozen,
            fixture=True,
        )


def test_frozen_loader_routes_only_to_validated_generated_v1(tmp_path: Path):
    root, freeze_path, _, _ = _frozen_fixture(tmp_path)
    selection = resolve_frozen_dataset(
        "scifact", freeze_path=freeze_path, generated_root=root, fixture=True
    )
    assert selection.embedding_path == root / "embeddings/scifact.npz"
    assert selection.data_configuration_sha256 == load_data_configuration().sha256
    assert len(selection.data_sha256) == 64
    assert set(selection.index_sha256_by_backend) == {"faiss", "plaid"}
    flat, starts, queries = load_dataset(
        "scifact",
        frozen=True,
        freeze_manifest=freeze_path,
        generated_root=root,
        freeze_fixture=True,
    )
    quality, quality_ids = load_eval_queries(
        "scifact",
        frozen=True,
        freeze_manifest=freeze_path,
        generated_root=root,
        freeze_fixture=True,
    )
    assert flat.shape[1] == 128
    assert len(starts) == 6
    assert len(queries) == 5
    assert len(quality) == 2
    assert quality_ids == ["scifact-q1", "scifact-q3"]


def test_frozen_loader_refuses_legacy_cache_override(tmp_path: Path):
    root, freeze_path, _, _ = _frozen_fixture(tmp_path)
    legacy = tmp_path / "data/embeddings"
    legacy.mkdir(parents=True)
    with pytest.raises(DataFreezeError, match="legacy or alternate"):
        load_dataset(
            "scifact",
            data_dir=legacy,
            frozen=True,
            freeze_manifest=freeze_path,
            generated_root=root,
            freeze_fixture=True,
        )


def test_freeze_and_frozen_loader_refuse_generated_file_drift(tmp_path: Path):
    root, freeze_path, frozen, _ = _frozen_fixture(tmp_path)
    ids = root / "beir_ids/scifact_ids.json"
    ids.write_text(ids.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest does not match"):
        validate_freeze_bundle(
            freeze_path, generated_root=root, frozen=frozen, fixture=True
        )
    with pytest.raises(RuntimeError, match="manifest does not match"):
        load_dataset(
            "scifact",
            frozen=True,
            freeze_manifest=freeze_path,
            generated_root=root,
            freeze_fixture=True,
        )


def test_sampled_equivalence_cannot_be_frozen(tmp_path: Path):
    root, _, frozen, _ = _frozen_fixture(tmp_path)
    equivalence = json.loads((root / "equivalence.json").read_text(encoding="utf-8"))
    equivalence["reports"][0]["rankings"]["mechanism"]["status"] = "sampled"
    (root / "equivalence.json").write_text(json.dumps(equivalence), encoding="utf-8")
    with pytest.raises(DataFreezeError, match="sampled equivalence"):
        create_freeze_bundle(
            generated_root=root,
            equivalence_path=root / "equivalence.json",
            output_path=tmp_path / "new-freeze.json",
            frozen=frozen,
            fixture=True,
        )


@pytest.mark.parametrize(
    "digest",
    ["not-a-digest", "A" * 64, "a" * 63, None],
)
def test_result_expectations_require_explicit_experiment_configuration_hash(
    tmp_path: Path, digest
):
    root, freeze_path, frozen, _ = _frozen_fixture(tmp_path)
    with pytest.raises(DataFreezeError, match="experiment configuration_sha256"):
        frozen_provenance_expectations(
            "scifact",
            configuration_sha256=digest,
            freeze_path=freeze_path,
            generated_root=root,
            frozen=frozen,
            fixture=True,
        )


def test_diagnostic_only_hardened_equivalence_suite_cannot_be_frozen(tmp_path: Path):
    root, _, frozen, _ = _frozen_fixture(tmp_path)
    equivalence_path = root / "equivalence.json"
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    equivalence["scope_complete"] = False
    equivalence["audits_complete"] = False
    equivalence["decision"] = "diagnostic_subset_no_eligibility_decision"
    equivalence_path.write_text(json.dumps(equivalence), encoding="utf-8")
    with pytest.raises(DataFreezeError, match="diagnostic-only"):
        create_freeze_bundle(
            generated_root=root,
            equivalence_path=equivalence_path,
            output_path=tmp_path / "diagnostic-freeze.json",
            frozen=frozen,
            fixture=True,
        )


def test_schema_1_1_freezes_missing_legacy_quality_only_as_mandatory_rerun(
    tmp_path: Path,
):
    root, _, frozen, _ = _frozen_fixture(tmp_path)
    equivalence_path = root / "equivalence.json"
    suite = _use_absent_legacy_quality_option(
        json.loads(equivalence_path.read_text(encoding="utf-8"))
    )
    equivalence_path.write_text(json.dumps(suite), encoding="utf-8")
    bundle = create_freeze_bundle(
        generated_root=root,
        equivalence_path=equivalence_path,
        output_path=tmp_path / "option-1-freeze.json",
        frozen=frozen,
        fixture=True,
    )
    assert bundle["equivalence"]["decision"] == (
        "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
    )
    assert bundle["equivalence"]["reports"]["scidocs"][
        "quality_comparison_status"
    ] == "not_comparable_no_legacy_quality_artifact"


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("legacy", "exists", True, "absence is unproven"),
        ("legacy", "status", "missing", "absence is unproven"),
        ("legacy", "fallback_main_query_count", 0, "absence is unproven"),
        ("generated", "authoritative", False, "coverage is incomplete"),
        ("generated", "full_qrels_coverage", False, "coverage is incomplete"),
        ("generated", "qrels_query_count", 1, "coverage is incomplete"),
    ],
)
def test_schema_1_1_rejects_tampered_quality_absence_or_coverage(
    tmp_path: Path, section: str, field: str, value, match: str
):
    root, _, frozen, _ = _frozen_fixture(tmp_path)
    equivalence_path = root / "equivalence.json"
    suite = _use_absent_legacy_quality_option(
        json.loads(equivalence_path.read_text(encoding="utf-8"))
    )
    report = next(row for row in suite["reports"] if row["dataset"] == "scidocs")
    report["quality_artifacts"][section][field] = value
    equivalence_path.write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(DataFreezeError, match=match):
        create_freeze_bundle(
            generated_root=root,
            equivalence_path=equivalence_path,
            output_path=tmp_path / "tampered-option-1-freeze.json",
            frozen=frozen,
            fixture=True,
        )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("rankings", "quality", "status"), "not_comparable", "ranking status"),
        (("qrels", "metrics_status"), "not_comparable", "qrels contract"),
        (("qrels", "semantic_equal"), False, "qrels contract"),
        (("representative_accounting", "status"), "not_comparable", "accounting"),
    ],
)
def test_schema_1_1_rejects_generic_noncomparable_or_incomplete_cross_checks(
    tmp_path: Path, path: tuple[str, ...], value, match: str
):
    root, _, frozen, _ = _frozen_fixture(tmp_path)
    equivalence_path = root / "equivalence.json"
    suite = _use_absent_legacy_quality_option(
        json.loads(equivalence_path.read_text(encoding="utf-8"))
    )
    report = next(row for row in suite["reports"] if row["dataset"] == "scidocs")
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    equivalence_path.write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(DataFreezeError, match=match):
        create_freeze_bundle(
            generated_root=root,
            equivalence_path=equivalence_path,
            output_path=tmp_path / "invalid-option-1-freeze.json",
            frozen=frozen,
            fixture=True,
        )
