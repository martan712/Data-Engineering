from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bondmaxsim.data.config import load_data_configuration
from bondmaxsim.data.generation import PublicSource, generate_dataset
from bondmaxsim.data.indexes import (
    DerivedIndexError,
    build_derived_indexes,
    effective_index_specs,
    validate_derived_indexes,
)


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
    del dataset, frozen
    return PublicSource(
        tuple(f"d{index}" for index in range(6)),
        tuple(f"document {index}" for index in range(6)),
        tuple(f"q{index}" for index in range(5)),
        tuple(f"query {index}" for index in range(5)),
        {"q1": {"d1": 1}, "q3": {"d4": 2}},
    )


def _fixture_data(root: Path):
    frozen = load_data_configuration()
    generate_dataset(
        "scifact",
        output_root=root,
        frozen=frozen,
        source_loader=_source,
        encoder=_FixtureEncoder(),
        fixture=True,
    )
    return frozen


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
    first = destination / "metadata.json"
    second = destination / "fast_plaid_index" / "index.bin"
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(payload)
    second.write_bytes(payload[::-1])
    return (second, first)


_FAKE_BUILDERS = {"faiss": _fake_builder, "plaid": _fake_builder}


def test_effective_settings_freeze_adapter_defaults():
    frozen = load_data_configuration()
    faiss, plaid = effective_index_specs(frozen, total_tokens=4096)
    assert faiss.backend == "faiss"
    assert faiss.settings == {
        "implementation": "faiss.IndexIVFFlat",
        "metric": "inner_product",
        "seed": 42,
        "n_lists_policy": "power_of_two_nearest_4sqrt_total_tokens",
        "n_lists": 256,
        "nprobe": 32,
        "kmeans_niters": 10,
        "train_points_per_centroid": 64,
    }
    assert plaid.settings["implementation"] == "pylate-1.6.0-public-api"
    assert plaid.settings["seed"] == 42
    assert plaid.settings["nbits"] == 4
    assert plaid.settings["kmeans_niters"] == 4
    assert plaid.settings["n_ivf_probe"] == 8


def test_fixture_indexes_record_source_settings_artifacts_and_resume(tmp_path: Path):
    frozen = _fixture_data(tmp_path)
    result = build_derived_indexes(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        builders=_FAKE_BUILDERS,
        fixture=True,
    )
    assert result.resumed is False
    manifest = validate_derived_indexes(
        tmp_path, "scifact", frozen=frozen, fixture=True
    )
    assert manifest["configuration_sha256"] == frozen.sha256
    assert set(manifest["source_data"]["outputs"]) == {
        "beir_ids/scifact_ids.json",
        "embeddings/scifact.npz",
        "embeddings/scifact_test_queries.npz",
        "qrels/scifact.tsv",
    }
    assert set(manifest["indexes"]) == {"faiss", "plaid"}
    assert len(manifest["indexes"]["faiss"]["artifacts"]) == 1
    assert len(manifest["indexes"]["plaid"]["artifacts"]) == 2
    resumed = build_derived_indexes(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        builders={
            "faiss": lambda *_: pytest.fail("resume rebuilt FAISS"),
            "plaid": lambda *_: pytest.fail("resume rebuilt PLAID"),
        },
        fixture=True,
    )
    assert resumed.resumed is True
    assert not list(tmp_path.rglob("*.tmp"))


def test_validation_rejects_modified_index_artifact(tmp_path: Path):
    frozen = _fixture_data(tmp_path)
    result = build_derived_indexes(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        builders=_FAKE_BUILDERS,
        fixture=True,
    )
    artifact = result.manifest["indexes"]["faiss"]["artifacts"][0]
    (tmp_path / artifact["path"]).write_bytes(b"modified")
    with pytest.raises(DerivedIndexError, match="checksum mismatch"):
        validate_derived_indexes(tmp_path, "scifact", frozen=frozen, fixture=True)


def test_validation_rejects_changed_source_manifest_identity(tmp_path: Path):
    frozen = _fixture_data(tmp_path)
    build_derived_indexes(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        builders=_FAKE_BUILDERS,
        fixture=True,
    )
    data_manifest = tmp_path / "manifests/scifact.json"
    data_manifest.write_text(data_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DerivedIndexError, match="source data hash mismatch"):
        validate_derived_indexes(tmp_path, "scifact", frozen=frozen, fixture=True)


def test_validation_rejects_changed_frozen_index_settings(tmp_path: Path):
    frozen = _fixture_data(tmp_path)
    result = build_derived_indexes(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        builders=_FAKE_BUILDERS,
        fixture=True,
    )
    manifest = dict(result.manifest)
    manifest["indexes"]["faiss"]["settings"]["seed"] = 7
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DerivedIndexError, match="faiss settings mismatch"):
        validate_derived_indexes(tmp_path, "scifact", frozen=frozen, fixture=True)


def test_validation_rejects_index_shape_detached_from_source(tmp_path: Path):
    frozen = _fixture_data(tmp_path)
    result = build_derived_indexes(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        builders=_FAKE_BUILDERS,
        fixture=True,
    )
    manifest = dict(result.manifest)
    manifest["document_shape"] = [4096, 128]
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DerivedIndexError, match="document shape/source mismatch"):
        validate_derived_indexes(tmp_path, "scifact", frozen=frozen, fixture=True)
