from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bondmaxsim.data.config import load_data_configuration
from bondmaxsim.data.generation import (
    DataGenerationError,
    PublicSource,
    generate_dataset,
    refreeze_generated_dataset,
    validate_generated_dataset,
)


class _FixtureEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, is_query, **kwargs):
        self.calls.append({"is_query": is_query, **kwargs})
        rows = []
        for index, text in enumerate(texts):
            count = 2 + ((len(text) + index + int(is_query)) % 3)
            matrix = np.zeros((count, 128), dtype=np.float32)
            matrix[:, (index + int(is_query)) % 128] = 1.0
            rows.append(matrix)
        return rows


class _FailEncoder:
    def encode(self, *_args, **_kwargs):
        pytest.fail("manifest refreeze invoked the encoder")


def _source(dataset, frozen):
    documents = tuple(f"d{index}" for index in range(6))
    queries = tuple(f"q{index:02d}" for index in range(60))
    return PublicSource(
        documents,
        tuple(f"document {index}" for index in range(6)),
        queries,
        tuple(f"query {index}" for index in range(60)),
        {"q01": {"d1": 1}, "q03": {"d4": 2}},
    )


def _generate(tmp_path: Path, *, encoder=None):
    frozen = load_data_configuration()
    result = generate_dataset(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        source_loader=_source,
        encoder=encoder or _FixtureEncoder(),
        fixture=True,
    )
    return frozen, result


def _write_manifest(path: Path, manifest):
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_dataset_generation_freezes_settings_selection_and_is_resumable(tmp_path: Path):
    encoder = _FixtureEncoder()
    frozen, result = _generate(tmp_path, encoder=encoder)
    assert result.resumed is False
    manifest = validate_generated_dataset(
        tmp_path, "scifact", frozen=frozen, fixture=True
    )
    assert manifest["validation_contract"] == "strict-v2"
    assert manifest["dataset"] == "scifact"
    assert manifest["sources"]["corpus_queries_revision"] == frozen.dataset("scifact")[
        "corpus_queries_revision"
    ]
    assert manifest["encoder_settings"]["precision"] == "float32"
    assert manifest["counts"] == {
        "documents": 6,
        "source_queries": 60,
        "mechanism_population_queries": 60,
        "mechanism_selected_queries": 50,
        "quality_queries": 2,
        "qrels_queries": 2,
        "qrels_rows": 2,
    }
    expected_indices = sorted(
        np.random.default_rng(42).choice(60, size=50, replace=False).tolist()
    )
    mechanism = manifest["workloads"]["mechanism"]
    assert mechanism["selection_seed"] == 42
    assert mechanism["selected_source_indices"] == expected_indices
    assert mechanism["selected_query_ids"] == [f"q{index:02d}" for index in expected_indices]
    assert mechanism["token_statistics"]["count"] == 50
    assert mechanism["population_token_statistics"]["count"] == 60
    assert [call["is_query"] for call in encoder.calls] == [False, True, True]
    assert all(call["precision"] == "float32" for call in encoder.calls)
    assert all(call["padding"] is False for call in encoder.calls)
    assert all(call["normalize_embeddings"] is True for call in encoder.calls)

    with np.load(tmp_path / "embeddings/scifact_test_queries.npz") as values:
        assert values["query_ids"].tolist() == ["q01", "q03"]
    resumed = generate_dataset(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        source_loader=lambda *_: pytest.fail("resume fetched sources"),
        encoder=_FailEncoder(),
        fixture=True,
    )
    assert resumed.resumed is True
    assert not list(tmp_path.rglob("*.tmp"))


def test_fixture_mode_is_explicit_and_real_counts_fail_before_encoding(tmp_path: Path):
    frozen = load_data_configuration()
    with pytest.raises(DataGenerationError, match="pinned source counts"):
        generate_dataset(
            "scifact",
            output_root=tmp_path,
            frozen=frozen,
            source_loader=_source,
            encoder=_FailEncoder(),
        )


@pytest.mark.parametrize("field", ["dataset", "sources", "counts", "workloads"])
def test_validation_recomputes_manifest_identity_and_statistics(tmp_path: Path, field: str):
    frozen, result = _generate(tmp_path)
    manifest = dict(result.manifest)
    manifest[field] = "tampered"
    _write_manifest(result.manifest_path, manifest)
    with pytest.raises(DataGenerationError, match="manifest does not match"):
        validate_generated_dataset(tmp_path, "scifact", frozen=frozen, fixture=True)


@pytest.mark.parametrize("mutation", ["extra", "size", "hash"])
def test_validation_requires_exact_output_set_size_and_hash(tmp_path: Path, mutation: str):
    frozen, result = _generate(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        manifest["outputs"]["embeddings/extra.npz"] = {"bytes": 1, "sha256": "0" * 64}
    else:
        record = manifest["outputs"]["beir_ids/scifact_ids.json"]
        record["bytes" if mutation == "size" else "sha256"] = 0 if mutation == "size" else "0" * 64
    _write_manifest(result.manifest_path, manifest)
    with pytest.raises(DataGenerationError, match="manifest does not match"):
        validate_generated_dataset(tmp_path, "scifact", frozen=frozen, fixture=True)


@pytest.mark.parametrize(
    "field,values,match",
    [
        ("corpus_ids", ["d0", "d0", "d2", "d3", "d4", "d5"], "duplicate IDs"),
        ("mechanism_query_ids", [f"q{index:02d}" for index in range(1, 60)], "source prefix"),
        ("quality_query_ids", ["q03", "q01"], "quality query ID/order"),
    ],
)
def test_validation_rejects_duplicate_or_reordered_ids(
    tmp_path: Path, field: str, values: list[str], match: str
):
    frozen, _ = _generate(tmp_path)
    ids_path = tmp_path / "beir_ids/scifact_ids.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    ids[field] = values
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    with pytest.raises(DataGenerationError, match=match):
        validate_generated_dataset(tmp_path, "scifact", frozen=frozen, fixture=True)


@pytest.mark.parametrize(
    "replacement,match",
    [
        (
            "query-id\tcorpus-id\tscore\nq01\td1\t1\nq01\td1\t1\nq03\td4\t2\n",
            "duplicate qrels row",
        ),
        (
            "query-id\tcorpus-id\tscore\nq99\td1\t1\nq03\td4\t2\n",
            "unknown qrels query",
        ),
        (
            "query-id\tcorpus-id\tscore\nq01\td1\t0\nq03\td4\t2\n",
            "canonical positive integer",
        ),
    ],
)
def test_validation_rejects_invalid_qrels(tmp_path: Path, replacement: str, match: str):
    frozen, _ = _generate(tmp_path)
    (tmp_path / "qrels/scifact.tsv").write_text(replacement, encoding="utf-8")
    with pytest.raises(DataGenerationError, match=match):
        validate_generated_dataset(tmp_path, "scifact", frozen=frozen, fixture=True)


def _dangling_source(dataset, frozen):
    documents = tuple(f"d{index}" for index in range(6))
    queries = tuple(f"q{index:02d}" for index in range(60))
    return PublicSource(
        documents,
        tuple(f"document {index}" for index in range(6)),
        queries,
        tuple(f"query {index}" for index in range(60)),
        # q03 judges a document absent from the corpus, as BeIR arguana does.
        {"q01": {"d1": 1}, "q03": {"absent-doc": 2}},
    )


def test_dangling_qrels_documents_are_retained_and_counted(tmp_path: Path):
    frozen = load_data_configuration()
    generate_dataset(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        source_loader=_dangling_source,
        encoder=_FixtureEncoder(),
        fixture=True,
    )
    manifest = validate_generated_dataset(
        tmp_path, "scifact", frozen=frozen, fixture=True
    )
    assert manifest["counts"]["qrels_dangling_documents"] == 1
    assert manifest["counts"]["qrels_rows"] == 2
    assert manifest["counts"]["qrels_queries"] == 2
    # The dangling judgment is retained verbatim in the written qrels file.
    assert "absent-doc" in (tmp_path / "qrels/scifact.tsv").read_text(encoding="utf-8")


def test_completed_legacy_manifest_refreezes_without_encoding(tmp_path: Path):
    frozen, result = _generate(tmp_path)
    legacy = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    legacy.pop("validation_contract")
    legacy.pop("encoder_settings")
    _write_manifest(result.manifest_path, legacy)

    resumed = generate_dataset(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        source_loader=lambda *_: pytest.fail("refreeze fetched sources"),
        encoder=_FailEncoder(),
        fixture=True,
    )
    assert resumed.resumed is True
    assert resumed.manifest["validation_contract"] == "strict-v2"
    assert resumed.manifest["workloads"]["mechanism"]["selected_query_ids"]


def test_refreeze_requires_legacy_source_and_output_integrity(tmp_path: Path):
    frozen, result = _generate(tmp_path)
    legacy = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    legacy.pop("validation_contract")
    legacy["outputs"]["beir_ids/scifact_ids.json"]["bytes"] += 1
    _write_manifest(result.manifest_path, legacy)
    with pytest.raises(DataGenerationError, match="checksum/size mismatch"):
        refreeze_generated_dataset(tmp_path, "scifact", frozen=frozen, fixture=True)
