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
    validate_generated_dataset,
)


class _FixtureEncoder:
    def encode(self, texts, *, is_query, **kwargs):
        rows = []
        for index, text in enumerate(texts):
            count = 2 + ((len(text) + index + int(is_query)) % 3)
            matrix = np.zeros((count, 128), dtype=np.float32)
            matrix[:, (index + int(is_query)) % 128] = 1.0
            rows.append(matrix)
        return rows


def _source(dataset, frozen):
    documents = tuple(f"d{index}" for index in range(6))
    queries = tuple(f"q{index}" for index in range(5))
    return PublicSource(
        documents,
        tuple(f"document {index}" for index in range(6)),
        queries,
        tuple(f"query {index}" for index in range(5)),
        {"q1": {"d1": 1}, "q3": {"d4": 2}},
    )


def test_dataset_generation_is_atomic_validated_and_resumable(tmp_path: Path):
    frozen = load_data_configuration()
    result = generate_dataset(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        source_loader=_source,
        encoder=_FixtureEncoder(),
    )
    assert result.resumed is False
    manifest = validate_generated_dataset(tmp_path, "scifact", frozen=frozen)
    assert manifest["counts"] == {
        "documents": 6,
        "source_queries": 5,
        "mechanism_population_queries": 5,
        "quality_queries": 2,
        "qrels_queries": 2,
    }
    with np.load(tmp_path / "embeddings/scifact_test_queries.npz") as values:
        assert values["query_ids"].tolist() == ["q1", "q3"]
    resumed = generate_dataset(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        source_loader=lambda *_: pytest.fail("resume fetched sources"),
        encoder=_FixtureEncoder(),
    )
    assert resumed.resumed is True
    assert not list(tmp_path.rglob("*.tmp"))


def test_validation_rejects_modified_output(tmp_path: Path):
    frozen = load_data_configuration()
    generate_dataset(
        "scifact",
        output_root=tmp_path,
        frozen=frozen,
        source_loader=_source,
        encoder=_FixtureEncoder(),
    )
    ids_path = tmp_path / "beir_ids/scifact_ids.json"
    ids_path.write_text(json.dumps({"corpus_ids": []}), encoding="utf-8")
    with pytest.raises(DataGenerationError, match="checksum mismatch"):
        validate_generated_dataset(tmp_path, "scifact", frozen=frozen)
