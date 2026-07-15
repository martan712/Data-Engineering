from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from bondmaxsim.data.config import load_data_configuration
from bondmaxsim.data.equivalence import compare_dataset
from bondmaxsim.data.generation import PublicSource, generate_dataset


class _Encoder:
    def encode(self, texts, *, is_query, **kwargs):
        rows = []
        for index, _ in enumerate(texts):
            values = np.zeros((2, 128), dtype=np.float32)
            values[:, (index + int(is_query)) % 128] = 1.0
            rows.append(values)
        return rows


def _source(dataset, frozen):
    return PublicSource(
        ("d0", "d1", "d2"),
        ("a", "b", "c"),
        ("q0", "q1"),
        ("x", "y"),
        {"q0": {"d0": 1}, "q1": {"d2": 1}},
    )


def _legacy_from_generated(generated: Path, legacy: Path) -> None:
    for directory in ("embeddings", "beir_ids", "qrels"):
        shutil.copytree(generated / directory, legacy / directory)
    sidecar = legacy / "beir_ids/scifact_ids.json"
    ids = json.loads(sidecar.read_text())
    ids.pop("mechanism_query_ids")
    ids.pop("quality_query_ids")
    sidecar.write_text(json.dumps(ids), encoding="utf-8")


def test_bitwise_equivalent_fixture_keeps_non_timing_eligibility(tmp_path: Path):
    generated = tmp_path / "generated"
    legacy = tmp_path / "legacy"
    frozen = load_data_configuration()
    generate_dataset(
        "scifact",
        output_root=generated,
        frozen=frozen,
        source_loader=_source,
        encoder=_Encoder(),
    )
    _legacy_from_generated(generated, legacy)
    report = compare_dataset(
        "scifact",
        legacy_root=legacy,
        generated_root=generated,
        frozen=frozen,
        k=2,
    )
    assert report["bitwise_equivalent"] is True
    assert report["rankings"]["mechanism"]["topk_set_equal_count"] == 2
    assert report["decision"].startswith("bitwise_equivalent")


def test_one_embedding_change_forces_full_data_dependent_rerun(tmp_path: Path):
    generated = tmp_path / "generated"
    legacy = tmp_path / "legacy"
    frozen = load_data_configuration()
    generate_dataset(
        "scifact",
        output_root=generated,
        frozen=frozen,
        source_loader=_source,
        encoder=_Encoder(),
    )
    _legacy_from_generated(generated, legacy)
    path = legacy / "embeddings/scifact.npz"
    with np.load(path) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays["doc_values"][0, 0] = 0.0
    arrays["doc_values"][0, 1] = 1.0
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
    report = compare_dataset(
        "scifact",
        legacy_root=legacy,
        generated_root=generated,
        frozen=frozen,
        k=2,
    )
    assert report["bitwise_equivalent"] is False
    assert report["arrays"]["documents"]["max_abs_difference"] == 1.0
    assert report["decision"] == "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
