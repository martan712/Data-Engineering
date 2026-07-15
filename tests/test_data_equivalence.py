from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from bondmaxsim.data.config import DATASETS, load_data_configuration
from bondmaxsim.data.equivalence import aggregate_equivalence_reports, compare_dataset
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
        fixture=True,
    )
    _legacy_from_generated(generated, legacy)
    report = compare_dataset(
        "scifact",
        legacy_root=legacy,
        generated_root=generated,
        frozen=frozen,
        k=2,
        fixture=True,
    )
    assert report["bitwise_equivalent"] is True
    assert report["audit_complete"] is True
    assert report["final_eligibility"] is False
    assert report["rankings"]["mechanism"]["topk_set_equal_count"] == 2
    assert report["rankings"]["mechanism"]["boundary_tie_complete"] is True
    assert report["qrels"]["status"] == "complete"
    assert report["qrels"]["metrics_equal"] is True
    assert report["representative_accounting"]["equal"] is True
    assert report["decision"] == (
        "bitwise_equivalent_dataset_audit_complete_aggregate_required"
    )
    assert report["arrays"]["documents"]["difference_counts"]["gt_0e+00"] == 0
    assert report["arrays"]["documents"]["difference_quantiles"]["p99"] == 0.0


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
        fixture=True,
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
        fixture=True,
    )
    assert report["bitwise_equivalent"] is False
    assert report["arrays"]["documents"]["max_abs_difference"] == 1.0
    assert report["decision"] == "not_bitwise_equivalent_rerun_all_data_dependent_evidence"


def test_sampled_dataset_audit_never_emits_eligibility(tmp_path: Path):
    generated = tmp_path / "generated"
    legacy = tmp_path / "legacy"
    frozen = load_data_configuration()
    generate_dataset(
        "scifact",
        output_root=generated,
        frozen=frozen,
        source_loader=_source,
        encoder=_Encoder(),
        fixture=True,
    )
    _legacy_from_generated(generated, legacy)

    report = compare_dataset(
        "scifact",
        legacy_root=legacy,
        generated_root=generated,
        frozen=frozen,
        k=2,
        topk_limit=1,
        fixture=True,
    )

    assert report["bitwise_equivalent"] is True
    assert report["audit_complete"] is False
    assert report["decision"] == "diagnostic_incomplete_no_eligibility_decision"


def test_accounting_mismatch_blocks_bitwise_eligibility(tmp_path: Path):
    generated = tmp_path / "generated"
    legacy = tmp_path / "legacy"
    frozen = load_data_configuration()
    generate_dataset(
        "scifact",
        output_root=generated,
        frozen=frozen,
        source_loader=_source,
        encoder=_Encoder(),
        fixture=True,
    )
    _legacy_from_generated(generated, legacy)

    report = compare_dataset(
        "scifact",
        legacy_root=legacy,
        generated_root=generated,
        frozen=frozen,
        k=2,
        representative_accounting=lambda root, _: {"counter": int(root == generated)},
        fixture=True,
    )

    assert report["bitwise_equivalent"] is True
    assert report["audit_complete"] is True
    assert report["representative_accounting"]["equal"] is False
    assert report["final_eligibility"] is False
    assert report["decision"] == "audit_inconsistency_stop_no_eligibility_decision"


def test_qrels_semantic_or_metric_change_forces_rerun(tmp_path: Path):
    generated = tmp_path / "generated"
    legacy = tmp_path / "legacy"
    frozen = load_data_configuration()
    generate_dataset(
        "scifact",
        output_root=generated,
        frozen=frozen,
        source_loader=_source,
        encoder=_Encoder(),
        fixture=True,
    )
    _legacy_from_generated(generated, legacy)
    qrels_path = legacy / "qrels/scifact.tsv"
    qrels_path.write_text(
        qrels_path.read_text(encoding="utf-8").replace("q0\td0\t1", "q0\td1\t1"),
        encoding="utf-8",
    )

    report = compare_dataset(
        "scifact",
        legacy_root=legacy,
        generated_root=generated,
        frozen=frozen,
        k=2,
        fixture=True,
    )

    assert report["qrels"]["status"] == "complete"
    assert report["qrels"]["semantic_equal"] is False
    assert report["qrels"]["legacy_metrics"] != report["qrels"]["generated_metrics"]
    assert report["bitwise_equivalent"] is False
    assert report["decision"] == (
        "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
    )


def _complete_report(dataset: str, *, bitwise: bool = True, complete: bool = True):
    return {
        "dataset": dataset,
        "bitwise_equivalent": bitwise,
        "audit_complete": complete,
        "cross_checks_equal": bitwise and complete,
    }


def test_aggregate_requires_all_four_final_datasets_and_complete_audits():
    complete = [_complete_report(dataset) for dataset in DATASETS]
    final = aggregate_equivalence_reports(complete, DATASETS)
    assert final["scope_complete"] is True
    assert final["final_eligibility"] is True
    assert final["decision"] == (
        "bitwise_equivalent_non_timing_evidence_may_remain_eligible"
    )

    subset = aggregate_equivalence_reports(complete[:1], DATASETS[:1])
    assert subset["scope_complete"] is False
    assert subset["final_eligibility"] is False
    assert subset["decision"] == "diagnostic_subset_no_eligibility_decision"

    incomplete = list(complete)
    incomplete[-1] = _complete_report(DATASETS[-1], complete=False)
    diagnostic = aggregate_equivalence_reports(incomplete, DATASETS)
    assert diagnostic["final_eligibility"] is False
    assert diagnostic["decision"] == "diagnostic_incomplete_no_eligibility_decision"


def test_full_aggregate_non_bitwise_result_forces_rerun():
    reports = [_complete_report(dataset) for dataset in DATASETS]
    reports[0] = _complete_report(DATASETS[0], bitwise=False)
    aggregate = aggregate_equivalence_reports(reports, DATASETS)
    assert aggregate["final_eligibility"] is False
    assert aggregate["decision"] == (
        "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
    )
