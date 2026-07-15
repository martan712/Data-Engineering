"""Authoritative four-dataset data/index freeze and result expectations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.config import DATASETS, FrozenDataConfiguration, load_data_configuration
from bondmaxsim.data.generation import validate_generated_dataset
from bondmaxsim.data.indexes import validate_derived_indexes


DEFAULT_GENERATED_ROOT = REPO_ROOT / "data" / "generated" / "v1"
DEFAULT_FREEZE_MANIFEST = REPO_ROOT / "artifacts" / "data" / "freeze-v1.json"
_DECISION_EQUIVALENT = "bitwise_equivalent_non_timing_evidence_may_remain_eligible"
_DECISION_RERUN = "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
_DECISION_DATASET_EQUIVALENT = (
    "bitwise_equivalent_dataset_audit_complete_aggregate_required"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DataFreezeError(RuntimeError):
    """The final data/index freeze is incomplete or no longer matches disk."""


@dataclass(frozen=True)
class FrozenDatasetInputs:
    dataset: str
    generated_root: Path
    embedding_path: Path
    quality_queries_path: Path
    ids_path: Path
    qrels_path: Path
    workload_ids: Mapping[str, str]
    data_configuration_sha256: str
    data_sha256: str
    index_sha256_by_backend: Mapping[str, str]


def _canonical_sha256(value: Any) -> str:
    content = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataFreezeError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise DataFreezeError(f"{label} {path} must contain an object")
    return document


def _relative(path: Path, root: Path, label: str) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise DataFreezeError(f"{label} must be inside generated root: {path}") from error


def _equivalence_record(
    generated_root: Path,
    equivalence_path: Path,
    frozen: FrozenDataConfiguration,
) -> tuple[dict[str, Any], Mapping[str, Mapping[str, Any]]]:
    suite = _read_json(equivalence_path, "equivalence suite")
    if (
        suite.get("schema_name") != "bondmaxsim.data-equivalence-suite"
        or suite.get("schema_version") != "1.0.0"
    ):
        raise DataFreezeError("unsupported equivalence suite")
    reports = suite.get("reports")
    if not isinstance(reports, list) or len(reports) != len(DATASETS):
        raise DataFreezeError("equivalence suite must contain exactly four reports")
    selected_datasets = suite.get("selected_datasets")
    if (
        not isinstance(selected_datasets, list)
        or len(selected_datasets) != len(DATASETS)
        or any(not isinstance(value, str) for value in selected_datasets)
        or set(selected_datasets) != set(DATASETS)
        or suite.get("scope_complete") is not True
        or suite.get("audits_complete") is not True
    ):
        raise DataFreezeError(
            "equivalence suite is diagnostic-only, not a complete four-dataset audit"
        )
    by_dataset: dict[str, Mapping[str, Any]] = {}
    report_records: dict[str, Any] = {}
    for report in reports:
        if not isinstance(report, Mapping):
            raise DataFreezeError("equivalence report must be an object")
        dataset = report.get("dataset")
        if dataset not in DATASETS or dataset in by_dataset:
            raise DataFreezeError("equivalence reports must cover each frozen dataset once")
        if (
            report.get("schema_name") != "bondmaxsim.data-equivalence"
            or report.get("schema_version") != "1.0.0"
            or report.get("configuration_sha256") != frozen.sha256
        ):
            raise DataFreezeError(f"{dataset}: invalid equivalence report identity")
        if not isinstance(report.get("bitwise_equivalent"), bool):
            raise DataFreezeError(f"{dataset}: equivalence decision flag must be boolean")
        if report.get("audit_complete") is not True:
            raise DataFreezeError(f"{dataset}: diagnostic equivalence report cannot be frozen")
        if not isinstance(report.get("cross_checks_equal"), bool):
            raise DataFreezeError(f"{dataset}: cross-check decision flag must be boolean")
        if report.get("final_eligibility") is not False:
            raise DataFreezeError(f"{dataset}: dataset report cannot issue final eligibility")
        expected_decision = (
            _DECISION_DATASET_EQUIVALENT
            if report.get("bitwise_equivalent") is True
            and report.get("cross_checks_equal") is True
            else _DECISION_RERUN
        )
        if report.get("decision") != expected_decision:
            raise DataFreezeError(f"{dataset}: inconsistent equivalence decision")
        rankings = report.get("rankings")
        if not isinstance(rankings, Mapping) or set(rankings) != {"mechanism", "quality"}:
            raise DataFreezeError(f"{dataset}: incomplete equivalence rankings")
        for row in rankings.values():
            if not isinstance(row, Mapping) or row.get("status") != "complete":
                if isinstance(row, Mapping) and row.get("status") == "sampled":
                    raise DataFreezeError(
                        f"{dataset}: sampled equivalence audit cannot be frozen"
                    )
                raise DataFreezeError(f"{dataset}: invalid equivalence ranking status")
            if row.get("boundary_tie_complete") is not True:
                raise DataFreezeError(f"{dataset}: boundary-tie audit is incomplete")
        qrels = report.get("qrels")
        accounting = report.get("representative_accounting")
        if not isinstance(qrels, Mapping) or qrels.get("status") != "complete":
            raise DataFreezeError(f"{dataset}: qrels equivalence audit is incomplete")
        if not isinstance(accounting, Mapping) or accounting.get("status") != "complete":
            raise DataFreezeError(f"{dataset}: accounting equivalence audit is incomplete")
        expected_cross_checks = bool(
            qrels.get("semantic_equal") is True
            and qrels.get("metrics_equal") is True
            and accounting.get("equal") is True
        )
        if report.get("cross_checks_equal") is not expected_cross_checks:
            raise DataFreezeError(f"{dataset}: cross-check flags are inconsistent")
        by_dataset[str(dataset)] = report
        report_records[str(dataset)] = {
            "bitwise_equivalent": bool(report["bitwise_equivalent"]),
            "decision": expected_decision,
            "report_sha256": _canonical_sha256(report),
        }
    if set(by_dataset) != set(DATASETS):
        raise DataFreezeError("equivalence reports do not cover the four frozen datasets")
    expected_bitwise = all(
        report["bitwise_equivalent"] is True for report in by_dataset.values()
    )
    expected_cross_checks = all(
        report["cross_checks_equal"] is True for report in by_dataset.values()
    )
    expected_final_eligibility = expected_bitwise and expected_cross_checks
    expected_suite_decision = (
        _DECISION_EQUIVALENT if expected_final_eligibility else _DECISION_RERUN
    )
    if (
        suite.get("bitwise_equivalent") is not expected_bitwise
        or suite.get("cross_checks_equal") is not expected_cross_checks
        or suite.get("final_eligibility") is not expected_final_eligibility
    ):
        raise DataFreezeError("aggregate equivalence eligibility flags are inconsistent")
    if suite.get("decision") != expected_suite_decision:
        raise DataFreezeError("aggregate equivalence decision is inconsistent")
    return (
        {
            "path": _relative(equivalence_path, generated_root, "equivalence suite"),
            "sha256": _file_sha256(equivalence_path),
            "decision": expected_suite_decision,
            "reports": report_records,
        },
        by_dataset,
    )


def _dataset_record(
    generated_root: Path,
    dataset: str,
    frozen: FrozenDataConfiguration,
    equivalence_report: Mapping[str, Any],
    *,
    fixture: bool,
) -> dict[str, Any]:
    data_manifest = validate_generated_dataset(
        generated_root, dataset, frozen=frozen, fixture=fixture
    )
    index_manifest = validate_derived_indexes(
        generated_root, dataset, frozen=frozen, fixture=fixture
    )
    data_manifest_path = generated_root / "manifests" / f"{dataset}.json"
    index_manifest_path = generated_root / "index_manifests" / f"{dataset}.json"
    files = {
        relative: {"sha256": row["sha256"], "bytes": int(row["bytes"])}
        for relative, row in sorted(data_manifest["outputs"].items())
    }
    workloads = {
        regime: {
            "workload_id": row["workload_id"],
            "sha256": _canonical_sha256(row),
        }
        for regime, row in sorted(data_manifest["workloads"].items())
    }
    data_identity = {
        "configuration_sha256": frozen.sha256,
        "data_manifest_sha256": _file_sha256(data_manifest_path),
        "files": files,
        "workloads": workloads,
    }
    indexes: dict[str, Any] = {}
    for backend, row in sorted(index_manifest["indexes"].items()):
        index_identity = {
            "source_data": index_manifest["source_data"],
            "settings": row["settings"],
            "artifacts": row["artifacts"],
        }
        indexes[backend] = {
            "settings": row["settings"],
            "artifacts": row["artifacts"],
            "index_sha256": _canonical_sha256(index_identity),
        }
    return {
        "data_manifest": {
            "path": _relative(data_manifest_path, generated_root, "data manifest"),
            "sha256": data_identity["data_manifest_sha256"],
        },
        "files": files,
        "workloads": workloads,
        "data_sha256": _canonical_sha256(data_identity),
        "index_manifest": {
            "path": _relative(index_manifest_path, generated_root, "index manifest"),
            "sha256": _file_sha256(index_manifest_path),
        },
        "indexes": indexes,
        "equivalence": {
            "bitwise_equivalent": bool(equivalence_report["bitwise_equivalent"]),
            "decision": equivalence_report["decision"],
            "report_sha256": _canonical_sha256(equivalence_report),
        },
    }


def _build_freeze_document(
    generated_root: Path,
    equivalence_path: Path,
    frozen: FrozenDataConfiguration,
    *,
    fixture: bool,
) -> dict[str, Any]:
    equivalence, reports = _equivalence_record(generated_root, equivalence_path, frozen)
    datasets = {
        dataset: _dataset_record(
            generated_root,
            dataset,
            frozen,
            reports[dataset],
            fixture=fixture,
        )
        for dataset in DATASETS
    }
    return {
        "schema_name": "bondmaxsim.data-freeze",
        "schema_version": "1.0.0",
        "layout": "data/generated/v1",
        "fixture": fixture,
        "configuration": {
            "path": str(frozen.path.resolve().relative_to(REPO_ROOT.resolve())),
            "sha256": frozen.sha256,
        },
        "equivalence": equivalence,
        "datasets": datasets,
    }


def create_freeze_bundle(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    equivalence_path: Path | None = None,
    output_path: Path = DEFAULT_FREEZE_MANIFEST,
    frozen: FrozenDataConfiguration | None = None,
    fixture: bool = False,
) -> Mapping[str, Any]:
    """Validate all four datasets/indexes and atomically write their freeze."""
    frozen = frozen or load_data_configuration()
    generated_root = Path(generated_root)
    equivalence_path = equivalence_path or generated_root / "equivalence.json"
    document = _build_freeze_document(
        generated_root, equivalence_path, frozen, fixture=fixture
    )
    _atomic_json(output_path, document)
    return validate_freeze_bundle(
        output_path,
        generated_root=generated_root,
        frozen=frozen,
        fixture=fixture,
    )


def validate_freeze_bundle(
    freeze_path: Path = DEFAULT_FREEZE_MANIFEST,
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    frozen: FrozenDataConfiguration | None = None,
    fixture: bool = False,
) -> Mapping[str, Any]:
    """Fail closed if any frozen configuration, data, index, or decision drifts."""
    frozen = frozen or load_data_configuration()
    freeze_path = Path(freeze_path)
    generated_root = Path(generated_root)
    actual = _read_json(freeze_path, "data freeze")
    if (
        actual.get("schema_name") != "bondmaxsim.data-freeze"
        or actual.get("schema_version") != "1.0.0"
    ):
        raise DataFreezeError("unsupported data freeze")
    equivalence = actual.get("equivalence")
    if not isinstance(equivalence, Mapping) or not isinstance(equivalence.get("path"), str):
        raise DataFreezeError("data freeze has no equivalence suite path")
    equivalence_path = generated_root / equivalence["path"]
    expected = _build_freeze_document(
        generated_root, equivalence_path, frozen, fixture=fixture
    )
    if actual != expected:
        raise DataFreezeError("data freeze no longer matches generated inputs")
    return actual


def resolve_frozen_dataset(
    dataset: str,
    *,
    freeze_path: Path = DEFAULT_FREEZE_MANIFEST,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    index_backend: str | None = None,
    frozen: FrozenDataConfiguration | None = None,
    fixture: bool = False,
) -> FrozenDatasetInputs:
    """Resolve paths and exact result provenance from the authoritative freeze."""
    if dataset not in DATASETS:
        raise DataFreezeError(f"unsupported frozen dataset {dataset!r}")
    bundle = validate_freeze_bundle(
        freeze_path,
        generated_root=generated_root,
        frozen=frozen,
        fixture=fixture,
    )
    record = bundle["datasets"][dataset]
    if index_backend is not None:
        if index_backend not in record["indexes"]:
            raise DataFreezeError(
                f"{dataset}: index backend {index_backend!r} is not frozen"
            )
    root = Path(generated_root)
    return FrozenDatasetInputs(
        dataset=dataset,
        generated_root=root,
        embedding_path=root / "embeddings" / f"{dataset}.npz",
        quality_queries_path=root / "embeddings" / f"{dataset}_test_queries.npz",
        ids_path=root / "beir_ids" / f"{dataset}_ids.json",
        qrels_path=root / "qrels" / f"{dataset}.tsv",
        workload_ids={
            regime: row["workload_id"] for regime, row in record["workloads"].items()
        },
        data_configuration_sha256=bundle["configuration"]["sha256"],
        data_sha256=record["data_sha256"],
        index_sha256_by_backend={
            backend: row["index_sha256"]
            for backend, row in record["indexes"].items()
        },
    )


def frozen_provenance_expectations(
    dataset: str,
    *,
    configuration_sha256: str,
    index_backend: str | None = None,
    freeze_path: Path = DEFAULT_FREEZE_MANIFEST,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    frozen: FrozenDataConfiguration | None = None,
    fixture: bool = False,
) -> Mapping[str, str | None]:
    """Combine one experiment configuration SHA with frozen data/index SHAs."""
    if not isinstance(configuration_sha256, str) or not _SHA256.fullmatch(
        configuration_sha256
    ):
        raise DataFreezeError(
            "experiment configuration_sha256 must be a lowercase SHA-256 digest"
        )
    inputs = resolve_frozen_dataset(
        dataset,
        freeze_path=freeze_path,
        generated_root=generated_root,
        index_backend=index_backend,
        frozen=frozen,
        fixture=fixture,
    )
    return {
        "configuration_sha256": configuration_sha256,
        "data_sha256": inputs.data_sha256,
        "index_sha256": (
            None
            if index_backend is None
            else inputs.index_sha256_by_backend[index_backend]
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--equivalence", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.validate_only:
        validate_freeze_bundle(
            arguments.output,
            generated_root=arguments.generated_root,
        )
        print(f"validated {arguments.output}")
    else:
        create_freeze_bundle(
            generated_root=arguments.generated_root,
            equivalence_path=arguments.equivalence,
            output_path=arguments.output,
        )
        print(f"froze {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
