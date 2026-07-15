"""Compatibility readers for every frozen historical result shape."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from bondmaxsim.results.models import (
    CURRENT_SCHEMA_NAME,
    ExperimentResultEnvelope,
    ResultValidationError,
    default_provenance,
)

_FIXED_HISTORICAL_TIME = "1970-01-01T00:00:00Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable(value: Any, fallback: str) -> str:
    text = str(value or fallback).lower()
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in text).strip("-") or fallback


def identify_historical_shape(data: Mapping[str, Any]) -> tuple[str, str]:
    """Return stable historical schema and current payload-kind names."""
    if {"dataset", "method", "num_docs", "num_queries", "recall_vs_exact_at_10"} <= set(data):
        kind = "timing_comparison" if data.get("ms_per_query") is not None else "pruning_accounting"
        return "historical.result-record.v1", kind
    experiment = str(data.get("experiment", ""))
    if experiment == "e01_bound_slack":
        return "historical.stage3.bound-slack.v1", "validation_audit"
    if experiment in {"e02_pruning_rate", "e03_order_ablation", "e04_exact_safe_pruning", "e05_approximate_recall_sweep", "e06_threshold_policy_ablation"}:
        return f"historical.stage3.{experiment}.v1", "pruning_accounting"
    if experiment in {"e07_cache_layout_sensitivity", "e08_checkpoint_ablation", "e09_bound_tightness_ablation", "r12c_interleaved_exact_safe"}:
        return f"historical.stage3.{experiment}.v1", "timing_comparison"
    if experiment == "e01_fixed_candidate_arms":
        return "historical.stage4.fixed-candidate-arms.v1", "candidate_frontier"
    if experiment == "e02_seeded_tau_recovery":
        return "historical.stage4.seeded-tau-recovery.v1", "timing_comparison"
    if experiment == "e03_partitioned_fused_scan":
        return "historical.stage4.partitioned-frontier.v1", "candidate_frontier"
    if experiment == "e01_ir_evaluation":
        return "historical.stage5.ir-evaluation.v1", "ir_quality"
    if experiment == "e02_paired_significance":
        return "historical.stage5.paired-significance.v1", "paired_significance"
    if experiment == "e03_bm25_baseline":
        return "historical.stage5.bm25.v1", "ir_quality"
    raise ResultValidationError("unrecognized historical result shape")


def _workload_id(data: Mapping[str, Any], dataset: str, experiment: str) -> str:
    if dataset == "multi-dataset":
        return "multi-dataset-historical-v1"
    if "stage5" in experiment or experiment.startswith("e0") and "query_source" in data:
        return f"beir-{dataset}-qrels-test-v1"
    return f"beir-{dataset}-mechanism-seed42-n50-v1"


def migrate_historical(path: Path, data: Mapping[str, Any]) -> ExperimentResultEnvelope:
    historical_schema, payload_kind = identify_historical_shape(data)
    experiment = _stable(data.get("experiment"), path.stem)
    dataset = _stable(data.get("dataset"), "multi-dataset")
    provenance = default_provenance(command="historical artifact; producing command unavailable")
    provenance.update(
        {
            "git_dirty": False,
            "source_path": str(path),
            "source_sha256": _sha256(path),
            "historical_schema_name": historical_schema,
            "migration_name": "normalize-historical-v1",
        }
    )
    method_configuration = {
        key: data[key]
        for key in ("method", "methods", "method_accounting", "method_wallclock", "kernel")
        if key in data
    }
    protocol = {
        "historical": True,
        "repetitions": data.get("n_repeats", data.get("rep_best_of")),
        "thread_mode": data.get("threads", data.get("thread_count", "unknown")),
    }
    environment = {
        "historical": True,
        "machine": data.get("machine", "unknown"),
        "os": data.get("os", "unknown"),
    }
    return ExperimentResultEnvelope.create(
        artifact_id=_stable(path.stem, "historical-result"),
        experiment_id=experiment,
        dataset_id=dataset,
        workload_id=_workload_id(data, dataset, experiment),
        method_configuration=method_configuration,
        protocol=protocol,
        environment=environment,
        provenance=provenance,
        payload_kind=payload_kind,
        payload={
            "historical_migration": True,
            "historical_schema_name": historical_schema,
            "source_data": dict(data),
        },
        created_at_utc=_FIXED_HISTORICAL_TIME,
    )


def load_result(path: Path | str) -> ExperimentResultEnvelope:
    """Load a current envelope or normalize a supported historical artifact."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultValidationError(f"cannot read result {source}: {error}") from error
    if not isinstance(data, Mapping):
        raise ResultValidationError("result JSON root must be an object")
    if data.get("schema_name") == CURRENT_SCHEMA_NAME:
        return ExperimentResultEnvelope.from_dict(data)
    return migrate_historical(source, data)
