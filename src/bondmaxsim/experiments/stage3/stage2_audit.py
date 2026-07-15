"""Versioned Stage 2 normalization and exact-agreement audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.fixture import fixture_arrays
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.experiments.environment import capture_session_environment
from bondmaxsim.experiments.mechanism import PreparedFusedWorkload
from bondmaxsim.experiments.stage3.common import (
    configuration_sha256,
    load_mechanism_data,
    sha256_file,
)
from bondmaxsim.experiments.stage3.survival import PreparedWideAccountingWorkload
from bondmaxsim.oracle.normalization import check_unit_norm
from bondmaxsim.results.io import atomic_write_envelope, deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope, default_provenance
from bondmaxsim.testbed.config import RunConfig


@dataclass(frozen=True)
class Stage2AuditRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


def _write_audit(
    *,
    experiment_id: str,
    dataset: str,
    workload_id: str,
    command: str,
    method_configuration: dict[str, Any],
    protocol: dict[str, Any],
    checks: list[dict[str, Any]],
    configuration_hash: str,
    data_hash: str | None,
    output_dir: Path,
    session_id: str,
) -> Stage2AuditRun:
    environment = capture_session_environment(session_id, build_profile="release-native")
    provenance = default_provenance(command=command, git_revision=environment.get("code", {}).get("git_revision") or "unknown")
    provenance.update(
        {
            "git_dirty": bool(environment.get("code", {}).get("git_dirty")),
            "lockfile_sha256": environment.get("code", {}).get("uv_lock_sha256"),
            "configuration_sha256": configuration_hash,
            "data_sha256": data_hash,
            "native_binary_sha256": environment.get("toolchain", {}).get("native_binary_sha256"),
        }
    )
    passed = all(check["status"] == "pass" for check in checks)
    envelope = ExperimentResultEnvelope.create(
        artifact_id=f"{experiment_id}-{dataset}",
        experiment_id=experiment_id,
        dataset_id=dataset,
        workload_id=workload_id,
        method_configuration=method_configuration,
        protocol={"timing_evidence": False, **protocol},
        environment=environment,
        provenance=provenance,
        payload_kind="validation_audit",
        payload={"checks": checks, "passed": passed},
    )
    path = output_dir / deterministic_result_name(experiment_id, dataset, workload_id)
    atomic_write_envelope(path, envelope)
    if not passed:
        raise RuntimeError(f"{experiment_id} failed; diagnostic artifact: {path}")
    return Stage2AuditRun(envelope, path)


def run_normalization_audit(
    dataset: str,
    *,
    fixture: bool,
    output_dir: Path,
    session_id: str = "stage2-normalization-audit",
) -> Stage2AuditRun:
    if fixture:
        arrays = fixture_arrays()
        flat = arrays["doc_values"]
        values = arrays["query_values"]
        selected_dataset = "synthetic-small-v1"
        workload_id = "synthetic-small-v1"
        data_hash = None
    else:
        flat, _, queries = load_dataset(dataset, verify_norm=True)
        values = np.concatenate(queries, axis=0)
        selected_dataset = dataset
        workload_id = f"stage2-{dataset}-all-queries-v1"
        data_hash = sha256_file(REPO_ROOT / "data" / "embeddings" / f"{dataset}.npz")
    doc_stats = check_unit_norm(flat)
    query_stats = check_unit_norm(values)
    checks = [
        {
            "check_id": "document-token-unit-norm",
            "status": "pass" if doc_stats["n_violating"] == 0 else "fail",
            "details": doc_stats,
        },
        {
            "check_id": "query-token-unit-norm",
            "status": "pass" if query_stats["n_violating"] == 0 else "fail",
            "details": query_stats,
        },
    ]
    config = {"dataset": dataset, "fixture": fixture, "tolerance": "normalization.check_unit_norm default"}
    return _write_audit(
        experiment_id="stage2-s01-normalization-audit",
        dataset=selected_dataset,
        workload_id=workload_id,
        command=f"uv run python -m experiments.stage2_testbed.s01_normalization_guard --dataset {dataset}" + (" --fixture" if fixture else ""),
        method_configuration=config,
        protocol={"instrument": "full-token-normalization-audit"},
        checks=checks,
        configuration_hash=configuration_sha256(config),
        data_hash=data_hash,
        output_dir=output_dir,
        session_id=session_id,
    )


def run_exact_agreement_audit(
    dataset: str,
    *,
    fixture: bool,
    output_dir: Path,
    session_id: str = "stage2-exact-agreement-audit",
) -> Stage2AuditRun:
    data = load_mechanism_data(dataset, fixture=fixture)
    orders = ("natural", "bond") if fixture else ("natural", "bond", "pca")
    k = 3 if fixture else 10
    wide = PreparedWideAccountingWorkload(data, k)
    fused = PreparedFusedWorkload(data.flat_tokens, data.doc_starts, list(data.queries), list(data.query_ids))
    checks: list[dict[str, Any]] = []
    for order in orders:
        wide_result = wide.run("self_bound", order, 1.0)
        wide_agreement = wide_result["agreement"]
        checks.append(
            {
                "check_id": f"wide-accounting-{order}",
                "status": "pass" if wide_agreement["boundary_tie_equivalent"] else "fail",
                "details": wide_agreement,
            }
        )
        config = RunConfig(
            dataset=data.dataset,
            method="fused_panel_maxsim_bond",
            dimension_order=order,
            threshold_policy="self_bound",
            k=k,
            shrink=1.0,
            checkpoints=((4, 6) if fixture else (32, 64)),
        )
        prepared = fused.prepare_arm(config, scanner="bond", n_threads=1, level="doc")
        fused_agreement = prepared.validate(prepared.operation()).to_dict()
        checks.append(
            {
                "check_id": f"fused-document-{order}",
                "status": "pass" if fused_agreement["boundary_tie_equivalent"] else "fail",
                "details": fused_agreement,
            }
        )
    audit_config = {
        "dataset": data.dataset,
        "fixture": fixture,
        "orders": list(orders),
        "k": k,
        "shrink": 1.0,
        "wide_policy": "self_bound",
        "fused_policy": "self_bound",
    }
    return _write_audit(
        experiment_id="stage2-s02-exact-agreement-audit",
        dataset=data.dataset,
        workload_id=str(data.workload_metadata["workload_id"]),
        command=f"uv run python -m experiments.stage2_testbed.s02_exact_agreement --dataset {dataset}" + (" --fixture" if fixture else ""),
        method_configuration=audit_config,
        protocol={"instrument": "one-validated-native-pass-per-arm", "workload": dict(data.workload_metadata)},
        checks=checks,
        configuration_hash=configuration_sha256(audit_config),
        data_hash=data.data_sha256,
        output_dir=output_dir,
        session_id=session_id,
    )
