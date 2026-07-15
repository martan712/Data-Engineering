"""Paired significance analysis over saved, validated Stage 5 e01 results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from bondmaxsim.eval.significance import paired_permutation_test
from bondmaxsim.experiments.environment import capture_session_environment
from bondmaxsim.experiments.timing import new_session_id
from bondmaxsim.results import load_result
from bondmaxsim.results.io import atomic_write_envelope, deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope, default_provenance


@dataclass(frozen=True)
class Stage5SignificanceRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


def significance_from_ir_artifact(
    input_path: Path,
    *,
    output_dir: Path,
    baseline_arm_id: str = "dense-fused",
    candidate_arm_ids: Iterable[str] | None = None,
    n_permutations: int = 20_000,
    seed: int = 0,
    environment: dict[str, Any] | None = None,
) -> Stage5SignificanceRun:
    """Analyze retained per-query values without rebuilding retrieval systems."""
    source = load_result(input_path)
    if source.payload_kind != "timing_comparison":
        raise ValueError("Stage 5 e02 requires a migrated e01 timing artifact")
    sessions = source.payload.get("sessions", [])
    if not sessions or not all(session.get("complete") for session in sessions):
        raise ValueError("Stage 5 e02 refuses incomplete e01 timing artifacts")
    quality = source.payload.get("quality")
    if not isinstance(quality, dict) or not isinstance(quality.get("rows"), list):
        raise ValueError("e01 artifact has no validated per-query quality rows")
    rows = {row["arm_id"]: row for row in quality["rows"]}
    if baseline_arm_id not in rows:
        raise ValueError(f"baseline arm {baseline_arm_id!r} is absent")
    baseline = rows[baseline_arm_id].get("per_query_ndcg_at_10")
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("baseline has no retained per-query nDCG@10")
    selected = list(candidate_arm_ids) if candidate_arm_ids is not None else [
        arm_id
        for arm_id, row in rows.items()
        if arm_id != baseline_arm_id and row.get("comparison_scope") in {
            "probe_frontier", "equal_work_reranking", "system_cap"
        }
    ]
    if not selected:
        raise ValueError("no approximate candidate arms were selected")
    comparisons = []
    for arm_id in selected:
        if arm_id not in rows:
            raise ValueError(f"candidate arm {arm_id!r} is absent")
        candidate = rows[arm_id].get("per_query_ndcg_at_10")
        if not isinstance(candidate, dict) or not candidate:
            raise ValueError(f"candidate arm {arm_id!r} has no per-query values")
        comparisons.append(
            {
                "candidate_arm_id": arm_id,
                "baseline_arm_id": baseline_arm_id,
                "metric": "ndcg_at_10",
                "test": "paired-two-sided-sign-flip-permutation",
                **paired_permutation_test(
                    candidate,
                    baseline,
                    n_permutations=n_permutations,
                    seed=seed,
                ),
            }
        )
    command = (
        "uv run python -m experiments.stage5_corect.e02_paired_significance "
        f"--input {input_path}"
    )
    configuration = {
        "baseline_arm_id": baseline_arm_id,
        "candidate_arm_ids": selected,
        "metric": "ndcg_at_10",
        "n_permutations": n_permutations,
        "seed": seed,
        "retrieval_setup": "consumed from validated e01 artifact",
    }
    configuration_sha = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    snapshot = environment or capture_session_environment(
        new_session_id("stage5-e02"), build_profile="analysis"
    )
    provenance = default_provenance(
        command=command,
        git_revision=str(snapshot.get("code", {}).get("git_revision") or "unknown"),
    )
    provenance.update(
        {
            "git_dirty": bool(snapshot.get("code", {}).get("git_dirty")),
            "lockfile_sha256": snapshot.get("code", {}).get("uv_lock_sha256"),
            "configuration_sha256": configuration_sha,
            "data_sha256": source.provenance.get("data_sha256"),
            "index_sha256": source.provenance.get("index_sha256"),
            "native_binary_sha256": source.provenance.get("native_binary_sha256"),
            "input_artifact_ids": [source.artifact_id],
        }
    )
    experiment_id = "stage5-e02-paired-significance"
    envelope = ExperimentResultEnvelope.create(
        artifact_id=f"{experiment_id}-{source.dataset_id}",
        experiment_id=experiment_id,
        dataset_id=source.dataset_id,
        workload_id=source.workload_id,
        method_configuration=configuration,
        protocol={
            "analysis_only": True,
            "source_schema": f"{source.schema_name}@{source.schema_version}",
        },
        environment=snapshot,
        provenance=provenance,
        payload_kind="paired_significance",
        payload={"comparisons": comparisons},
    )
    output_path = output_dir / deterministic_result_name(
        experiment_id, source.dataset_id, source.workload_id
    )
    atomic_write_envelope(output_path, envelope)
    return Stage5SignificanceRun(envelope, output_path)
