"""Stable Stage 3 workload loading, hashing, and accounting serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_ids
from bondmaxsim.data.fixture import fixture_arrays
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.experiments.environment import capture_session_environment
from bondmaxsim.experiments.workloads import WorkloadMetadata
from bondmaxsim.results.io import atomic_write_envelope, deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope, default_provenance


@dataclass(frozen=True)
class MechanismData:
    dataset: str
    flat_tokens: np.ndarray
    doc_starts: np.ndarray
    queries: tuple[np.ndarray, ...]
    query_ids: tuple[str, ...]
    workload_metadata: Mapping[str, Any]
    data_sha256: str | None
    fixture: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_sha256(config: Any) -> str:
    value = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def selected_indices(population: int, count: int, seed: int) -> np.ndarray:
    if count > population:
        raise ValueError("query sample exceeds the available population")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(population, size=count, replace=False))


def _fixture_data() -> MechanismData:
    arrays = fixture_arrays()
    values = arrays["query_values"]
    starts = arrays["query_starts"]
    queries = tuple(
        np.ascontiguousarray(
            values[int(start) : int(starts[index + 1]) if index + 1 < len(starts) else len(values)],
            dtype=np.float32,
        )
        for index, start in enumerate(starts)
    )
    query_ids = tuple(f"q{index}" for index in range(len(queries)))
    token_counts = [len(query) for query in queries]
    workload = {
        "workload_id": "synthetic-small-v1",
        "dataset": "synthetic-small-v1",
        "regime": "fixture",
        "sample_size": len(queries),
        "token_counts": token_counts,
        "token_count_mean": float(np.mean(token_counts)),
        "token_count_median": float(np.median(token_counts)),
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "selection": "canonical deterministic offline fixture",
    }
    return MechanismData(
        dataset="synthetic-small-v1",
        flat_tokens=arrays["doc_values"],
        doc_starts=arrays["doc_starts"],
        queries=queries,
        query_ids=query_ids,
        workload_metadata=workload,
        data_sha256=None,
        fixture=True,
    )


def load_mechanism_data(
    dataset: str,
    *,
    fixture: bool,
    query_count: int = 50,
    query_seed: int = 42,
) -> MechanismData:
    """Load the frozen mechanism sample or the canonical diagnostic fixture."""
    if fixture:
        return _fixture_data()
    if query_count != 50 or query_seed != 42:
        raise ValueError("the current mechanism workload is frozen at seed 42 and 50 queries")
    flat, starts, population = load_dataset(dataset)
    indices = selected_indices(len(population), query_count, query_seed)
    queries = tuple(np.ascontiguousarray(population[int(index)], dtype=np.float32) for index in indices)
    all_ids = load_ids(dataset)["query_ids"][: len(population)]
    query_ids = tuple(str(all_ids[int(index)]) for index in indices)
    workload = WorkloadMetadata.from_queries(
        dataset=dataset,
        regime="mechanism",
        query_ids=query_ids,
        queries=queries,
        encoding_configuration={
            "model": "lightonai/GTE-ModernColBERT-v1",
            "dimension": int(flat.shape[1]),
            "l2_normalized": True,
        },
        selection="deterministic seed-42 sample of 50 archive queries, sorted source indices",
        source_revision="archive/preliminaries/02_bond_variance/.cache",
        source_split="queries-first-200",
    ).to_dict()
    data_path = REPO_ROOT / "data" / "embeddings" / f"{dataset}.npz"
    return MechanismData(
        dataset=dataset,
        flat_tokens=flat,
        doc_starts=starts,
        queries=queries,
        query_ids=query_ids,
        workload_metadata=workload,
        data_sha256=sha256_file(data_path),
        fixture=False,
    )


def write_accounting_result(
    *,
    experiment_id: str,
    artifact_id: str,
    data: MechanismData,
    command: str,
    method_configuration: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    configuration_hash: str,
    output_dir: Path,
    session_id: str,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[ExperimentResultEnvelope, Path]:
    """Create and atomically write one non-timing mechanism envelope."""
    environment = capture_session_environment(session_id, build_profile="release-native")
    provenance = default_provenance(
        command=command,
        git_revision=environment.get("code", {}).get("git_revision") or "unknown",
    )
    provenance.update(
        {
            "git_dirty": bool(environment.get("code", {}).get("git_dirty")),
            "lockfile_sha256": environment.get("code", {}).get("uv_lock_sha256"),
            "configuration_sha256": configuration_hash,
            "data_sha256": data.data_sha256,
            "native_binary_sha256": environment.get("toolchain", {}).get(
                "native_binary_sha256"
            ),
        }
    )
    envelope = ExperimentResultEnvelope.create(
        artifact_id=artifact_id,
        experiment_id=experiment_id,
        dataset_id=data.dataset,
        workload_id=str(data.workload_metadata["workload_id"]),
        method_configuration=method_configuration,
        protocol={
            "instrument": "accounting",
            "timing_evidence": False,
            "workload": dict(data.workload_metadata),
            **dict(protocol or {}),
        },
        environment=environment,
        provenance=provenance,
        payload_kind="pruning_accounting",
        payload={"rows": rows},
    )
    path = output_dir / deterministic_result_name(
        experiment_id, data.dataset, str(data.workload_metadata["workload_id"])
    )
    atomic_write_envelope(path, envelope)
    return envelope, path
