"""Shared execution for the Stage 4 serial checkpoint-ablation pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_ids
from bondmaxsim.data.fixture import fixture_arrays
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.experiments.arms import ArmSpec, ExperimentSpec
from bondmaxsim.experiments.execution import execute_prepared_timing_experiment
from bondmaxsim.experiments.mechanism import PreparedFusedWorkload
from bondmaxsim.experiments.timing import TimingProtocol, new_session_id
from bondmaxsim.experiments.workloads import WorkloadMetadata
from bondmaxsim.results.io import atomic_write_envelope, deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope
from bondmaxsim.testbed.config import RunConfig


@dataclass(frozen=True)
class CheckpointAblationConfig:
    dataset: str = "scifact"
    orders: tuple[str, ...] = ("natural", "bond")
    checkpoint_sets: tuple[tuple[int, ...], ...] = (
        (32,),
        (48,),
        (64,),
        (96,),
        (112,),
        (32, 64),
        (64, 112),
        (32, 64, 96, 112),
    )
    policy: str = "oracle"
    shrink: float = 1.0
    k: int = 10
    query_count: int = 50
    query_seed: int = 42
    n_threads: int = 1
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "CheckpointAblationConfig":
        return cls(
            dataset="synthetic-small-v1",
            orders=("natural", "bond"),
            checkpoint_sets=((8,),),
            k=3,
            query_count=4,
            n_threads=1,
            fixture=True,
        )

    @property
    def thread_tag(self) -> str:
        return f"{self.n_threads}t" if self.n_threads > 0 else "mt"


@dataclass(frozen=True)
class CheckpointAblationRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(config: CheckpointAblationConfig) -> str:
    value = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _selected_indices(population: int, count: int, seed: int) -> np.ndarray:
    if count > population:
        raise ValueError("query sample exceeds the available population")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(population, size=count, replace=False))


def assert_historical_methodology(
    config: CheckpointAblationConfig, historical_path: Path
) -> dict[str, Any]:
    """Check unchanged scientific controls without comparing old timings."""
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    expected = {
        "experiment": "e08_checkpoint_ablation",
        "dataset": config.dataset,
        "query_seed": config.query_seed,
        "k": config.k,
        "shrink": config.shrink,
        "policy": config.policy,
        "orders": list(config.orders),
        "checkpoint_sets": [list(row) for row in config.checkpoint_sets],
    }
    mismatches = {
        key: {"expected": value, "historical": historical.get(key)}
        for key, value in expected.items()
        if historical.get(key) != value
    }
    if historical.get("n_queries") != config.query_count:
        mismatches["n_queries"] = {
            "expected": config.query_count,
            "historical": historical.get("n_queries"),
        }
    if mismatches:
        raise ValueError(f"historical e08 methodology mismatch: {mismatches}")
    return {"status": "pass", "checked_fields": sorted([*expected, "n_queries"])}


def _load(config: CheckpointAblationConfig):
    if config.fixture:
        arrays = fixture_arrays()
        values, starts = arrays["query_values"], arrays["query_starts"]
        queries = [
            values[int(start):int(starts[index + 1]) if index + 1 < len(starts) else len(values)]
            for index, start in enumerate(starts)
        ]
        query_ids = [f"q{index}" for index in range(len(queries))]
        workload = {
            "workload_id": "synthetic-small-v1",
            "dataset": config.dataset,
            "regime": "fixture",
            "sample_size": len(queries),
            "token_counts": [len(query) for query in queries],
        }
        return arrays["doc_values"], arrays["doc_starts"], queries, query_ids, workload, None

    flat_tokens, doc_starts, population = load_dataset(config.dataset)
    all_ids = load_ids(config.dataset)["query_ids"][:len(population)]
    indices = _selected_indices(len(population), config.query_count, config.query_seed)
    queries = [population[int(index)] for index in indices]
    query_ids = [all_ids[int(index)] for index in indices]
    workload = WorkloadMetadata.from_queries(
        dataset=config.dataset,
        regime="mechanism",
        query_ids=query_ids,
        queries=queries,
        encoding_configuration={
            "model": "lightonai/GTE-ModernColBERT-v1",
            "dimension": int(flat_tokens.shape[1]),
            "l2_normalized": True,
        },
        selection="deterministic seed-42 sample of 50 archive queries, sorted source indices",
        source_revision="archive/preliminaries/02_bond_variance/.cache",
        source_split="queries-first-200",
    ).to_dict()
    data_path = REPO_ROOT / "data" / "embeddings" / f"{config.dataset}.npz"
    return flat_tokens, doc_starts, queries, query_ids, workload, _sha256(data_path)


def _arm_id(order: str, checkpoints: tuple[int, ...]) -> str:
    suffix = "-".join(f"c{value:03d}" for value in checkpoints)
    return f"bond-{order}-{suffix}"


def run_checkpoint_ablation(
    config: CheckpointAblationConfig,
    *,
    output_dir: Path,
    session_id: str | None = None,
    environment: dict[str, Any] | None = None,
) -> CheckpointAblationRun:
    """Prepare, time, validate, account, and atomically serialize one session."""
    selected_session = session_id or new_session_id("stage3-e08")

    def prepare() -> ExperimentSpec:
        flat, starts, queries, query_ids, workload_metadata, data_sha = _load(config)
        historical_check = None
        if not config.fixture:
            historical_check = assert_historical_methodology(
                config,
                REPO_ROOT / "results" / "json"
                / f"stage3_mechanism_e08_checkpoint_ablation_{config.dataset}.json",
            )
        workload = PreparedFusedWorkload(flat, starts, queries, query_ids)
        arms = []
        for order in config.orders:
            for checkpoints in config.checkpoint_sets:
                run_config = RunConfig(
                    dataset=config.dataset,
                    method="fused_panel_maxsim_bond",
                    dimension_order=order,
                    threshold_policy=config.policy,
                    k=config.k,
                    shrink=config.shrink,
                    checkpoints=checkpoints,
                )
                prepared = workload.prepare_arm(
                    run_config, scanner="bond", n_threads=config.n_threads
                )
                arms.append(
                    ArmSpec(
                        arm_id=_arm_id(order, checkpoints),
                        method_family="fused-panel-bond",
                        operation=prepared.operation,
                        parameters={
                            "dimension_order": order,
                            "checkpoints": list(checkpoints),
                            "threshold_policy": config.policy,
                            "shrink": config.shrink,
                            "thread_tag": config.thread_tag,
                        },
                        exactness="exact_safe",
                        comparison_scope="uncapped_reference",
                        validator=prepared.validate,
                        metadata_extractor=prepared.metadata,
                        timer_scope_id="native-workload-pass",
                    )
                )
        dense_config = RunConfig(
            dataset=config.dataset,
            method="fused_panel_maxsim_bond",
            dimension_order="natural",
            threshold_policy="none",
            k=config.k,
            shrink=1.0,
        )
        dense = workload.prepare_arm(
            dense_config, scanner="brute", n_threads=config.n_threads
        )
        arms.append(
            ArmSpec(
                arm_id="dense-fused",
                method_family="fused-panel-dense",
                operation=dense.operation,
                parameters={"thread_tag": config.thread_tag},
                exactness="reference",
                comparison_scope="uncapped_reference",
                validator=dense.validate,
                metadata_extractor=dense.metadata,
                timer_scope_id="native-workload-pass",
            )
        )
        experiment_id = f"stage3-e08-checkpoint-ablation-{config.thread_tag}"
        return ExperimentSpec(
            experiment_id=experiment_id,
            artifact_id=f"{experiment_id}-{config.dataset}",
            dataset_id=config.dataset,
            workload_id=workload_metadata["workload_id"],
            arms=tuple(arms),
            command=(
                "uv run python -m experiments.stage3_mechanism.e08_checkpoint_ablation "
                f"--dataset {config.dataset} --threads {config.n_threads}"
            ),
            baseline_arm_id="dense-fused",
            metadata={
                "checkpoint_ablation": asdict(config),
                "historical_methodology_check": historical_check,
            },
            workload_metadata=workload_metadata,
            provenance_inputs={
                "configuration_sha256": _configuration_sha256(config),
                "data_sha256": data_sha,
                "index_sha256": None,
                "input_artifact_ids": (
                    [f"historical-stage3-e08-{config.dataset}"]
                    if historical_check is not None
                    else []
                ),
            },
        )

    protocol = TimingProtocol.preset("fixture" if config.fixture else "mechanism")
    envelope, _, spec = execute_prepared_timing_experiment(
        prepare,
        protocol,
        session_id=selected_session,
        build_profile="release-native",
        environment=environment,
    )
    output_path = output_dir / deterministic_result_name(
        spec.experiment_id, spec.dataset_id, spec.workload_id
    )
    atomic_write_envelope(output_path, envelope)
    return CheckpointAblationRun(envelope, output_path)
