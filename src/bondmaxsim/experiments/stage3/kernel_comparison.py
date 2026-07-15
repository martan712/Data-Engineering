"""Counterbalanced E03 and R12c fused-kernel timing experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_ids
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.experiments.arms import ArmSpec, ExperimentSpec
from bondmaxsim.experiments.execution import execute_prepared_timing_experiment
from bondmaxsim.experiments.mechanism import (
    MechanismPassResult,
    PreparedFusedWorkload,
    QueryPassResult,
)
from bondmaxsim.experiments.stage3.common import (
    MechanismData,
    configuration_sha256,
    load_mechanism_data,
    sha256_file,
)
from bondmaxsim.experiments.timing import TimingProtocol, new_session_id
from bondmaxsim.experiments.workloads import (
    embedding_configuration_sha256,
    ordered_query_id_sha256,
)
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores
from bondmaxsim.results.io import atomic_write_envelope, deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope
from bondmaxsim.testbed.config import RunConfig


@dataclass(frozen=True)
class KernelComparisonConfig:
    dataset: str = "scifact"
    orders: tuple[str, ...] = ("natural", "bond", "pca")
    levels: tuple[str, ...] = ("doc", "token")
    checkpoints: tuple[int, ...] = (32, 64)
    policy: str = "oracle"
    shrink: float = 1.0
    k: int = 10
    query_count: int = 50
    query_seed: int = 42
    n_threads: int = 1
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "KernelComparisonConfig":
        return cls(dataset="synthetic-small-v1", orders=("natural", "bond"), checkpoints=(4, 6), k=3, n_threads=1, fixture=True)

    @property
    def thread_tag(self) -> str:
        return f"{self.n_threads}t" if self.n_threads > 0 else "mt"


@dataclass(frozen=True)
class ExactSafeInterleavedConfig:
    dataset: str = "scifact"
    checkpoint_sets: tuple[tuple[int, ...], ...] = ((32,), (112,), (64, 112))
    order: str = "natural"
    policy: str = "oracle"
    shrink: float = 1.0
    k: int = 10
    query_count: int = 50
    query_seed: int = 42
    n_threads: int = 0
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "ExactSafeInterleavedConfig":
        return cls(dataset="synthetic-small-v1", checkpoint_sets=((4,), (6,), (4, 6)), k=3, n_threads=1, fixture=True)

    @property
    def thread_tag(self) -> str:
        return f"{self.n_threads}t" if self.n_threads > 0 else "mt"


@dataclass(frozen=True)
class BaselineProbeConfig:
    dataset: str = "arguana"
    checkpoint_sets: tuple[tuple[int, ...], ...] = (
        (32,),
        (112,),
        (64, 112),
        (32, 64, 96, 112),
    )
    order: str = "natural"
    policy: str = "oracle"
    shrink: float = 1.0
    k: int = 10
    query_count: int = 120
    n_threads: int = 0
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "BaselineProbeConfig":
        return cls(
            dataset="synthetic-small-v1",
            checkpoint_sets=((4,), (6,), (4, 6)),
            k=3,
            query_count=4,
            n_threads=1,
            fixture=True,
        )

    @property
    def thread_tag(self) -> str:
        return f"{self.n_threads}t" if self.n_threads > 0 else "mt"


@dataclass(frozen=True)
class TimingExperimentRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


class _PreparedNumpyArm:
    def __init__(self, workload: PreparedFusedWorkload, dataset: str, k: int, n_threads: int) -> None:
        self.workload = workload
        self.k = k
        # The controller applies the requested BLAS limit for this one-process
        # experiment. E03 executes one explicit thread mode per invocation.
        self._thread_controller = (
            threadpool_limits(limits=n_threads, user_api="blas") if n_threads > 0 else None
        )
        self._validator_arm = self.workload.prepare_arm(
            RunConfig(
                dataset=dataset,
                method="fused_panel_maxsim_bond",
                dimension_order="natural",
                threshold_policy="none",
                k=self.k,
                shrink=1.0,
            ),
            scanner="brute",
            n_threads=1,
        )

    def operation(self) -> MechanismPassResult:
        rows = []
        for query in self.workload.queries:
            scores = exact_maxsim_scores(query, self.workload.packing.flat_tokens, self.workload.packing.doc_starts)
            ids, top_scores = topk_from_scores(scores, self.k)
            rows.append(QueryPassResult(ids.astype(np.uint32), top_scores, np.zeros(3, dtype=np.uint64)))
        return MechanismPassResult(tuple(rows))

    def validate(self, result: MechanismPassResult):
        # Reuse the shared independently scored gate rather than asserting the
        # NumPy reference agrees with itself by construction.
        return self._validator_arm.validate(result)

    def metadata(self, result: MechanismPassResult) -> dict[str, Any]:
        return {
            "pruning_accounting": [
                {
                    "query_id": query_id,
                    "cells_scanned_pct": 100.0,
                    "pruned_docs_pct": 0.0,
                }
                for query_id in self.workload.query_ids
            ]
        }


def _dense_arm(workload: PreparedFusedWorkload, dataset: str, k: int, n_threads: int) -> ArmSpec:
    config = RunConfig(
        dataset=dataset,
        method="fused_panel_maxsim_bond",
        dimension_order="natural",
        threshold_policy="none",
        k=k,
        shrink=1.0,
    )
    prepared = workload.prepare_arm(config, scanner="brute", n_threads=n_threads)
    return ArmSpec(
        arm_id="dense-fused",
        method_family="fused-panel-dense",
        operation=prepared.operation,
        parameters={"thread_count": n_threads},
        exactness="reference",
        comparison_scope="uncapped_reference",
        validator=prepared.validate,
        metadata_extractor=prepared.metadata,
        timer_scope_id="native-workload-pass",
    )


def _bond_arm(
    workload: PreparedFusedWorkload,
    *,
    dataset: str,
    k: int,
    n_threads: int,
    order: str,
    level: str,
    checkpoints: tuple[int, ...],
    policy: str,
    shrink: float,
    arm_id: str,
) -> ArmSpec:
    config = RunConfig(
        dataset=dataset,
        method="fused_panel_maxsim_bond_token" if level == "token" else "fused_panel_maxsim_bond",
        dimension_order=order,
        threshold_policy=policy,
        k=k,
        shrink=shrink,
        checkpoints=checkpoints,
    )
    prepared = workload.prepare_arm(config, scanner="bond", n_threads=n_threads, level=level)

    def metadata(result: MechanismPassResult) -> dict[str, Any]:
        extracted = prepared.metadata(result)
        if level != "token":
            return extracted
        for row, query_result, query_prepared in zip(
            extracted["pruning_accounting"], result.queries, prepared.prepared
        ):
            padded_tokens = int(query_prepared[0].doc_offsets[-1])
            tokens_pruned = int(query_result.stats[2])
            row["tokens_pruned"] = tokens_pruned
            row["tokens_pruned_pct"] = (
                100.0 * tokens_pruned / padded_tokens if padded_tokens else 0.0
            )
        return extracted

    return ArmSpec(
        arm_id=arm_id,
        method_family=f"fused-panel-bond-{level}",
        operation=prepared.operation,
        parameters={
            "dimension_order": order,
            "prune_level": level,
            "checkpoints": list(checkpoints),
            "threshold_policy": policy,
            "shrink": shrink,
            "thread_count": n_threads,
        },
        exactness="exact_safe",
        comparison_scope="uncapped_reference",
        validator=prepared.validate,
        metadata_extractor=metadata,
        timer_scope_id="native-workload-pass",
    )


def _write_timing(
    config: Any,
    *,
    experiment_id: str,
    output_dir: Path,
    session_id: str | None,
    prepare_spec,
) -> TimingExperimentRun:
    selected_session = session_id or new_session_id(experiment_id)
    envelope, _, spec = execute_prepared_timing_experiment(
        prepare_spec,
        TimingProtocol.preset("fixture" if config.fixture else "mechanism"),
        session_id=selected_session,
        build_profile="release-native",
    )
    path = output_dir / deterministic_result_name(spec.experiment_id, spec.dataset_id, spec.workload_id)
    atomic_write_envelope(path, envelope)
    return TimingExperimentRun(envelope, path)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_baseline_probe_data(config: BaselineProbeConfig) -> MechanismData:
    if config.fixture:
        return load_mechanism_data(config.dataset, fixture=True)
    flat, starts, population = load_dataset(config.dataset)
    if config.query_count <= 0 or config.query_count > len(population):
        raise ValueError("R12b query_count must be within the encoded query population")
    queries = tuple(
        np.ascontiguousarray(query, dtype=np.float32)
        for query in population[: config.query_count]
    )
    query_ids = tuple(
        str(query_id)
        for query_id in load_ids(config.dataset)["query_ids"][: config.query_count]
    )
    token_counts = tuple(len(query) for query in queries)
    query_values = np.concatenate(queries, axis=0)
    encoding = {
        "model": "lightonai/GTE-ModernColBERT-v1",
        "dimension": int(flat.shape[1]),
        "l2_normalized": True,
    }
    workload_id = f"beir-{config.dataset}-r12b-prefix-n{config.query_count}-v1"
    workload = {
        "workload_id": workload_id,
        "dataset": config.dataset,
        "regime": "r12b_baseline_probe",
        "ordered_query_id_sha256": ordered_query_id_sha256(query_ids),
        "embedding_configuration_sha256": embedding_configuration_sha256(
            query_values, encoding
        ),
        "sample_size": config.query_count,
        "token_count_mean": float(mean(token_counts)),
        "token_count_median": float(median(token_counts)),
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "token_counts": list(token_counts),
        "token_counts_sha256": _canonical_sha256(list(token_counts)),
        "selection": f"first {config.query_count} archive queries in source order",
        "source_revision": "archive/preliminaries/02_bond_variance/.cache",
        "source_split": "queries-first-200",
    }
    data_path = REPO_ROOT / "data" / "embeddings" / f"{config.dataset}.npz"
    return MechanismData(
        dataset=config.dataset,
        flat_tokens=flat,
        doc_starts=starts,
        queries=queries,
        query_ids=query_ids,
        workload_metadata=workload,
        data_sha256=sha256_file(data_path),
        fixture=False,
    )


def run_kernel_comparison(
    config: KernelComparisonConfig,
    *,
    output_dir: Path,
    session_id: str | None = None,
) -> TimingExperimentRun:
    experiment_id = f"stage3-e03-kernel-comparison-{config.thread_tag}"

    def prepare() -> ExperimentSpec:
        data = load_mechanism_data(config.dataset, fixture=config.fixture, query_count=config.query_count, query_seed=config.query_seed)
        workload = PreparedFusedWorkload(data.flat_tokens, data.doc_starts, list(data.queries), list(data.query_ids))
        arms = []
        for order in config.orders:
            for level in config.levels:
                arms.append(
                    _bond_arm(
                        workload,
                        dataset=data.dataset,
                        k=config.k,
                        n_threads=config.n_threads,
                        order=order,
                        level=level,
                        checkpoints=config.checkpoints,
                        policy=config.policy,
                        shrink=config.shrink,
                        arm_id=f"bond-{level}-{order}",
                    )
                )
        arms.append(_dense_arm(workload, data.dataset, config.k, config.n_threads))
        numpy_arm = _PreparedNumpyArm(workload, data.dataset, config.k, config.n_threads)
        arms.append(
            ArmSpec(
                arm_id="dense-numpy",
                method_family="numpy-blas-dense",
                operation=numpy_arm.operation,
                parameters={"thread_count": config.n_threads},
                exactness="reference",
                comparison_scope="uncapped_reference",
                validator=numpy_arm.validate,
                metadata_extractor=numpy_arm.metadata,
                timer_scope_id="numpy-workload-pass",
            )
        )
        return ExperimentSpec(
            experiment_id=experiment_id,
            artifact_id=f"{experiment_id}-{data.dataset}",
            dataset_id=data.dataset,
            workload_id=str(data.workload_metadata["workload_id"]),
            arms=tuple(arms),
            command=(
                "uv run python -m experiments.stage3_mechanism.e03_order_ablation "
                f"--dataset {config.dataset} --threads {config.n_threads}" + (" --fixture" if config.fixture else "")
            ),
            baseline_arm_id="dense-fused",
            metadata={"kernel_comparison": asdict(config)},
            workload_metadata=data.workload_metadata,
            provenance_inputs={
                "configuration_sha256": configuration_sha256(config),
                "data_sha256": data.data_sha256,
                "index_sha256": None,
                "input_artifact_ids": [],
            },
        )

    return _write_timing(config, experiment_id=experiment_id, output_dir=output_dir, session_id=session_id, prepare_spec=prepare)


def run_exact_safe_interleaved(
    config: ExactSafeInterleavedConfig,
    *,
    output_dir: Path,
    session_id: str | None = None,
) -> TimingExperimentRun:
    experiment_id = f"stage3-r12c-exact-safe-interleaved-{config.thread_tag}"

    def prepare() -> ExperimentSpec:
        data = load_mechanism_data(config.dataset, fixture=config.fixture, query_count=config.query_count, query_seed=config.query_seed)
        workload = PreparedFusedWorkload(data.flat_tokens, data.doc_starts, list(data.queries), list(data.query_ids))
        arms = [_dense_arm(workload, data.dataset, config.k, config.n_threads)]
        for checkpoints in config.checkpoint_sets:
            suffix = "-".join(str(value) for value in checkpoints)
            arms.append(
                _bond_arm(
                    workload,
                    dataset=data.dataset,
                    k=config.k,
                    n_threads=config.n_threads,
                    order=config.order,
                    level="doc",
                    checkpoints=checkpoints,
                    policy=config.policy,
                    shrink=config.shrink,
                    arm_id=f"bond-c{suffix}",
                )
            )
        return ExperimentSpec(
            experiment_id=experiment_id,
            artifact_id=f"{experiment_id}-{data.dataset}",
            dataset_id=data.dataset,
            workload_id=str(data.workload_metadata["workload_id"]),
            arms=tuple(arms),
            command=(
                "uv run python -m experiments.stage3_mechanism.r12c_interleaved_exact_safe "
                f"--dataset {config.dataset} --threads {config.n_threads}" + (" --fixture" if config.fixture else "")
            ),
            baseline_arm_id="dense-fused",
            metadata={
                "exact_safe_interleaved": asdict(config),
                "supersedes_fields": ["historical-stage3-e08.wall-clock", "historical-stage3-e09.wall-clock"],
            },
            workload_metadata=data.workload_metadata,
            provenance_inputs={
                "configuration_sha256": configuration_sha256(config),
                "data_sha256": data.data_sha256,
                "index_sha256": None,
                "input_artifact_ids": [
                    f"historical-stage3-e08-{config.dataset}",
                    f"historical-stage3-r12c-{config.thread_tag}",
                ],
            },
        )

    return _write_timing(config, experiment_id=experiment_id, output_dir=output_dir, session_id=session_id, prepare_spec=prepare)


def run_baseline_probe(
    config: BaselineProbeConfig,
    *,
    output_dir: Path,
    session_id: str | None = None,
) -> TimingExperimentRun:
    """Run the R12b standalone-baseline diagnostic under the shared protocol."""
    experiment_id = f"stage3-r12b-interleaved-baseline-probe-{config.thread_tag}"

    def prepare() -> ExperimentSpec:
        data = _load_baseline_probe_data(config)
        workload = PreparedFusedWorkload(
            data.flat_tokens,
            data.doc_starts,
            list(data.queries),
            list(data.query_ids),
        )
        arms = [_dense_arm(workload, data.dataset, config.k, config.n_threads)]
        for checkpoints in config.checkpoint_sets:
            suffix = "-".join(str(value) for value in checkpoints)
            arms.append(
                _bond_arm(
                    workload,
                    dataset=data.dataset,
                    k=config.k,
                    n_threads=config.n_threads,
                    order=config.order,
                    level="doc",
                    checkpoints=checkpoints,
                    policy=config.policy,
                    shrink=config.shrink,
                    arm_id=f"bond-c{suffix}",
                )
            )
        return ExperimentSpec(
            experiment_id=experiment_id,
            artifact_id=f"{experiment_id}-{data.dataset}",
            dataset_id=data.dataset,
            workload_id=str(data.workload_metadata["workload_id"]),
            arms=tuple(arms),
            command=(
                "uv run python -m experiments.stage3_mechanism.r12b_interleaved_baseline_probe "
                f"--dataset {config.dataset} --queries {config.query_count} "
                f"--threads {config.n_threads}"
                + (" --fixture" if config.fixture else "")
            ),
            baseline_arm_id="dense-fused",
            metadata={
                "baseline_probe": asdict(config),
                "evidence_status": "diagnostic",
                "purpose": "test standalone-baseline measurement artifact before R12c",
            },
            workload_metadata=data.workload_metadata,
            provenance_inputs={
                "configuration_sha256": configuration_sha256(config),
                "data_sha256": data.data_sha256,
                "index_sha256": None,
                "input_artifact_ids": [f"historical-stage3-e08-{config.dataset}"],
            },
        )

    return _write_timing(
        config,
        experiment_id=experiment_id,
        output_dir=output_dir,
        session_id=session_id,
        prepare_spec=prepare,
    )
