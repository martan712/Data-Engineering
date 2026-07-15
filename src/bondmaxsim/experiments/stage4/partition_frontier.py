"""Migrated Stage 4 e03 partition-probe frontier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.arms import ArmSpec, ExperimentSpec
from bondmaxsim.experiments.execution import execute_prepared_timing_experiment
from bondmaxsim.experiments.mechanism import PreparedFusedWorkload
from bondmaxsim.experiments.stage4.common import (
    RetrievalPassResult,
    RetrievalQueryResult,
    Stage4Data,
    assert_historical_controls,
    configuration_sha256,
    load_stage4_data,
    retrieval_metadata,
    validate_retrieval,
)
from bondmaxsim.experiments.timing import TimingProtocol, new_session_id
from bondmaxsim.kernels.fused_panel import load_fused_panel_kernel
from bondmaxsim.partitioned_scan import PartitionedFusedScan
from bondmaxsim.experiments.persistence import write_or_append_timing_sessions
from bondmaxsim.results.io import deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope
from bondmaxsim.testbed.config import RunConfig


@dataclass(frozen=True)
class PartitionFrontierConfig:
    dataset: str = "scifact"
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    query_count: int = 50
    query_seed: int = 42
    k: int = 10
    n_threads: int = 1
    checkpoints: tuple[int, ...] = (112,)
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "PartitionFrontierConfig":
        return cls(
            dataset="synthetic-small-v1",
            nprobes=(1, 2),
            query_count=4,
            k=2,
            checkpoints=(8,),
            fixture=True,
        )

    @property
    def thread_tag(self) -> str:
        return f"{self.n_threads}t" if self.n_threads > 0 else "mt"


@dataclass(frozen=True)
class PartitionFrontierRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


class _PartitionArm:
    def __init__(
        self,
        data: Stage4Data,
        index: PartitionedFusedScan,
        *,
        scanner: str,
        nprobe: int,
        config: PartitionFrontierConfig,
    ) -> None:
        self.data = data
        self.index = index
        self.scanner = scanner
        self.nprobe = nprobe
        self.config = config
        self.lib = load_fused_panel_kernel()

    def operation(self) -> RetrievalPassResult:
        rows = []
        for query in self.data.queries:
            ids, scores, stats = self.index.search(
                self.lib,
                query,
                k=self.config.k,
                nprobe=self.nprobe,
                checkpoints=self.config.checkpoints,
                bound="tight",
                n_threads=self.config.n_threads,
                scanner=self.scanner,
            )
            rows.append(
                RetrievalQueryResult(
                    ids,
                    scores,
                    candidate_work=_candidate_work_from_stats(stats),
                    accounting={
                        "documents_probed": stats["documents_probed"],
                        "partitions_probed": stats["partitions_probed"],
                        "documents_probed_pct": stats["docs_probed_pct"],
                        "documents_pruned_pct_of_probed": stats["docs_pruned_pct_of_probed"],
                    },
                )
            )
        return RetrievalPassResult(tuple(rows))

    def metadata(self, result: RetrievalPassResult) -> dict[str, Any]:
        metadata = retrieval_metadata(result, self.data.query_ids)
        metadata["accounting_semantics"] = "exact counts from selected partition offsets and probe-list length"
        return metadata


def _candidate_work_from_stats(stats: dict[str, Any]):
    from bondmaxsim.experiments.candidate_work import CandidateWorkObservation, CountObservation

    source = stats["candidate_work"]
    observations = {
        name: CountObservation(value=value["value"], quality=value["quality"], source=value["source"])
        for name, value in source.items()
    }
    return CandidateWorkObservation(**observations)


def run_partition_frontier(
    config: PartitionFrontierConfig,
    *,
    output_dir: Path,
    session_id: str | None = None,
    environment: dict[str, Any] | None = None,
) -> PartitionFrontierRun:
    def prepare() -> ExperimentSpec:
        data = load_stage4_data(
            dataset=config.dataset,
            query_count=config.query_count,
            query_seed=config.query_seed,
            k=config.k,
            fixture=config.fixture,
        )
        historical = None
        if not config.fixture:
            historical = assert_historical_controls(
                REPO_ROOT / "results" / "json" / f"stage4_integration_e03_partitioned_fused_scan_{config.dataset}.json",
                {
                    "experiment": "e03_partitioned_fused_scan",
                    "dataset": config.dataset,
                    "n_queries": config.query_count,
                    "query_seed": config.query_seed,
                    "k": config.k,
                    "kernel": {
                        "bound": "tight",
                        "order": "natural",
                        "checkpoints": list(config.checkpoints),
                        "shrink": 1.0,
                    },
                },
            )
        index = PartitionedFusedScan(
            data.flat_tokens,
            data.doc_starts,
            n_partitions=2 if config.fixture else None,
        )
        build_stats = index.build()
        partition_count = int(build_stats["n_partitions_nonempty"])
        nprobes = tuple(sorted({min(value, partition_count) for value in config.nprobes} | {partition_count}))
        bound_prunable, bound_slack = [], []
        for query, exact_scores in zip(data.queries, data.exact_scores):
            centroid_score = (query @ index.centroids.T).sum(axis=0)
            upper_bound = centroid_score + len(query) * index.radii
            tau = float(exact_scores[-1])
            bound_prunable.append(100.0 * float(np.mean(upper_bound < tau)))
            bound_slack.append(float(np.min(upper_bound) - tau))
        workload = PreparedFusedWorkload(data.flat_tokens, data.doc_starts, list(data.queries), list(data.query_ids))
        dense = workload.prepare_arm(
            RunConfig(dataset=config.dataset, method="fused_panel_maxsim_bond", dimension_order="natural", threshold_policy="none", k=config.k, shrink=1.0),
            scanner="brute",
            n_threads=config.n_threads,
        )
        arms: list[ArmSpec] = [
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
        ]
        for scanner in ("brute", "bond"):
            for nprobe in nprobes:
                prepared = _PartitionArm(data, index, scanner=scanner, nprobe=nprobe, config=config)
                arms.append(
                    ArmSpec(
                        arm_id=f"partition-{scanner}-p{nprobe:03d}",
                        method_family=f"partitioned-fused-{scanner}",
                        operation=prepared.operation,
                        parameters={
                            "scanner": scanner,
                            "nprobe": nprobe,
                            "partition_count": partition_count,
                            "is_full_probe": nprobe == partition_count,
                            "count_semantics": "exact",
                        },
                        exactness="exact_safe" if nprobe == partition_count else "approximate",
                        comparison_scope="probe_frontier",
                        validator=lambda result, data=data: validate_retrieval(result, data, config.k),
                        metadata_extractor=prepared.metadata,
                        timer_scope_id="partitioned-native-workload-pass",
                    )
                )
        experiment_id = f"stage4-e03-partition-frontier-{config.thread_tag}"
        return ExperimentSpec(
            experiment_id=experiment_id,
            artifact_id=f"{experiment_id}-{config.dataset}",
            dataset_id=config.dataset,
            workload_id=data.workload_metadata["workload_id"],
            arms=tuple(arms),
            command=(
                "uv run python -m experiments.stage4_integration.e03_partitioned_fused_scan "
                f"--dataset {config.dataset} --threads {config.n_threads}"
                + (" --fixture" if config.fixture else "")
            ),
            baseline_arm_id="dense-fused",
            metadata={
                "partition_frontier_configuration": asdict(config),
                "partition_build": build_stats,
                "exact_safe_partition_bound": {
                    "prunable_partitions_pct_oracle_tau": float(np.mean(bound_prunable)),
                    "min_ub_minus_oracle_tau_mean": float(np.mean(bound_slack)),
                },
                "historical_methodology_check": historical,
            },
            workload_metadata=data.workload_metadata,
            provenance_inputs={
                "configuration_sha256": configuration_sha256(config),
                "data_sha256": data.data_sha256,
                "index_sha256": None,
                "input_artifact_ids": [],
            },
        )

    protocol = TimingProtocol.preset("fixture" if config.fixture else "full_system")
    envelope, _, spec = execute_prepared_timing_experiment(
        prepare,
        protocol,
        session_id=session_id or new_session_id("stage4-e03"),
        build_profile="release-native",
        environment=environment,
    )
    path = output_dir / deterministic_result_name(spec.experiment_id, spec.dataset_id, spec.workload_id)
    merged = write_or_append_timing_sessions(path, envelope)
    return PartitionFrontierRun(merged, path)
