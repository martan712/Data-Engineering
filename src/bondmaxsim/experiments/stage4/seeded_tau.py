"""Migrated Stage 4 e02 seeded-threshold recovery experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline
from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.packing import build_qcum
from bondmaxsim.experiments.arms import ArmSpec, ExperimentSpec
from bondmaxsim.experiments.candidate_work import CandidateWorkObservation, CountObservation
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
from bondmaxsim.kernels.fused_panel import run_fused_panel_bond_validated
from bondmaxsim.experiments.persistence import write_or_append_timing_sessions
from bondmaxsim.results.io import deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.threshold.policies import candidate_seed_threshold

_TAU_EPS = 1e-3


@dataclass(frozen=True)
class SeedConfiguration:
    nprobe: int
    k_token: int
    candidate_cap: int


@dataclass(frozen=True)
class SeededTauConfig:
    dataset: str = "scifact"
    query_count: int = 50
    query_seed: int = 42
    k: int = 10
    n_threads: int = 1
    checkpoints: tuple[int, ...] = (112,)
    cheap: SeedConfiguration = SeedConfiguration(4, 32, 10)
    strong: SeedConfiguration = SeedConfiguration(16, 128, 100)
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "SeededTauConfig":
        return cls(
            dataset="synthetic-small-v1",
            query_count=4,
            k=2,
            checkpoints=(8,),
            cheap=SeedConfiguration(1, 2, 2),
            strong=SeedConfiguration(1, 4, 4),
            fixture=True,
        )

    @property
    def thread_tag(self) -> str:
        return f"{self.n_threads}t" if self.n_threads > 0 else "mt"


@dataclass(frozen=True)
class SeedPassResult:
    taus: tuple[float, ...]
    candidate_work: tuple[CandidateWorkObservation, ...]


@dataclass(frozen=True)
class SeededTauRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


class _Seeder:
    def __init__(self, data: Stage4Data, config: SeedConfiguration, baseline: FaissIVFBaseline | None) -> None:
        self.data = data
        self.config = config
        self.baseline = baseline

    def operation(self) -> SeedPassResult:
        taus, work = [], []
        for query in self.data.queries:
            if self.baseline is None:
                admitted = min(self.config.candidate_cap, self.data.num_documents)
                candidates = np.arange(admitted, dtype=np.int64)
                from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores

                scores = exact_maxsim_scores(query, self.data.flat_tokens, self.data.doc_starts)[candidates]
                observation = CandidateWorkObservation(
                    configured_candidate_cap=CountObservation.exact(self.config.candidate_cap, "fixture seed cap"),
                    configured_full_score_cap=CountObservation.exact(self.config.candidate_cap, "fixture seed exact-score cap"),
                    unique_candidates_generated=CountObservation.exact(admitted, "fixture seed candidates"),
                    documents_admitted_to_scoring=CountObservation.exact(admitted, "fixture seed candidates"),
                    documents_fully_scored=CountObservation.exact(admitted, "fixture exact seed scoring"),
                    documents_probed=CountObservation.unavailable("seed has no partitions"),
                    partitions_probed=CountObservation.unavailable("seed has no partitions"),
                    token_hits_inspected=CountObservation.unavailable("fixture seed has no token index"),
                )
            else:
                _, scores, observation = self.baseline.topk_with_work(
                    query,
                    k=self.data.exact_ids[0].size,
                    candidate_budget=self.config.candidate_cap,
                    k_token=self.config.k_token,
                )
            taus.append(candidate_seed_threshold(scores, self.data.exact_ids[0].size) - _TAU_EPS)
            work.append(observation)
        return SeedPassResult(tuple(taus), tuple(work))

    @staticmethod
    def metadata(result: SeedPassResult) -> dict[str, Any]:
        return {
            "cost_representation": "direct_seed_only_observation",
            "candidate_work": [row.to_dict() for row in result.candidate_work],
        }


class _TauKernel:
    def __init__(self, data: Stage4Data, config: SeededTauConfig) -> None:
        from bondmaxsim.testbed.packing_cache import PackingCache
        from bondmaxsim.kernels.fused_panel import load_fused_panel_kernel

        self.data = data
        self.config = config
        self.packing = PackingCache(data.flat_tokens, data.doc_starts)
        self.lib = load_fused_panel_kernel()
        self.checkpoints = np.asarray(config.checkpoints, dtype=np.uint32)
        self.prepared = []
        for query in data.queries:
            corpus, effective, order = self.packing.dispatch_order_panel_corpus(query, "natural")
            self.prepared.append((corpus, effective, order, build_qcum(effective, order)))

    def run(self, taus: tuple[float, ...]) -> RetrievalPassResult:
        rows = []
        for (corpus, query, order, cumulative), tau in zip(self.prepared, taus):
            ids, scores, stats = run_fused_panel_bond_validated(
                self.lib,
                corpus,
                query,
                order,
                cumulative,
                shrink=1.0,
                tau_seed=tau,
                K=self.config.k,
                n_threads=self.config.n_threads,
                level="doc",
                checkpoints=self.checkpoints,
                bound="tight",
            )
            rows.append(
                RetrievalQueryResult(
                    ids,
                    scores,
                    accounting={
                        "documents_pruned": int(stats[1]),
                        "cells_scanned": int(stats[0]),
                    },
                )
            )
        return RetrievalPassResult(tuple(rows))

    def metadata(self, result: RetrievalPassResult) -> dict[str, Any]:
        return retrieval_metadata(result, self.data.query_ids)


def run_seeded_tau_recovery(
    config: SeededTauConfig,
    *,
    output_dir: Path,
    session_id: str | None = None,
    environment: dict[str, Any] | None = None,
) -> SeededTauRun:
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
                REPO_ROOT / "results" / "json" / f"stage4_integration_e02_seeded_tau_recovery_{config.dataset}.json",
                {
                    "experiment": "e02_seeded_tau_recovery",
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
                parameters={"cost_component": "reference", "thread_tag": config.thread_tag},
                exactness="reference",
                comparison_scope="uncapped_reference",
                validator=dense.validate,
                metadata_extractor=dense.metadata,
                timer_scope_id="native-workload-pass",
            )
        ]
        for arm_id, policy in (("self-bound-kernel", "self_bound"), ("seed-partial-kernel", "seed"), ("oracle-kernel", "oracle")):
            prepared = workload.prepare_arm(
                RunConfig(
                    dataset=config.dataset,
                    method="fused_panel_maxsim_bond",
                    dimension_order="natural",
                    threshold_policy=policy,
                    k=config.k,
                    shrink=1.0,
                    checkpoints=config.checkpoints,
                ),
                scanner="bond",
                n_threads=config.n_threads,
            )
            arms.append(
                ArmSpec(
                    arm_id=arm_id,
                    method_family="fused-panel-bond",
                    operation=prepared.operation,
                    parameters={"threshold_policy": policy, "cost_component": "kernel_only", "total_cost_claim_allowed": False},
                    exactness="exact_safe",
                    comparison_scope="uncapped_reference",
                    validator=prepared.validate,
                    metadata_extractor=prepared.metadata,
                    timer_scope_id="native-kernel-workload-pass",
                )
            )
        kernel = _TauKernel(data, config)
        for seed_name, seed_config in (("cheap", config.cheap), ("strong", config.strong)):
            baseline = None
            if not config.fixture:
                baseline = FaissIVFBaseline(data.flat_tokens, data.doc_starts, nprobe=seed_config.nprobe)
                baseline.build(cache_path=str(REPO_ROOT / "data" / "faiss_indexes" / f"{config.dataset}.faiss"))
                baseline.index.nprobe = seed_config.nprobe
            seeder = _Seeder(data, seed_config, baseline)
            static_seed = seeder.operation()

            def kernel_only(seed=static_seed) -> RetrievalPassResult:
                return kernel.run(seed.taus)

            def end_to_end(seeder=seeder) -> RetrievalPassResult:
                fresh = seeder.operation()
                result = kernel.run(fresh.taus)
                return RetrievalPassResult(tuple(
                    RetrievalQueryResult(row.ids, row.scores, work, row.accounting)
                    for row, work in zip(result.queries, fresh.candidate_work)
                ))

            shared_parameters = {
                "seed_configuration": asdict(seed_config),
                "threshold_policy": f"ivf_seed_{seed_name}",
            }
            arms.extend(
                [
                    ArmSpec(
                        arm_id=f"ivf-seed-{seed_name}-only",
                        method_family="faiss-ivf-seed",
                        operation=seeder.operation,
                        parameters={**shared_parameters, "cost_component": "seed_only", "total_cost_claim_allowed": False},
                        exactness="approximate",
                        comparison_scope="system_cap",
                        metadata_extractor=seeder.metadata,
                        timer_scope_id="seed-workload-pass",
                    ),
                    ArmSpec(
                        arm_id=f"ivf-seed-{seed_name}-kernel",
                        method_family="fused-panel-bond",
                        operation=kernel_only,
                        parameters={**shared_parameters, "cost_component": "kernel_only", "total_cost_claim_allowed": False},
                        exactness="exact_safe",
                        comparison_scope="uncapped_reference",
                        validator=lambda result, data=data: validate_retrieval(result, data, config.k),
                        metadata_extractor=kernel.metadata,
                        timer_scope_id="native-kernel-workload-pass",
                    ),
                    ArmSpec(
                        arm_id=f"ivf-seed-{seed_name}-end-to-end",
                        method_family="seeded-fused-panel-bond",
                        operation=end_to_end,
                        parameters={**shared_parameters, "cost_component": "direct_end_to_end", "derived_by_summing_components": False},
                        exactness="exact_safe",
                        comparison_scope="system_cap",
                        validator=lambda result, data=data: validate_retrieval(result, data, config.k),
                        metadata_extractor=lambda result, ids=data.query_ids: retrieval_metadata(result, ids),
                        timer_scope_id="seed-and-native-workload-pass",
                    ),
                ]
            )
        experiment_id = f"stage4-e02-seeded-tau-recovery-{config.thread_tag}"
        return ExperimentSpec(
            experiment_id=experiment_id,
            artifact_id=f"{experiment_id}-{config.dataset}",
            dataset_id=config.dataset,
            workload_id=data.workload_metadata["workload_id"],
            arms=tuple(arms),
            command=(
                "uv run python -m experiments.stage4_integration.e02_seeded_tau_recovery "
                f"--dataset {config.dataset} --threads {config.n_threads}"
                + (" --fixture" if config.fixture else "")
            ),
            baseline_arm_id="dense-fused",
            metadata={
                "seeded_tau_configuration": asdict(config),
                "cost_interpretation": "seed-only and kernel-only observations are never summed; end-to-end costs are directly observed",
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
        session_id=session_id or new_session_id("stage4-e02"),
        build_profile="release-native",
        environment=environment,
    )
    path = output_dir / deterministic_result_name(spec.experiment_id, spec.dataset_id, spec.workload_id)
    merged = write_or_append_timing_sessions(path, envelope)
    return SeededTauRun(merged, path)
