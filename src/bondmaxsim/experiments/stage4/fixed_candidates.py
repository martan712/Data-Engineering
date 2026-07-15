"""Migrated Stage 4 e01 fixed-candidate experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline
from bondmaxsim.baselines.plaid import PLAIDBaseline
from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.loader import unpack_embeddings
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
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores
from bondmaxsim.experiments.persistence import write_or_append_timing_sessions
from bondmaxsim.results.io import deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope
from bondmaxsim.testbed.config import RunConfig


@dataclass(frozen=True)
class FixedCandidateConfig:
    dataset: str = "scifact"
    budgets: tuple[int, ...] = (100, 500, 1000, 5000)
    query_count: int = 50
    query_seed: int = 42
    k: int = 10
    n_threads: int = 1
    faiss_nprobe: int = 32
    plaid_n_ivf_probe: int = 8
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "FixedCandidateConfig":
        return cls(dataset="synthetic-small-v1", budgets=(2, 4), query_count=4, k=2, fixture=True)

    @property
    def thread_tag(self) -> str:
        return f"{self.n_threads}t" if self.n_threads > 0 else "mt"


@dataclass(frozen=True)
class FixedCandidateOutput:
    comparison_scope: str
    envelope: ExperimentResultEnvelope
    output_path: Path


@dataclass(frozen=True)
class FixedCandidateRun:
    equal_work: FixedCandidateOutput
    system_cap: FixedCandidateOutput


def _fixture_candidate_operation(data: Stage4Data, budget: int, *, k: int, observable: bool) -> Callable[[], RetrievalPassResult]:
    admitted = min(budget, data.num_documents)

    def operation() -> RetrievalPassResult:
        rows = []
        candidates = np.arange(admitted, dtype=np.int64)
        for query in data.queries:
            scores = exact_maxsim_scores(query, data.flat_tokens, data.doc_starts)
            actual_k = min(len(candidates), k)
            local = topk_from_scores(scores[candidates], actual_k)[0]
            ids = candidates[local]
            returned_scores = scores[ids]
            if observable:
                work = CandidateWorkObservation(
                    configured_candidate_cap=CountObservation.exact(budget, "fixture configured candidate cap"),
                    configured_full_score_cap=CountObservation.exact(budget, "fixture rerank cap"),
                    unique_candidates_generated=CountObservation.exact(admitted, "fixture deterministic candidate set"),
                    documents_admitted_to_scoring=CountObservation.exact(admitted, "fixture deterministic candidate set"),
                    documents_fully_scored=CountObservation.exact(admitted, "fixture exact MaxSim rerank"),
                    documents_probed=CountObservation.unavailable("fixture candidate arm has no partitions"),
                    partitions_probed=CountObservation.unavailable("fixture candidate arm has no partitions"),
                    token_hits_inspected=CountObservation.unavailable("fixture candidate arm has no token index"),
                )
            else:
                unavailable = CountObservation.unavailable("fixture PLAID surrogate retains pinned backend observability limits")
                work = CandidateWorkObservation(
                    configured_candidate_cap=CountObservation.unavailable("PLAID uses a configured full-score cap"),
                    configured_full_score_cap=CountObservation.exact(budget, "fixture PLAID n_full_scores"),
                    unique_candidates_generated=unavailable,
                    documents_admitted_to_scoring=unavailable,
                    documents_fully_scored=unavailable,
                    documents_probed=unavailable,
                    partitions_probed=unavailable,
                    token_hits_inspected=unavailable,
                )
            rows.append(RetrievalQueryResult(ids, returned_scores, work))
        return RetrievalPassResult(tuple(rows))

    return operation


def _faiss_operation(data: Stage4Data, baseline: FaissIVFBaseline, budget: int) -> Callable[[], RetrievalPassResult]:
    def operation() -> RetrievalPassResult:
        rows = []
        for query in data.queries:
            ids, scores, work = baseline.topk_with_work(query, k=data.exact_ids[0].size, candidate_budget=budget)
            work.require_equal_work_reranking()
            rows.append(RetrievalQueryResult(ids, scores, work))
        return RetrievalPassResult(tuple(rows))

    return operation


def _plaid_operation(data: Stage4Data, baseline: PLAIDBaseline, budget: int) -> Callable[[], RetrievalPassResult]:
    def operation() -> RetrievalPassResult:
        results, work = baseline.search_with_work(list(data.queries), k=data.exact_ids[0].size, comparison_scope="system_cap")
        return RetrievalPassResult(tuple(
            RetrievalQueryResult(ids, scores, observation)
            for (ids, scores), observation in zip(results, work)
        ))

    return operation


def run_fixed_candidate_arms(
    config: FixedCandidateConfig,
    *,
    output_dir: Path,
    session_id: str | None = None,
    environment: dict[str, Any] | None = None,
) -> FixedCandidateRun:
    """Run and serialize distinct strict-equal-work and system-cap sessions."""
    data_cache: Stage4Data | None = None
    prepared_cache: dict[str, Any] | None = None

    def prepare_shared() -> tuple[Stage4Data, dict[str, Any]]:
        nonlocal data_cache, prepared_cache
        if data_cache is not None and prepared_cache is not None:
            return data_cache, prepared_cache
        data = load_stage4_data(
            dataset=config.dataset,
            query_count=config.query_count,
            query_seed=config.query_seed,
            k=config.k,
            fixture=config.fixture,
        )
        budgets = tuple(sorted({min(value, data.num_documents) for value in config.budgets if value >= config.k}))
        if not budgets:
            raise ValueError("at least one candidate budget must be >= k")
        workload = PreparedFusedWorkload(data.flat_tokens, data.doc_starts, list(data.queries), list(data.query_ids))
        dense = workload.prepare_arm(
            RunConfig(dataset=config.dataset, method="fused_panel_maxsim_bond", dimension_order="natural", threshold_policy="none", k=config.k, shrink=1.0),
            scanner="brute",
            n_threads=config.n_threads,
        )
        historical = None
        if not config.fixture:
            historical = assert_historical_controls(
                REPO_ROOT / "results" / "json" / f"stage4_integration_e01_fixed_candidate_arms_{config.dataset}.json",
                {
                    "experiment": "e01_fixed_candidate_arms",
                    "dataset": config.dataset,
                    "n_queries": config.query_count,
                    "query_seed": config.query_seed,
                    "k": config.k,
                    "budgets": list(budgets),
                    "kernel": {
                        "bound": "tight",
                        "order": "natural",
                        "checkpoints": [112],
                        "shrink": 1.0,
                    },
                },
            )
        if config.fixture:
            equal_operations = {budget: _fixture_candidate_operation(data, budget, k=config.k, observable=True) for budget in budgets}
            system_operations = {budget: _fixture_candidate_operation(data, budget, k=config.k, observable=False) for budget in budgets}
            compatibility = {"fixture_backend": True, "actual_full_score_count_available": False}
        else:
            faiss = FaissIVFBaseline(data.flat_tokens, data.doc_starts, nprobe=config.faiss_nprobe)
            faiss.build(cache_path=str(REPO_ROOT / "data" / "faiss_indexes" / f"{config.dataset}.faiss"))
            equal_operations = {budget: _faiss_operation(data, faiss, budget) for budget in budgets}
            system_operations = {}
            compatibility = None
            for budget in budgets:
                plaid = PLAIDBaseline(
                    config.dataset,
                    n_ivf_probe=config.plaid_n_ivf_probe,
                    n_full_scores=budget,
                    num_threads=config.n_threads if config.n_threads > 0 else None,
                )
                if plaid.exists():
                    plaid.load()
                else:
                    plaid.build(unpack_embeddings(data.flat_tokens, data.doc_starts))
                system_operations[budget] = _plaid_operation(data, plaid, budget)
                compatibility = compatibility or plaid.compatibility_record()
        data_cache = data
        prepared_cache = {
            "data": data,
            "budgets": budgets,
            "dense": dense,
            "equal_operations": equal_operations,
            "system_operations": system_operations,
            "historical": historical,
            "plaid_compatibility": compatibility,
        }
        return data, prepared_cache

    def spec_for(scope: str) -> ExperimentSpec:
        data, prepared = prepare_shared()
        dense = prepared["dense"]
        arms = [
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
        operations = prepared["equal_operations"] if scope == "equal_work_reranking" else prepared["system_operations"]
        for budget, operation in operations.items():
            prefix = "faiss" if scope == "equal_work_reranking" else "plaid"
            arms.append(
                ArmSpec(
                    arm_id=f"{prefix}-b{budget:05d}",
                    method_family="faiss-ivf-rerank" if prefix == "faiss" else "plaid-fastplaid",
                    operation=operation,
                    parameters={
                        "configured_budget": budget,
                        "thread_tag": config.thread_tag,
                        "actual_work_requirement": "exact documents_fully_scored" if prefix == "faiss" else "unavailable",
                    },
                    exactness="approximate",
                    comparison_scope=scope,
                    validator=lambda result, data=data: validate_retrieval(result, data, config.k),
                    metadata_extractor=lambda result, ids=data.query_ids: retrieval_metadata(result, ids),
                    timer_scope_id="candidate-workload-pass",
                )
            )
        scope_id = "equal-work" if scope == "equal_work_reranking" else "system-cap"
        experiment_id = f"stage4-e01-fixed-candidates-{scope_id}-{config.thread_tag}"
        command = (
            "uv run python -m experiments.stage4_integration.e01_fixed_candidate_arms "
            f"--dataset {config.dataset} --threads {config.n_threads}"
            + (" --fixture" if config.fixture else "")
        )
        return ExperimentSpec(
            experiment_id=experiment_id,
            artifact_id=f"{experiment_id}-{config.dataset}",
            dataset_id=config.dataset,
            workload_id=data.workload_metadata["workload_id"],
            arms=tuple(arms),
            command=command,
            baseline_arm_id="dense-fused",
            metadata={
                "fixed_candidate_configuration": asdict(config),
                "output_semantics": scope,
                "historical_methodology_check": prepared["historical"],
                "plaid_compatibility": prepared["plaid_compatibility"] if scope == "system_cap" else None,
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
    root_session = session_id or new_session_id("stage4-e01")
    outputs = []
    for scope, suffix in (("equal_work_reranking", "equal"), ("system_cap", "system")):
        envelope, _, spec = execute_prepared_timing_experiment(
            lambda scope=scope: spec_for(scope),
            protocol,
            session_id=f"{root_session}-{suffix}",
            build_profile="release-native",
            environment=environment,
        )
        path = output_dir / deterministic_result_name(spec.experiment_id, spec.dataset_id, spec.workload_id)
        merged = write_or_append_timing_sessions(path, envelope)
        outputs.append(FixedCandidateOutput(scope, merged, path))
    return FixedCandidateRun(outputs[0], outputs[1])
