"""Declarative Stage 5 e01 execution on the shared experiment framework."""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline
from bondmaxsim.baselines.plaid import PLAIDBaseline
from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_ids, load_qrels_tsv, qrels_path
from bondmaxsim.data.fixture import fixture_arrays
from bondmaxsim.data.loader import load_dataset, load_eval_queries, unpack_embeddings
from bondmaxsim.experiments.arms import ArmSpec, ExperimentSpec
from bondmaxsim.experiments.candidate_work import (
    CandidateWorkObservation,
    CountObservation,
)
from bondmaxsim.experiments.execution import execute_prepared_timing_experiment
from bondmaxsim.experiments.mechanism import PreparedFusedArm, PreparedFusedWorkload
from bondmaxsim.experiments.stage5.quality import (
    RankedQueryResult,
    RetrievalPassResult,
    evaluate_pass,
    validate_exact_pass,
    validate_ranked_pass,
)
from bondmaxsim.experiments.stage5.persistence import write_or_append_timing_sessions
from bondmaxsim.experiments.timing import TimingProtocol, new_session_id
from bondmaxsim.experiments.workloads import WorkloadMetadata
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, exact_maxsim_topk, topk_from_scores
from bondmaxsim.partitioned_scan import PartitionedFusedScan
from bondmaxsim.results.io import deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope, IRQualityPayload
from bondmaxsim.testbed.config import RunConfig


DATASETS = ("scifact", "nfcorpus", "arguana", "scidocs")


@dataclass(frozen=True)
class Stage5IREvaluationConfig:
    dataset: str = "scifact"
    n_threads: int = 1
    k_retrieve: int = 100
    k_eval: int = 10
    checkpoints: tuple[int, ...] = (112,)
    partition_nprobes: tuple[int, ...] = (16, 32)
    candidate_caps: tuple[int, ...] = (100, 1000, 5000)
    faiss_nprobe: int = 32
    plaid_n_ivf_probe: int = 8
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "Stage5IREvaluationConfig":
        return cls(
            dataset="synthetic-small-v1",
            n_threads=1,
            k_retrieve=6,
            k_eval=3,
            checkpoints=(8,),
            partition_nprobes=(1, 2),
            candidate_caps=(),
            fixture=True,
        )

    @property
    def thread_tag(self) -> str:
        return "mt" if self.n_threads == 0 else f"{self.n_threads}t"


@dataclass(frozen=True)
class Stage5IRRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


@dataclass
class _Prepared:
    spec: ExperimentSpec
    latest: dict[str, RetrievalPassResult]
    qrels: dict[str, dict[str, int]]
    corpus_ids: list[str]
    exact_ids: tuple[np.ndarray, ...]
    exact_scores: tuple[np.ndarray, ...]
    exact_full_scores: tuple[np.ndarray, ...]
    query_ids: tuple[str, ...]
    num_documents: int


def _sha256_bytes(chunks: Iterator[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(config: Stage5IREvaluationConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _path_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        with path.open("rb") as handle:
            return _sha256_bytes(iter(lambda: handle.read(8 * 1024 * 1024), b""))
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(str(candidate.relative_to(path)).encode())
        with candidate.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _retrieval_data_sha256(
    flat: np.ndarray,
    starts: np.ndarray,
    queries: list[np.ndarray],
    query_ids: list[str],
    corpus_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> str:
    """Hash the effective ordered arrays, IDs, and judgments used by e01."""
    digest = hashlib.sha256()
    for array in (flat, starts, *queries):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    digest.update(json.dumps(query_ids, separators=(",", ":")).encode())
    digest.update(json.dumps(corpus_ids, separators=(",", ":")).encode())
    digest.update(json.dumps(qrels, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _partition_index_sha256(index: PartitionedFusedScan) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(index.centroids).tobytes())
    digest.update(np.ascontiguousarray(index.radii).tobytes())
    for partition in index.partitions:
        digest.update(np.ascontiguousarray(partition.doc_ids).tobytes())
    return digest.hexdigest()


def _fixture_inputs(config: Stage5IREvaluationConfig):
    arrays = fixture_arrays()
    starts = arrays["query_starts"]
    values = arrays["query_values"]
    queries = [
        values[int(start): int(starts[index + 1]) if index + 1 < len(starts) else len(values)]
        for index, start in enumerate(starts)
    ]
    query_ids = [f"q{index}" for index in range(len(queries))]
    corpus_ids = [f"d{index}" for index in range(len(arrays["doc_starts"]))]
    qrels = {"q0": {"d0": 2}, "q1": {"d2": 1}, "q2": {"d4": 1}, "q3": {"d5": 2}}
    workload = {
        "workload_id": "synthetic-small-qrels-test-v1",
        "dataset": config.dataset,
        "regime": "fixture",
        "ordered_query_id_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
        "sample_size": len(query_ids),
        "token_counts": [len(query) for query in queries],
        "token_count_mean": float(np.mean([len(query) for query in queries])),
        "token_count_median": float(np.median([len(query) for query in queries])),
        "token_count_min": min(len(query) for query in queries),
        "token_count_max": max(len(query) for query in queries),
        "selection": "all deterministic fixture qrels queries in source order",
        "source_revision": "bondmaxsim.fixture-manifest-1.0.0",
        "source_split": "qrels-test",
    }
    data_sha = _retrieval_data_sha256(
        arrays["doc_values"], arrays["doc_starts"], queries, query_ids,
        corpus_ids, qrels,
    )
    return (
        arrays["doc_values"], arrays["doc_starts"], queries, query_ids,
        corpus_ids, qrels, workload, data_sha,
    )


def _full_inputs(config: Stage5IREvaluationConfig):
    flat, starts, _ = load_dataset(config.dataset)
    queries, query_ids = load_eval_queries(config.dataset)
    corpus_ids = list(load_ids(config.dataset)["corpus_ids"])
    qrels = load_qrels_tsv(qrels_path(config.dataset))
    workload = WorkloadMetadata.from_queries(
        dataset=config.dataset,
        regime="qrels_test",
        query_ids=query_ids,
        queries=queries,
        encoding_configuration={
            "model": "lightonai/GTE-ModernColBERT-v1",
            "dimension": int(flat.shape[1]),
            "l2_normalized": True,
            "document_text": "text-only",
        },
        selection="all encoded queries with test qrels, source order",
        source_revision="archive/preliminaries/02_bond_variance/.cache",
        source_split="qrels-test",
    ).to_dict()
    data_sha = _retrieval_data_sha256(
        flat, starts, queries, query_ids, corpus_ids, qrels
    )
    return flat, starts, queries, query_ids, corpus_ids, qrels, workload, data_sha


def _work_observations(
    result: RetrievalPassResult,
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": row.query_id,
            **(row.candidate_work or CandidateWorkObservation.empty().to_dict()),
        }
        for row in result.queries
    ]


def _metadata(result: RetrievalPassResult) -> dict[str, Any]:
    return {
        "candidate_work": [
            row.candidate_work or CandidateWorkObservation.empty().to_dict()
            for row in result.queries
        ],
        "query_ids": [row.query_id for row in result.queries],
    }


def _reference_work(num_documents: int, source: str) -> dict[str, Any]:
    unavailable = CountObservation.unavailable(f"{source} does not generate candidates")
    exact = CountObservation.exact(num_documents, source)
    return CandidateWorkObservation(
        configured_candidate_cap=unavailable,
        configured_full_score_cap=unavailable,
        unique_candidates_generated=unavailable,
        documents_admitted_to_scoring=exact,
        documents_fully_scored=exact,
        documents_probed=exact,
        partitions_probed=CountObservation.unavailable(f"{source} is an unpartitioned scan"),
        token_hits_inspected=CountObservation.unavailable(f"{source} does not inspect IVF hits"),
    ).to_dict()


def _native_pass(
    prepared: PreparedFusedArm,
    query_ids: tuple[str, ...],
    num_documents: int,
) -> RetrievalPassResult:
    result = prepared.operation()
    rows = []
    for query_id, query_result in zip(query_ids, result.queries, strict=True):
        pruned = int(query_result.stats[1]) if prepared.scanner == "bond" else 0
        work = _reference_work(num_documents, prepared.scanner)
        work["documents_fully_scored"] = CountObservation.exact(
            num_documents - pruned,
            "documents not terminated by the fused document scanner",
        ).to_dict()
        rows.append(
            RankedQueryResult(
                query_id,
                query_result.ids,
                query_result.scores,
                candidate_work=work,
                accounting={"documents_pruned": pruned},
            )
        )
    return RetrievalPassResult(tuple(rows))


def _capture(
    arm_id: str,
    operation,
    latest: dict[str, RetrievalPassResult],
):
    def run() -> RetrievalPassResult:
        result = operation()
        latest[arm_id] = result
        return result
    return run


def prepare_ir_evaluation(config: Stage5IREvaluationConfig) -> _Prepared:
    """Prepare every corpus-side object and return explicit single-pass arms."""
    loaded = _fixture_inputs(config) if config.fixture else _full_inputs(config)
    flat, starts, queries, query_ids_list, corpus_ids, qrels, workload, data_sha = loaded
    queries = [np.ascontiguousarray(query, dtype=np.float32) for query in queries]
    query_ids = tuple(query_ids_list)
    num_documents = len(starts)
    if not any(query_id in qrels for query_id in query_ids):
        raise ValueError(f"{config.dataset}: qrels do not cover the selected workload")

    exact_full_scores = tuple(exact_maxsim_scores(query, flat, starts) for query in queries)
    exact_pairs = tuple(topk_from_scores(scores, config.k_retrieve) for scores in exact_full_scores)
    exact_ids = tuple(pair[0] for pair in exact_pairs)
    exact_scores = tuple(pair[1] for pair in exact_pairs)
    latest: dict[str, RetrievalPassResult] = {}
    arms: list[ArmSpec] = []

    fused = PreparedFusedWorkload(flat, starts, queries, list(query_ids))
    dense_config = RunConfig(
        dataset=config.dataset,
        method="fused_panel_maxsim_bond",
        dimension_order="natural",
        threshold_policy="none",
        k=config.k_retrieve,
        shrink=1.0,
    )
    dense = fused.prepare_arm(dense_config, scanner="brute", n_threads=config.n_threads)

    def exact_validator(result: RetrievalPassResult):
        return validate_exact_pass(
            result,
            query_ids=query_ids,
            exact_ids=exact_ids,
            exact_scores=exact_scores,
            exact_full_scores=exact_full_scores,
            num_documents=num_documents,
            # Exact-safe retrieval feeds both @10 and @100 metrics, so gate
            # the complete retrieved ranking rather than only the evaluation
            # cutoff used by the headline nDCG/MRR metrics.
            k=config.k_retrieve,
        )

    dense_id = "dense-fused"
    arms.append(
        ArmSpec(
            arm_id=dense_id,
            method_family="fused-panel-dense",
            operation=_capture(
                dense_id,
                lambda: _native_pass(dense, query_ids, num_documents),
                latest,
            ),
            parameters={"k": config.k_retrieve, "thread_tag": config.thread_tag},
            exactness="exact_safe",
            comparison_scope="uncapped_reference",
            validator=exact_validator,
            metadata_extractor=_metadata,
            timer_scope_id="retrieval-workload-pass",
        )
    )

    openblas_id = "openblas-exact"
    def openblas_operation() -> RetrievalPassResult:
        rows = []
        for query_id, query in zip(query_ids, queries, strict=True):
            ids, scores = exact_maxsim_topk(query, flat, starts, k=config.k_retrieve)
            rows.append(RankedQueryResult(query_id, ids, scores, _reference_work(num_documents, "NumPy/OpenBLAS exact scan")))
        return RetrievalPassResult(tuple(rows))
    arms.append(
        ArmSpec(
            arm_id=openblas_id,
            method_family="numpy-openblas-maxsim",
            operation=_capture(openblas_id, openblas_operation, latest),
            parameters={"k": config.k_retrieve, "thread_tag": config.thread_tag},
            exactness="exact_safe",
            comparison_scope="uncapped_reference",
            validator=exact_validator,
            metadata_extractor=_metadata,
            timer_scope_id="retrieval-workload-pass",
        )
    )

    bond_config = RunConfig(
        dataset=config.dataset,
        method="fused_panel_maxsim_bond",
        dimension_order="natural",
        threshold_policy="self_bound",
        k=config.k_retrieve,
        shrink=1.0,
        checkpoints=config.checkpoints,
    )
    bond = fused.prepare_arm(
        bond_config, scanner="bond", n_threads=config.n_threads, bound="tight"
    )
    bond_id = "bond-exact-safe"
    arms.append(
        ArmSpec(
            arm_id=bond_id,
            method_family="fused-panel-bond",
            operation=_capture(
                bond_id,
                lambda: _native_pass(bond, query_ids, num_documents),
                latest,
            ),
            parameters={
                "bound": "tight",
                "checkpoints": list(config.checkpoints),
                "dimension_order": "natural",
                "k": config.k_retrieve,
                "shrink": 1.0,
                "threshold_policy": "self_bound",
                "thread_tag": config.thread_tag,
            },
            exactness="exact_safe",
            comparison_scope="uncapped_reference",
            validator=exact_validator,
            metadata_extractor=_metadata,
            timer_scope_id="retrieval-workload-pass",
        )
    )

    partitioned = PartitionedFusedScan(
        flat, starts, n_partitions=2 if config.fixture else None
    )
    partition_stats = partitioned.build()
    for nprobe in config.partition_nprobes:
        arm_id = f"partitioned-nprobe-{nprobe:03d}"
        def partition_operation(selected=nprobe) -> RetrievalPassResult:
            rows = []
            for query_id, query in zip(query_ids, queries, strict=True):
                ids, scores, stats = partitioned.search(
                    fused.lib,
                    query,
                    k=config.k_retrieve,
                    nprobe=selected,
                    checkpoints=config.checkpoints,
                    bound="tight",
                    n_threads=config.n_threads,
                    scanner="bond",
                )
                rows.append(
                    RankedQueryResult(
                        query_id, ids, scores, stats["candidate_work"], accounting=stats
                    )
                )
            return RetrievalPassResult(tuple(rows))
        arms.append(
            ArmSpec(
                arm_id=arm_id,
                method_family="partitioned-fused-bond",
                operation=_capture(arm_id, partition_operation, latest),
                parameters={
                    "bound": "tight",
                    "checkpoints": list(config.checkpoints),
                    "k": config.k_retrieve,
                    "nprobe": nprobe,
                    "thread_tag": config.thread_tag,
                },
                exactness="approximate",
                comparison_scope="probe_frontier",
                validator=lambda result: validate_ranked_pass(
                    result, query_ids=query_ids, num_documents=num_documents
                ),
                metadata_extractor=_metadata,
                timer_scope_id="retrieval-workload-pass",
            )
        )

    index_hashes: list[str] = [_partition_index_sha256(partitioned)]
    budgets = sorted({min(cap, num_documents) for cap in config.candidate_caps})
    if budgets:
        faiss = FaissIVFBaseline(flat, starts, nprobe=config.faiss_nprobe)
        faiss_cache = REPO_ROOT / "data" / "faiss_indexes" / f"{config.dataset}.faiss"
        faiss.build(cache_path=str(faiss_cache))
        faiss.index.nprobe = config.faiss_nprobe
        faiss_hash = _path_sha256(faiss_cache)
        if faiss_hash:
            index_hashes.append(faiss_hash)
        for cap in budgets:
            arm_id = f"faiss-cap-{cap:05d}"
            def faiss_operation(selected=cap) -> RetrievalPassResult:
                rows = []
                for query_id, query in zip(query_ids, queries, strict=True):
                    ids, scores, work = faiss.topk_with_work(
                        query, k=config.k_retrieve, candidate_budget=selected
                    )
                    rows.append(RankedQueryResult(query_id, ids, scores, work.to_dict()))
                return RetrievalPassResult(tuple(rows))
            arms.append(
                ArmSpec(
                    arm_id=arm_id,
                    method_family="faiss-ivf-exact-rerank",
                    operation=_capture(arm_id, faiss_operation, latest),
                    parameters={
                        "candidate_cap": cap,
                        "k": config.k_retrieve,
                        "metric": "inner_product",
                        "n_lists": faiss.n_lists,
                        "nprobe": config.faiss_nprobe,
                        "thread_tag": config.thread_tag,
                    },
                    exactness="approximate",
                    comparison_scope="equal_work_reranking",
                    validator=lambda result: validate_ranked_pass(
                        result, query_ids=query_ids, num_documents=num_documents
                    ),
                    metadata_extractor=_metadata,
                    timer_scope_id="retrieval-workload-pass",
                )
            )

        plaid_arms: dict[int, PLAIDBaseline] = {}
        for cap in budgets:
            plaid = PLAIDBaseline(
                config.dataset,
                n_ivf_probe=config.plaid_n_ivf_probe,
                n_full_scores=cap,
                num_threads=config.n_threads if config.n_threads > 0 else None,
            )
            if plaid.exists():
                plaid.load()
            else:
                plaid.build(unpack_embeddings(flat, starts))
            plaid_arms[cap] = plaid
            arm_id = f"plaid-cap-{cap:05d}"
            def plaid_operation(selected=cap) -> RetrievalPassResult:
                selected_plaid = plaid_arms[selected]
                rankings, work = selected_plaid.search_with_work(
                    queries, k=config.k_retrieve, comparison_scope="system_cap"
                )
                return RetrievalPassResult(
                    tuple(
                        RankedQueryResult(query_id, ids, scores, observed.to_dict())
                        for query_id, (ids, scores), observed in zip(
                            query_ids, rankings, work, strict=True
                        )
                    )
                )
            arms.append(
                ArmSpec(
                    arm_id=arm_id,
                    method_family="plaid-fastplaid",
                    operation=_capture(arm_id, plaid_operation, latest),
                    parameters={
                        "configured_full_score_cap": cap,
                        "k": config.k_retrieve,
                        "n_ivf_probe": config.plaid_n_ivf_probe,
                        "thread_tag": config.thread_tag,
                        "work_observability": "actual fully scored unavailable",
                    },
                    exactness="approximate",
                    comparison_scope="system_cap",
                    validator=lambda result: validate_ranked_pass(
                        result, query_ids=query_ids, num_documents=num_documents
                    ),
                    metadata_extractor=_metadata,
                    timer_scope_id="retrieval-workload-pass",
                )
            )
        plaid_path = next(iter(plaid_arms.values())).index_root / config.dataset
        plaid_hash = _path_sha256(plaid_path)
        if plaid_hash:
            index_hashes.append(plaid_hash)

    experiment_id = f"stage5-e01-ir-evaluation-{config.thread_tag}"
    index_sha = hashlib.sha256("\n".join(sorted(index_hashes)).encode()).hexdigest() if index_hashes else None
    spec = ExperimentSpec(
        experiment_id=experiment_id,
        artifact_id=f"{experiment_id}-{config.dataset}",
        dataset_id=config.dataset,
        workload_id=workload["workload_id"],
        arms=tuple(arms),
        command=(
            "uv run python -m experiments.stage5_corect.e01_ir_evaluation "
            f"--dataset {config.dataset} --threads {config.n_threads}"
            + (" --fixture" if config.fixture else "")
        ),
        baseline_arm_id=dense_id,
        metadata={
            "evaluation": {
                "k_eval": config.k_eval,
                "k_retrieve": config.k_retrieve,
                "standard_metrics": ["ndcg@10", "recall@100", "mrr@10"],
                "corect_role": "standard metric cross-validation only",
                "evidence_status": "diagnostic_only" if config.fixture else "eligible_after_final_rerun",
            },
            "partition_index": partition_stats,
        },
        workload_metadata=workload,
        provenance_inputs={
            "configuration_sha256": _configuration_sha256(config),
            "data_sha256": data_sha,
            "index_sha256": index_sha,
            "input_artifact_ids": [],
        },
    )
    return _Prepared(
        spec, latest, qrels, corpus_ids, exact_ids, exact_scores,
        exact_full_scores, query_ids, num_documents,
    )


@contextlib.contextmanager
def _thread_context(n_threads: int):
    """Apply the requested FAISS/BLAS thread mode outside measured scopes."""
    import faiss
    from threadpoolctl import threadpool_limits

    if n_threads > 0:
        previous = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(n_threads)
        try:
            with threadpool_limits(limits=n_threads, user_api="blas"):
                yield
        finally:
            faiss.omp_set_num_threads(previous)
    else:
        yield


def run_ir_evaluation(
    config: Stage5IREvaluationConfig,
    *,
    output_dir: Path,
    session_id: str | None = None,
    environment: dict[str, Any] | None = None,
) -> Stage5IRRun:
    """Prepare, time, validate, evaluate, and atomically serialize e01."""
    selected_session = session_id or new_session_id("stage5-e01")
    holder: dict[str, _Prepared] = {}

    def prepare() -> ExperimentSpec:
        prepared = prepare_ir_evaluation(config)
        holder["prepared"] = prepared
        return prepared.spec

    with _thread_context(config.n_threads):
        envelope, session, spec = execute_prepared_timing_experiment(
            prepare,
            TimingProtocol.preset("fixture" if config.fixture else "full_system"),
            session_id=selected_session,
            build_profile="release-native",
            environment=environment,
        )
    if not session.complete:
        raise RuntimeError("Stage 5 e01 timing session did not complete")
    prepared = holder["prepared"]
    quality_rows = []
    for arm in spec.arms:
        result = prepared.latest[arm.arm_id]
        quality_rows.append(
            {
                "arm_id": arm.arm_id,
                "method_family": arm.method_family,
                "comparison_scope": arm.comparison_scope,
                **evaluate_pass(
                    result,
                    corpus_ids=prepared.corpus_ids,
                    qrels=prepared.qrels,
                    exact_ids=prepared.exact_ids,
                    k_eval=config.k_eval,
                    crosscheck_corect=arm.arm_id == spec.baseline_arm_id,
                ),
                "candidate_work": _work_observations(result),
            }
        )
    validated_quality = IRQualityPayload(tuple(quality_rows)).to_dict()
    enriched = replace(
        envelope,
        payload={
            **dict(envelope.payload),
            "quality": {
                **validated_quality,
                "corect_crosscheck_arm_id": spec.baseline_arm_id,
                "corect_scope": "standard metrics only",
            },
        },
    ).validate()
    output_path = output_dir / deterministic_result_name(
        spec.experiment_id, spec.dataset_id, spec.workload_id
    )
    saved = write_or_append_timing_sessions(output_path, enriched)
    return Stage5IRRun(saved, output_path)
