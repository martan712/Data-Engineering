"""Standalone Stage 5 BM25 reference with explicit timing non-comparability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bondmaxsim.baselines.bm25 import BM25Baseline
from bondmaxsim.data.beir_ids import load_qrels_tsv, qrels_path
from bondmaxsim.data.loader import load_eval_queries
from bondmaxsim.eval.corect import compute_corect_standard_metrics
from bondmaxsim.eval.qrels import compute_quality_metrics, per_query_ndcg_at_10
from bondmaxsim.experiments.arms import ArmSpec, ExperimentSpec
from bondmaxsim.experiments.execution import execute_prepared_timing_experiment
from bondmaxsim.experiments.stage5.quality import RankedQueryResult, RetrievalPassResult, rank_run, validate_ranked_pass
from bondmaxsim.experiments.stage5.persistence import write_or_append_timing_sessions
from bondmaxsim.experiments.timing import TimingProtocol, new_session_id
from bondmaxsim.experiments.workloads import WorkloadMetadata
from bondmaxsim.results.io import deterministic_result_name
from bondmaxsim.results.models import ExperimentResultEnvelope, IRQualityPayload


@dataclass(frozen=True)
class Stage5BM25Config:
    dataset: str = "scifact"
    k_retrieve: int = 100
    k1: float = 1.5
    b: float = 0.75
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "Stage5BM25Config":
        return cls(dataset="synthetic-small-v1", k_retrieve=6, fixture=True)


@dataclass(frozen=True)
class Stage5BM25Run:
    envelope: ExperimentResultEnvelope
    output_path: Path


def _fixture_inputs():
    corpus_ids = [f"d{index}" for index in range(6)]
    texts = [
        "alpha evidence", "beta unrelated", "gamma finding",
        "delta result", "epsilon study", "zeta conclusion",
    ]
    query_ids = [f"q{index}" for index in range(4)]
    query_texts = ["alpha", "gamma", "epsilon", "zeta"]
    qrels = {"q0": {"d0": 2}, "q1": {"d2": 1}, "q2": {"d4": 1}, "q3": {"d5": 2}}
    workload = {
        "workload_id": "synthetic-small-qrels-test-v1",
        "dataset": "synthetic-small-v1",
        "regime": "fixture",
        "ordered_query_id_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
        "sample_size": len(query_ids),
        "token_counts": [1, 1, 1, 1],
        "token_count_mean": 1.0,
        "token_count_median": 1.0,
        "token_count_min": 1,
        "token_count_max": 1,
        "selection": "all deterministic fixture qrels queries in source order",
        "source_revision": "bondmaxsim.fixture-manifest-1.0.0",
        "source_split": "qrels-test",
    }
    data_sha = hashlib.sha256(
        json.dumps(
            [corpus_ids, texts, query_ids, query_texts, qrels],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return corpus_ids, texts, query_ids, query_texts, qrels, workload, data_sha


def _full_inputs(config: Stage5BM25Config):
    from datasets import load_dataset as hf_load_dataset

    qrels = load_qrels_tsv(qrels_path(config.dataset))
    encoded_queries, query_ids = load_eval_queries(config.dataset)
    corpus = hf_load_dataset(f"BeIR/{config.dataset}", "corpus", split="corpus")
    corpus_ids = [str(value) for value in corpus["_id"]]
    texts = list(corpus["text"])
    query_data = hf_load_dataset(f"BeIR/{config.dataset}", "queries", split="queries")
    text_by_id = {str(query_id): text for query_id, text in zip(query_data["_id"], query_data["text"], strict=True)}
    missing = [query_id for query_id in query_ids if query_id not in text_by_id]
    if missing:
        raise ValueError(f"{config.dataset}: missing query texts for {missing[:3]}")
    workload = WorkloadMetadata.from_queries(
        dataset=config.dataset,
        regime="qrels_test",
        query_ids=query_ids,
        queries=encoded_queries,
        encoding_configuration={
            "model": "lightonai/GTE-ModernColBERT-v1",
            "measurement": "encoded query token lengths; BM25 uses raw text",
        },
        selection="all encoded queries with test qrels, source order",
        source_revision="archive embeddings plus pinned BEIR raw query split",
        source_split="qrels-test",
    ).to_dict()
    query_texts = [text_by_id[q] for q in query_ids]
    data_sha = hashlib.sha256(
        json.dumps(
            [corpus_ids, texts, query_ids, query_texts, qrels],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return corpus_ids, texts, query_ids, query_texts, qrels, workload, data_sha


def run_bm25(
    config: Stage5BM25Config,
    *,
    output_dir: Path,
    session_id: str | None = None,
    environment: dict[str, Any] | None = None,
) -> Stage5BM25Run:
    selected_session = session_id or new_session_id("stage5-e03")
    state: dict[str, Any] = {}

    def prepare() -> ExperimentSpec:
        loaded = _fixture_inputs() if config.fixture else _full_inputs(config)
        corpus_ids, texts, query_ids, query_texts, qrels, workload, data_sha = loaded
        baseline = BM25Baseline(k1=config.k1, b=config.b)
        build_stats = baseline.build(corpus_ids, texts)
        id_to_index = {document_id: index for index, document_id in enumerate(corpus_ids)}
        latest: dict[str, RetrievalPassResult] = {}

        def operation() -> RetrievalPassResult:
            rankings = baseline.search(query_texts, k=min(config.k_retrieve, len(corpus_ids)))
            result = RetrievalPassResult(
                tuple(
                    RankedQueryResult(
                        query_id,
                        np.asarray([id_to_index[value] for value in document_ids], dtype=np.int64),
                        np.asarray(scores, dtype=np.float32),
                    )
                    for query_id, (document_ids, scores) in zip(query_ids, rankings, strict=True)
                )
            )
            latest["bm25"] = result
            return result

        state.update(
            latest=latest, corpus_ids=corpus_ids, qrels=qrels,
            query_ids=query_ids, workload=workload,
        )
        experiment_id = "stage5-e03-bm25-standalone"
        return ExperimentSpec(
            experiment_id=experiment_id,
            artifact_id=f"{experiment_id}-{config.dataset}",
            dataset_id=config.dataset,
            workload_id=workload["workload_id"],
            arms=(
                ArmSpec(
                    arm_id="bm25",
                    method_family="bm25-lucene",
                    operation=operation,
                    parameters={
                        "b": config.b,
                        "document_text": "text-only",
                        "k": config.k_retrieve,
                        "k1": config.k1,
                        "query_tokenization_in_timed_scope": True,
                        "scoring": "lucene",
                        "stemmer": "snowball-english",
                        "stopwords": "en",
                    },
                    exactness="reference",
                    comparison_scope="uncapped_reference",
                    validator=lambda result: validate_ranked_pass(
                        result,
                        query_ids=query_ids,
                        num_documents=len(corpus_ids),
                    ),
                    timer_scope_id="standalone-lexical-retrieval-pass",
                ),
            ),
            command=(
                "uv run python -m experiments.stage5_corect.e03_bm25_baseline "
                f"--dataset {config.dataset}"
                + (" --fixture" if config.fixture else "")
            ),
            metadata={
                "build": build_stats,
                "timing_status": {
                    "mode": "standalone",
                    "comparable_to_vector_arm_margins": False,
                    "reason": "BM25 runs on an independent lexical stack and is not counterpaired with e01 arms",
                },
                "evidence_status": "diagnostic_only" if config.fixture else "eligible_after_final_rerun",
            },
            workload_metadata=workload,
            provenance_inputs={
                "configuration_sha256": hashlib.sha256(
                    json.dumps(asdict(config), sort_keys=True).encode()
                ).hexdigest(),
                "data_sha256": data_sha,
                "index_sha256": hashlib.sha256(
                    f"bm25s-lucene:{config.k1}:{config.b}:{data_sha}".encode()
                ).hexdigest(),
                "input_artifact_ids": [],
            },
        )

    envelope, session, spec = execute_prepared_timing_experiment(
        prepare,
        TimingProtocol.preset("fixture" if config.fixture else "full_system"),
        session_id=selected_session,
        build_profile="python-lexical",
        environment=environment,
    )
    if not session.complete:
        raise RuntimeError("Stage 5 e03 standalone timing session did not complete")
    result = state["latest"]["bm25"]
    run = rank_run(result, state["corpus_ids"])
    metrics = compute_quality_metrics(run, state["qrels"])
    quality = {
        "arm_id": "bm25",
        "method_family": "bm25-lucene",
        "comparison_scope": "uncapped_reference",
        "ndcg_at_10": metrics["nDCG_at_10"],
        "recall_at_100": metrics["recall_at_100"],
        "mrr_at_10": metrics["MRR_at_10"],
        "recall_vs_oracle_set": None,
        "corect_standard_metrics": compute_corect_standard_metrics(run, state["qrels"]),
        "per_query_ndcg_at_10": per_query_ndcg_at_10(run, state["qrels"]),
    }
    validated_quality = IRQualityPayload((quality,)).to_dict()
    enriched = replace(
        envelope,
        payload={
            **dict(envelope.payload),
            "quality": {
                **validated_quality,
                "corect_scope": "standard metrics only",
            },
        },
    ).validate()
    output_path = output_dir / deterministic_result_name(
        spec.experiment_id, spec.dataset_id, spec.workload_id
    )
    saved = write_or_append_timing_sessions(output_path, enriched)
    return Stage5BM25Run(saved, output_path)
