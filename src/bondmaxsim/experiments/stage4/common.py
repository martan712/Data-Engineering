"""Small shared pieces for migrated Stage 4 experiment drivers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_ids
from bondmaxsim.data.fixture import fixture_arrays
from bondmaxsim.data.loader import load_dataset
from bondmaxsim.experiments.candidate_work import CandidateWorkObservation
from bondmaxsim.experiments.workloads import WorkloadMetadata
from bondmaxsim.oracle.agreement import AgreementBatchResult, batch_agreement_result, validate_boundary_tie_equivalence
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores


@dataclass(frozen=True)
class Stage4Data:
    flat_tokens: np.ndarray
    doc_starts: np.ndarray
    queries: tuple[np.ndarray, ...]
    query_ids: tuple[str, ...]
    workload_metadata: dict[str, Any]
    exact_full_scores: tuple[np.ndarray, ...]
    exact_ids: tuple[np.ndarray, ...]
    exact_scores: tuple[np.ndarray, ...]
    data_sha256: str | None

    @property
    def num_documents(self) -> int:
        return len(self.doc_starts)


@dataclass(frozen=True)
class RetrievalQueryResult:
    ids: np.ndarray
    scores: np.ndarray
    candidate_work: CandidateWorkObservation | None = None
    accounting: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievalPassResult:
    queries: tuple[RetrievalQueryResult, ...]


def configuration_sha256(config: Any) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage4_data(*, dataset: str, query_count: int, query_seed: int, k: int, fixture: bool) -> Stage4Data:
    if fixture:
        arrays = fixture_arrays()
        values, starts = arrays["query_values"], arrays["query_starts"]
        queries = tuple(
            np.ascontiguousarray(
                values[int(start):int(starts[index + 1]) if index + 1 < len(starts) else len(values)],
                dtype=np.float32,
            )
            for index, start in enumerate(starts)
        )
        query_ids = tuple(f"q{index}" for index in range(len(queries)))
        workload = {
            "workload_id": "synthetic-small-v1",
            "dataset": "synthetic-small-v1",
            "regime": "fixture",
            "sample_size": len(queries),
            "token_counts": [len(query) for query in queries],
        }
        flat_tokens = arrays["doc_values"]
        doc_starts = arrays["doc_starts"]
        data_sha = None
    else:
        flat_tokens, doc_starts, population = load_dataset(dataset)
        if query_count > len(population):
            raise ValueError("query sample exceeds the available population")
        rng = np.random.default_rng(query_seed)
        selected = np.sort(rng.choice(len(population), size=query_count, replace=False))
        queries = tuple(np.ascontiguousarray(population[int(index)], dtype=np.float32) for index in selected)
        all_ids = load_ids(dataset)["query_ids"][:len(population)]
        query_ids = tuple(all_ids[int(index)] for index in selected)
        workload = WorkloadMetadata.from_queries(
            dataset=dataset,
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
        path = REPO_ROOT / "data" / "embeddings" / f"{dataset}.npz"
        data_sha = _file_sha256(path)
    exact_full = tuple(exact_maxsim_scores(query, flat_tokens, doc_starts) for query in queries)
    pairs = tuple(topk_from_scores(scores, k) for scores in exact_full)
    return Stage4Data(
        np.ascontiguousarray(flat_tokens, dtype=np.float32),
        np.asarray(doc_starts, dtype=np.int64),
        queries,
        query_ids,
        workload,
        exact_full,
        tuple(ids for ids, _ in pairs),
        tuple(scores for _, scores in pairs),
        data_sha,
    )


def validate_retrieval(result: RetrievalPassResult, data: Stage4Data, k: int) -> AgreementBatchResult:
    rows = []
    for returned, exact_ids, exact_scores, full_scores in zip(
        result.queries, data.exact_ids, data.exact_scores, data.exact_full_scores
    ):
        rows.append(
            validate_boundary_tie_equivalence(
                returned.ids,
                exact_ids,
                exact_scores,
                k=k,
                num_documents=data.num_documents,
                exact_scores_by_id=full_scores,
                returned_scores=returned.scores,
                ranked_tie_break="none",
            )
        )
    return batch_agreement_result(rows)


def retrieval_metadata(result: RetrievalPassResult, query_ids: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "retrieval_quality": [
            {"query_id": query_id, "returned_ids": row.ids.astype(int).tolist()}
            for query_id, row in zip(query_ids, result.queries)
        ]
    }
    if all(row.candidate_work is not None for row in result.queries):
        metadata["candidate_work"] = [
            row.candidate_work.to_dict() for row in result.queries if row.candidate_work is not None
        ]
    if any(row.accounting is not None for row in result.queries):
        metadata["probe_accounting"] = [
            {"query_id": query_id, **(row.accounting or {})}
            for query_id, row in zip(query_ids, result.queries)
        ]
    return metadata


def assert_historical_controls(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    historical = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": value, "historical": historical.get(key)}
        for key, value in expected.items()
        if historical.get(key) != value
    }
    if mismatches:
        raise ValueError(f"historical Stage 4 methodology mismatch: {mismatches}")
    return {"status": "pass", "checked_fields": sorted(expected)}
