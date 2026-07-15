"""Stable query-workload identities and offset-derived metadata."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np

_DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
WORKLOAD_REGIMES = {"mechanism", "qrels_test"}


def mechanism_workload_id(dataset: str) -> str:
    _validate_dataset(dataset)
    return f"beir-{dataset}-mechanism-seed42-n50-v1"


def qrels_test_workload_id(dataset: str) -> str:
    _validate_dataset(dataset)
    return f"beir-{dataset}-qrels-test-v1"


def _validate_dataset(dataset: str) -> None:
    if not isinstance(dataset, str) or not _DATASET_ID.fullmatch(dataset):
        raise ValueError("dataset must be a stable lowercase ID")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_query_id_sha256(query_ids: Sequence[str]) -> str:
    ids = list(query_ids)
    if not ids or any(not isinstance(query_id, str) or not query_id for query_id in ids):
        raise ValueError("query IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("query IDs must be unique")
    return _canonical_sha256(ids)


def token_counts_from_offsets(
    query_offsets: np.ndarray | Sequence[int], total_query_tokens: int
) -> tuple[int, ...]:
    offsets = np.asarray(query_offsets)
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("query offsets must be a non-empty vector")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise TypeError("query offsets must have an integer dtype")
    if isinstance(total_query_tokens, bool) or not isinstance(total_query_tokens, int):
        raise TypeError("total_query_tokens must be an integer")
    if total_query_tokens <= 0 or int(offsets[0]) != 0:
        raise ValueError("query offsets must start at zero and cover non-empty tokens")
    extended = np.append(offsets.astype(np.int64, copy=False), total_query_tokens)
    counts = np.diff(extended)
    if np.any(counts <= 0):
        raise ValueError("query offsets must be strictly increasing and in range")
    return tuple(int(value) for value in counts)


def embedding_configuration_sha256(
    query_values: np.ndarray,
    encoding_configuration: Mapping[str, Any],
) -> str:
    values = np.ascontiguousarray(query_values)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("query embeddings must be a non-empty rank-2 array")
    digest = hashlib.sha256()
    header = {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "encoding_configuration": dict(encoding_configuration),
    }
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkloadMetadata:
    workload_id: str
    dataset: str
    regime: str
    ordered_query_id_sha256: str
    embedding_configuration_sha256: str
    sample_size: int
    token_count_mean: float
    token_count_median: float
    token_count_min: int
    token_count_max: int
    token_counts: tuple[int, ...]
    token_counts_sha256: str
    selection: str
    source_revision: str
    source_split: str

    def __post_init__(self) -> None:
        _validate_dataset(self.dataset)
        if self.regime not in WORKLOAD_REGIMES:
            raise ValueError(f"invalid workload regime {self.regime!r}")
        expected_id = (
            mechanism_workload_id(self.dataset)
            if self.regime == "mechanism"
            else qrels_test_workload_id(self.dataset)
        )
        if self.workload_id != expected_id:
            raise ValueError(f"workload_id must be {expected_id!r}")
        if self.regime == "mechanism" and self.sample_size != 50:
            raise ValueError("mechanism workload IDs require exactly 50 queries")
        if self.sample_size != len(self.token_counts) or self.sample_size <= 0:
            raise ValueError("sample_size must match the token-count vector")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in self.token_counts
        ):
            raise ValueError("token counts must be positive integers")
        expected_statistics = (
            float(mean(self.token_counts)),
            float(median(self.token_counts)),
            min(self.token_counts),
            max(self.token_counts),
        )
        if expected_statistics != (
            self.token_count_mean,
            self.token_count_median,
            self.token_count_min,
            self.token_count_max,
        ):
            raise ValueError("token-count summary does not match token-count vector")
        for digest in (
            self.ordered_query_id_sha256,
            self.embedding_configuration_sha256,
            self.token_counts_sha256,
        ):
            if not _SHA256.fullmatch(digest):
                raise ValueError("workload hashes must be lowercase SHA-256 values")
        if self.token_counts_sha256 != _canonical_sha256(list(self.token_counts)):
            raise ValueError("token-count checksum does not match token-count vector")
        if not self.selection or not self.source_revision or not self.source_split:
            raise ValueError("selection, source_revision, and source_split are required")

    @classmethod
    def from_offsets(
        cls,
        *,
        dataset: str,
        regime: str,
        query_ids: Sequence[str],
        query_offsets: np.ndarray | Sequence[int],
        total_query_tokens: int,
        embedding_configuration_sha256: str,
        selection: str,
        source_revision: str,
        source_split: str,
    ) -> "WorkloadMetadata":
        ids = list(query_ids)
        counts = token_counts_from_offsets(query_offsets, total_query_tokens)
        if len(ids) != len(counts):
            raise ValueError("query IDs and query offsets must have the same length")
        if regime not in WORKLOAD_REGIMES:
            raise ValueError(f"invalid workload regime {regime!r}")
        workload_id = (
            mechanism_workload_id(dataset)
            if regime == "mechanism"
            else qrels_test_workload_id(dataset)
        )
        return cls(
            workload_id=workload_id,
            dataset=dataset,
            regime=regime,
            ordered_query_id_sha256=ordered_query_id_sha256(ids),
            embedding_configuration_sha256=embedding_configuration_sha256,
            sample_size=len(ids),
            token_count_mean=float(mean(counts)),
            token_count_median=float(median(counts)),
            token_count_min=min(counts),
            token_count_max=max(counts),
            token_counts=counts,
            token_counts_sha256=_canonical_sha256(list(counts)),
            selection=selection,
            source_revision=source_revision,
            source_split=source_split,
        )

    @classmethod
    def from_queries(
        cls,
        *,
        dataset: str,
        regime: str,
        query_ids: Sequence[str],
        queries: Sequence[np.ndarray],
        encoding_configuration: Mapping[str, Any],
        selection: str,
        source_revision: str,
        source_split: str,
    ) -> "WorkloadMetadata":
        if not queries:
            raise ValueError("queries must be non-empty")
        arrays = [np.asarray(query) for query in queries]
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("queries must be rank-2 arrays with one shared dimension")
        dimensions = {array.shape[1] for array in arrays}
        if len(dimensions) != 1 or next(iter(dimensions)) <= 0:
            raise ValueError("queries must be rank-2 arrays with one shared dimension")
        counts = [len(query) for query in arrays]
        offsets = np.zeros(len(queries), dtype=np.int64)
        np.cumsum(counts[:-1], out=offsets[1:])
        values = np.concatenate(
            arrays, axis=0
        )
        return cls.from_offsets(
            dataset=dataset,
            regime=regime,
            query_ids=query_ids,
            query_offsets=offsets,
            total_query_tokens=len(values),
            embedding_configuration_sha256=embedding_configuration_sha256(
                values, encoding_configuration
            ),
            selection=selection,
            source_revision=source_revision,
            source_split=source_split,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "token_counts": list(self.token_counts),
        }
