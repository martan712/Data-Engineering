from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bondmaxsim.experiments.workloads import (
    WorkloadMetadata,
    embedding_configuration_sha256,
    mechanism_workload_id,
    ordered_query_id_sha256,
    qrels_test_workload_id,
    token_counts_from_offsets,
)


def test_stable_workload_ids():
    assert mechanism_workload_id("scifact") == (
        "beir-scifact-mechanism-seed42-n50-v1"
    )
    assert qrels_test_workload_id("nfcorpus") == "beir-nfcorpus-qrels-test-v1"


def test_token_counts_are_derived_from_validated_offsets():
    assert token_counts_from_offsets(np.array([0, 3, 5]), 9) == (3, 2, 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        token_counts_from_offsets(np.array([0, 3, 3]), 9)
    with pytest.raises(TypeError, match="integer dtype"):
        token_counts_from_offsets(np.array([0.0, 3.0]), 5)


def test_ordered_query_hash_changes_with_order_and_rejects_duplicates():
    assert ordered_query_id_sha256(["q1", "q2"]) != ordered_query_id_sha256(
        ["q2", "q1"]
    )
    with pytest.raises(ValueError, match="unique"):
        ordered_query_id_sha256(["q1", "q1"])


def test_embedding_hash_covers_values_and_configuration():
    values = np.eye(3, dtype=np.float32)
    first = embedding_configuration_sha256(values, {"model": "fixture", "max": 32})
    assert first == embedding_configuration_sha256(
        values.copy(), {"max": 32, "model": "fixture"}
    )
    assert first != embedding_configuration_sha256(
        values, {"model": "fixture", "max": 48}
    )


def test_qrels_workload_metadata_contains_exact_length_statistics():
    queries = [
        np.ones((3, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((4, 2), dtype=np.float32),
    ]
    metadata = WorkloadMetadata.from_queries(
        dataset="scifact",
        regime="qrels_test",
        query_ids=["q1", "q2", "q3"],
        queries=queries,
        encoding_configuration={"model": "fixture", "max_length": 32},
        selection="all qrels-evaluable test queries in source order",
        source_revision="fixture-revision",
        source_split="test",
    )
    assert metadata.workload_id == "beir-scifact-qrels-test-v1"
    assert metadata.token_counts == (3, 2, 4)
    assert metadata.token_count_mean == 3.0
    assert metadata.token_count_median == 3.0
    assert metadata.token_count_min == 2
    assert metadata.token_count_max == 4
    assert metadata.to_dict()["token_counts"] == [3, 2, 4]
    with pytest.raises(ValueError, match="summary"):
        replace(metadata, token_count_mean=99.0)
    with pytest.raises(ValueError, match="checksum"):
        replace(metadata, token_counts_sha256="0" * 64)


def test_mechanism_workload_id_cannot_describe_wrong_sample_size():
    with pytest.raises(ValueError, match="exactly 50"):
        WorkloadMetadata.from_queries(
            dataset="scifact",
            regime="mechanism",
            query_ids=["q1"],
            queries=[np.ones((2, 3), dtype=np.float32)],
            encoding_configuration={"model": "fixture"},
            selection="seed-42 sample",
            source_revision="fixture",
            source_split="train",
        )
