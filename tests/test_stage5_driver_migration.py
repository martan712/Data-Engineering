from __future__ import annotations

from pathlib import Path

import pytest

from bondmaxsim.experiments.stage5.ir_evaluation import (
    Stage5IREvaluationConfig,
    run_ir_evaluation,
)
from bondmaxsim.results import load_result


def _environment():
    return {
        "session": {"session_id": "fixture-stage5-session"},
        "threads": {"OMP_NUM_THREADS": "1"},
        "code": {
            "git_revision": "fixture-revision",
            "git_dirty": False,
            "uv_lock_sha256": "fixture-lock",
        },
        "toolchain": {"native_binary_sha256": {"fused": "fixture-native"}},
    }


@pytest.fixture(scope="module")
def e01_artifact(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("stage5-e01")
    run = run_ir_evaluation(
        Stage5IREvaluationConfig.fixture_config(),
        output_dir=output,
        session_id="fixture-stage5-e01",
        environment=_environment(),
    )
    return run.output_path


def test_e01_fixture_is_counterbalanced_validated_and_artifact_ready(e01_artifact):
    result = load_result(e01_artifact)
    assert result.schema_version == "1.0.0"
    assert result.workload_id == "synthetic-small-qrels-test-v1"
    assert result.protocol["protocol_class"] == "fixture"
    assert result.protocol["workload"]["token_counts"] == [1, 2, 3, 2]
    assert result.protocol["workload"]["token_count_mean"] == 2.0
    assert result.provenance["command"].endswith("--fixture")

    session = result.payload["sessions"][0]
    assert session["complete"]
    assert len(session["observations"]) == 5 * (1 + 3)
    assert {row["arm_id"] for row in session["validation_results"]} == {
        "dense-fused", "openblas-exact", "bond-exact-safe"
    }
    positions = {
        row["arm_id"]: row["arm_position"]
        for row in session["observations"]
        if row["phase"] == "measured" and row["round_index"] == 0
    }
    assert sorted(positions.values()) == [0, 1, 2, 3, 4]

    arms = {arm["arm_id"]: arm for arm in result.method_configuration["arms"]}
    assert arms["partitioned-nprobe-001"]["comparison_scope"] == "probe_frontier"
    assert arms["bond-exact-safe"]["parameters"]["checkpoints"] == [8]
    quality = {row["arm_id"]: row for row in result.payload["quality"]["rows"]}
    assert result.payload["quality"]["corect_scope"] == "standard metrics only"
    assert quality["dense-fused"]["recall_vs_oracle_set"] == 1.0
    assert set(quality["dense-fused"]["per_query_ndcg_at_10"]) == {"q0", "q1", "q2", "q3"}
    work = quality["partitioned-nprobe-001"]["candidate_work"]
    assert all(row["documents_probed"]["quality"] == "exact" for row in work)
    assert all(row["partitions_probed"]["value"] == 1 for row in work)


def test_e01_fixture_reopens_as_the_same_envelope(e01_artifact):
    first = load_result(e01_artifact)
    second = load_result(e01_artifact)
    assert first == second
