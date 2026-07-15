from __future__ import annotations

from pathlib import Path

import pytest

from bondmaxsim.experiments.stage4.fixed_candidates import FixedCandidateConfig, run_fixed_candidate_arms
from bondmaxsim.experiments.stage4.partition_frontier import PartitionFrontierConfig, run_partition_frontier
from bondmaxsim.experiments.stage4.seeded_tau import SeededTauConfig, run_seeded_tau_recovery
from bondmaxsim.experiments.stage4.common import assert_historical_controls
from bondmaxsim.config import REPO_ROOT
from bondmaxsim.render.stage4.fixed_candidates import render_fixed_candidates
from bondmaxsim.render.stage4.partition_frontier import render_partition_frontier
from bondmaxsim.render.stage4.seeded_tau import render_seeded_tau
from bondmaxsim.results import load_result


def _environment(session_id: str) -> dict:
    return {
        "session": {"session_id": session_id},
        "threads": {"OMP_NUM_THREADS": "1"},
        "code": {
            "git_revision": "fixture-revision",
            "git_dirty": False,
            "uv_lock_sha256": "fixture-lock",
        },
        "toolchain": {"native_binary_sha256": {"fused": "fixture-native"}},
    }


@pytest.fixture(scope="module")
def migrated_outputs(tmp_path_factory):
    output = tmp_path_factory.mktemp("stage4-migrated")
    e01 = run_fixed_candidate_arms(
        FixedCandidateConfig.fixture_config(),
        output_dir=output,
        session_id="fixture-e01",
        environment=_environment("fixture-e01"),
    )
    e02 = run_seeded_tau_recovery(
        SeededTauConfig.fixture_config(),
        output_dir=output,
        session_id="fixture-e02",
        environment=_environment("fixture-e02"),
    )
    e03 = run_partition_frontier(
        PartitionFrontierConfig.fixture_config(),
        output_dir=output,
        session_id="fixture-e03",
        environment=_environment("fixture-e03"),
    )
    return output, e01, e02, e03


def _assert_shared_timing_contract(envelope) -> None:
    session = envelope.payload["sessions"][0]
    assert session["complete"]
    assert " --fixture" in envelope.provenance["command"]
    setup = [event for event in session["scope_events"] if event["scope_id"] == "experiment-setup"]
    assert len(setup) == 1
    assert setup[0]["measured"] is False
    assert all(
        event["scope_id"] not in {"experiment-setup", "result-validation", "result-metadata"}
        for event in session["scope_events"]
        if event["measured"]
    )


def test_e01_separates_equal_work_from_system_cap_and_retains_actual_work(migrated_outputs):
    _, run, _, _ = migrated_outputs
    equal = load_result(run.equal_work.output_path)
    system = load_result(run.system_cap.output_path)
    assert equal == run.equal_work.envelope
    assert system == run.system_cap.envelope
    _assert_shared_timing_contract(equal)
    _assert_shared_timing_contract(system)
    assert equal.method_configuration["output_semantics"] == "equal_work_reranking"
    assert system.method_configuration["output_semantics"] == "system_cap"
    equal_arms = equal.method_configuration["arms"]
    system_arms = system.method_configuration["arms"]
    assert {row["comparison_scope"] for row in equal_arms if row["arm_id"] != "dense-fused"} == {"equal_work_reranking"}
    assert {row["comparison_scope"] for row in system_arms if row["arm_id"] != "dense-fused"} == {"system_cap"}

    equal_metadata = next(
        row for row in equal.payload["sessions"][0]["result_metadata"]
        if row["arm_id"].startswith("faiss-")
    )
    system_metadata = next(
        row for row in system.payload["sessions"][0]["result_metadata"]
        if row["arm_id"].startswith("plaid-")
    )
    assert all(row["documents_fully_scored"]["quality"] == "exact" for row in equal_metadata["candidate_work"])
    assert all(row["documents_fully_scored"]["quality"] == "unavailable" for row in system_metadata["candidate_work"])


def test_e02_records_components_and_direct_total_without_summing_minima(migrated_outputs):
    _, _, run, _ = migrated_outputs
    envelope = load_result(run.output_path)
    _assert_shared_timing_contract(envelope)
    assert "never summed" in envelope.method_configuration["cost_interpretation"]
    arms = {row["arm_id"]: row for row in envelope.method_configuration["arms"]}
    for strength in ("cheap", "strong"):
        assert arms[f"ivf-seed-{strength}-only"]["parameters"]["cost_component"] == "seed_only"
        assert arms[f"ivf-seed-{strength}-kernel"]["parameters"]["cost_component"] == "kernel_only"
        total = arms[f"ivf-seed-{strength}-end-to-end"]["parameters"]
        assert total["cost_component"] == "direct_end_to_end"
        assert total["derived_by_summing_components"] is False
    assert not any("combined_min" in key for row in arms.values() for key in row["parameters"])


def test_e03_retains_exact_per_query_document_and_partition_counts(migrated_outputs):
    _, _, _, run = migrated_outputs
    envelope = load_result(run.output_path)
    _assert_shared_timing_contract(envelope)
    session = envelope.payload["sessions"][0]
    metadata = next(
        row for row in session["result_metadata"]
        if row["arm_id"].startswith("partition-bond-")
    )
    for work, accounting in zip(metadata["candidate_work"], metadata["probe_accounting"]):
        assert work["documents_probed"]["quality"] == "exact"
        assert work["partitions_probed"]["quality"] == "exact"
        assert work["documents_probed"]["value"] == accounting["documents_probed"]
        assert work["partitions_probed"]["value"] == accounting["partitions_probed"]


def test_stage4_renderers_only_need_saved_envelopes(migrated_outputs):
    output, e01, e02, e03 = migrated_outputs
    figures = (
        render_fixed_candidates(e01.equal_work.output_path, output / "e01.png"),
        render_seeded_tau(e02.output_path, output / "e02.png"),
        render_partition_frontier(e03.output_path, output / "e03.png"),
    )
    assert all(path.stat().st_size > 0 for path in figures)


@pytest.mark.parametrize(
    ("experiment", "kernel"),
    (
        ("e01_fixed_candidate_arms", {"bound": "tight", "order": "natural", "checkpoints": [112], "shrink": 1.0}),
        ("e02_seeded_tau_recovery", {"bound": "tight", "order": "natural", "checkpoints": [112], "shrink": 1.0}),
        ("e03_partitioned_fused_scan", {"bound": "tight", "order": "natural", "checkpoints": [112], "shrink": 1.0}),
    ),
)
def test_full_scifact_controls_match_frozen_historical_methodology(experiment, kernel):
    result = assert_historical_controls(
        REPO_ROOT / "results" / "json" / f"stage4_integration_{experiment}_scifact.json",
        {
            "experiment": experiment,
            "dataset": "scifact",
            "n_queries": 50,
            "query_seed": 42,
            "k": 10,
            "kernel": kernel,
        },
    )
    assert result["status"] == "pass"
