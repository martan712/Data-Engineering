from __future__ import annotations

from pathlib import Path

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.experiments.checkpoint_ablation import (
    CheckpointAblationConfig,
    assert_historical_methodology,
    run_checkpoint_ablation,
)
from bondmaxsim.results import load_result
from bondmaxsim.render.checkpoint_ablation import render_checkpoint_ablation


def _environment():
    return {
        "session": {"session_id": "fixture-e08-session"},
        "threads": {"OMP_NUM_THREADS": "1"},
        "code": {
            "git_revision": "fixture-revision",
            "git_dirty": False,
            "uv_lock_sha256": "fixture-lock",
        },
        "toolchain": {"native_binary_sha256": {"fused": "fixture-native"}},
    }


def test_e08_fixture_runs_serializes_and_renders_from_saved_result(tmp_path):
    run = run_checkpoint_ablation(
        CheckpointAblationConfig.fixture_config(),
        output_dir=tmp_path,
        session_id="fixture-e08-session",
        environment=_environment(),
    )
    loaded = load_result(run.output_path)
    assert loaded == run.envelope
    assert loaded.payload["sessions"][0]["complete"]
    assert len(loaded.payload["sessions"][0]["observations"]) == 12
    assert loaded.protocol["workload"]["token_counts"] == [1, 2, 3, 2]
    assert loaded.payload["sessions"][0]["result_metadata"][0]["pruning_accounting"]

    figure = tmp_path / "e08.png"
    assert render_checkpoint_ablation(run.output_path, figure) == figure
    assert figure.stat().st_size > 0


def test_e08_scifact_controls_match_frozen_historical_methodology():
    result = assert_historical_methodology(
        CheckpointAblationConfig(),
        REPO_ROOT / "results" / "json"
        / "stage3_mechanism_e08_checkpoint_ablation_scifact.json",
    )
    assert result["status"] == "pass"
    historical = load_result(
        REPO_ROOT / "artifacts" / "baseline" / "fixtures"
        / "shape_07__stage3_mechanism_e08_checkpoint_ablation_nfcorpus.json"
    )
    assert historical.payload["historical_migration"] is True
