from __future__ import annotations

from pathlib import Path

from bondmaxsim.experiments.stage5.bm25 import Stage5BM25Config, run_bm25
from bondmaxsim.experiments.stage5.ir_evaluation import (
    Stage5IREvaluationConfig,
    run_ir_evaluation,
)
from bondmaxsim.experiments.stage5.significance import significance_from_ir_artifact
from bondmaxsim.render.stage5.ir import latex_rows, render_quality_latency
from bondmaxsim.results import load_result


def _environment():
    return {
        "session": {"session_id": "fixture-stage5-analysis"},
        "threads": {"OMP_NUM_THREADS": "1"},
        "code": {
            "git_revision": "fixture-revision",
            "git_dirty": False,
            "uv_lock_sha256": "fixture-lock",
        },
        "toolchain": {"native_binary_sha256": {"fused": "fixture-native"}},
    }


def test_e02_consumes_saved_per_query_values_without_retrieval_setup(tmp_path):
    source = run_ir_evaluation(
        Stage5IREvaluationConfig.fixture_config(),
        output_dir=tmp_path,
        session_id="fixture-stage5-analysis-e01",
        environment=_environment(),
    )
    appended = run_ir_evaluation(
        Stage5IREvaluationConfig.fixture_config(),
        output_dir=tmp_path,
        session_id="fixture-stage5-analysis-e01-confirmation",
        environment=_environment(),
    )
    assert len(appended.envelope.payload["sessions"]) == 2
    analysis = significance_from_ir_artifact(
        source.output_path,
        output_dir=tmp_path,
        n_permutations=100,
        seed=7,
        environment=_environment(),
    )
    result = load_result(analysis.output_path)
    assert result.payload_kind == "paired_significance"
    assert result.provenance["input_artifact_ids"] == [source.envelope.artifact_id]
    comparison = result.payload["comparisons"][0]
    assert comparison["candidate_arm_id"] == "partitioned-nprobe-001"
    assert comparison["n_queries"] == 4
    assert result.method_configuration["retrieval_setup"] == "consumed from validated e01 artifact"


def test_e03_fixture_marks_timing_standalone_and_renders(tmp_path):
    run = run_bm25(
        Stage5BM25Config.fixture_config(),
        output_dir=tmp_path,
        session_id="fixture-stage5-e03",
        environment=_environment(),
    )
    result = load_result(run.output_path)
    timing = result.method_configuration["timing_status"]
    assert timing["mode"] == "standalone"
    assert timing["comparable_to_vector_arm_margins"] is False
    assert result.provenance["command"].endswith("--fixture")
    assert result.payload["quality"]["corect_scope"] == "standard metrics only"
    assert result.payload["quality"]["rows"][0]["ndcg_at_10"] == 1.0

    table = latex_rows([run.output_path])
    assert "synthetic-small-v1" in table
    assert "bm25" in table
    figure = render_quality_latency(run.output_path, tmp_path / "bm25.png")
    assert figure.stat().st_size > 0


def test_owned_stage5_code_has_no_legacy_metric_api_names():
    root = Path(__file__).parents[1]
    paths = [
        *sorted((root / "experiments" / "stage5_corect").glob("*.py")),
        *sorted((root / "src" / "bondmaxsim" / "experiments" / "stage5").glob("*.py")),
        *sorted((root / "src" / "bondmaxsim" / "render" / "stage5").glob("*.py")),
    ]
    forbidden = ("compute_rc_metrics", "corect_smoke_test", "CoRECT_RC_metrics")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(name in text for name in forbidden), path
