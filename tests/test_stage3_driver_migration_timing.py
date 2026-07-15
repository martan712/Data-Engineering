from __future__ import annotations

from bondmaxsim.experiments.stage3.kernel_comparison import (
    ExactSafeInterleavedConfig,
    KernelComparisonConfig,
    run_exact_safe_interleaved,
    run_kernel_comparison,
)
from bondmaxsim.render.stage3.kernel_comparison import render_kernel_comparison
from bondmaxsim.results import load_result


def test_e03_fixture_uses_shared_counterbalanced_protocol(tmp_path):
    result = run_kernel_comparison(
        KernelComparisonConfig.fixture_config(),
        output_dir=tmp_path,
        session_id="e03-fixture-test",
    )
    loaded = load_result(result.output_path)
    session = loaded.payload["sessions"][0]
    assert session["complete"] is True
    assert session["protocol"] == {
        "protocol_class": "fixture",
        "warmup_rounds": 1,
        "measured_rounds": 3,
    }
    assert len(session["observations"]) == len(session["arm_ids"]) * 4
    assert "dense-fused" in session["arm_ids"]
    assert "dense-numpy" in session["arm_ids"]
    assert session["validation_results"]
    assert session["result_metadata"]
    token_metadata = next(
        row
        for row in session["result_metadata"]
        if row["arm_id"].startswith("bond-token-")
    )
    assert "tokens_pruned_pct" in token_metadata["pruning_accounting"][0]
    assert session["paired_comparisons"]
    output = render_kernel_comparison(result.output_path, tmp_path / "e03.png")
    assert output.is_file()


def test_r12c_fixture_pairs_each_late_checkpoint_with_dense(tmp_path):
    result = run_exact_safe_interleaved(
        ExactSafeInterleavedConfig.fixture_config(),
        output_dir=tmp_path,
        session_id="r12c-fixture-test",
    )
    loaded = load_result(result.output_path)
    session = loaded.payload["sessions"][0]
    assert session["complete"] is True
    assert set(session["paired_comparisons"]) == {"bond-c4", "bond-c6", "bond-c4-6"}
    assert all(
        comparison["baseline_arm_id"] == "dense-fused"
        for comparison in session["paired_comparisons"].values()
    )
    assert loaded.method_configuration["supersedes_fields"] == [
        "historical-stage3-e08.wall-clock",
        "historical-stage3-e09.wall-clock",
    ]
