from __future__ import annotations

from bondmaxsim.experiments.stage3.bound_slack import BoundSlackConfig, run_bound_slack
from bondmaxsim.experiments.stage3.stage2_audit import (
    run_exact_agreement_audit,
    run_normalization_audit,
)
from bondmaxsim.experiments.stage3.survival import SurvivalConfig, run_survival
from bondmaxsim.render.stage3.bound_slack import render_bound_slack
from bondmaxsim.render.stage3.survival import render_survival
from bondmaxsim.results import load_result


def test_e01_fixture_writes_versioned_accounting_and_renders(tmp_path):
    result = run_bound_slack(BoundSlackConfig.fixture_config(), output_dir=tmp_path)
    loaded = load_result(result.output_path)
    assert loaded.payload_kind == "pruning_accounting"
    assert loaded.protocol["timing_evidence"] is False
    assert loaded.protocol["workload"]["sample_size"] == 4
    assert {row["dimension_order"] for row in loaded.payload["rows"]} == {"natural", "bond"}
    output = render_bound_slack(result.output_path, tmp_path / "e01.png")
    assert output.is_file()


def test_e02_fixture_calls_native_accounting_and_retains_per_query_rows(tmp_path):
    result = run_survival(SurvivalConfig.fixture_config(), output_dir=tmp_path)
    loaded = load_result(result.output_path)
    arm = next(row for row in loaded.payload["rows"] if row.get("row_type") != "numpy-crosscheck")
    assert arm["agreement"]["boundary_tie_equivalent"] is True
    assert len(arm["per_query_accounting"]) == 4
    assert arm["fetch_boundary_dims"] == [4, 8]
    output = render_survival(result.output_path, tmp_path / "e02.png")
    assert output.is_file()


def test_stage2_fixture_audits_are_versioned_and_pass(tmp_path):
    normalization = run_normalization_audit("synthetic-small-v1", fixture=True, output_dir=tmp_path)
    agreement = run_exact_agreement_audit("synthetic-small-v1", fixture=True, output_dir=tmp_path)
    for result in (normalization, agreement):
        loaded = load_result(result.output_path)
        assert loaded.payload_kind == "validation_audit"
        assert loaded.payload["passed"] is True
        assert all(check["status"] == "pass" for check in loaded.payload["checks"])
    assert len(agreement.envelope.payload["checks"]) == 4
