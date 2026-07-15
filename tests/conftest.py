"""Research test-tier classification and native-required enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest


_NATIVE_BINARIES = (
    Path("cpp/per_document_oracle/per_document_oracle.so"),
    Path("cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.so"),
    Path("cpp/fused_panel_maxsim/fused_panel_maxsim.so"),
)
_NATIVE_GATE_MODULES = {
    "test_runner_gate.py",
    "test_wide_block_gate.py",
    "test_fused_panel_gate.py",
}
_NATIVE_MODULES = _NATIVE_GATE_MODULES | {
    "test_native_validation.py",
    "test_native_build_profiles.py",
}
_INTEGRATION_PREFIXES = (
    "test_checkpoint_ablation_pilot.py",
    "test_experiment_end_to_end.py",
    "test_mechanism_experiment.py",
    "test_stage4_baselines.py",
    "test_stage3_driver_migration",
    "test_stage4_driver_migration",
    "test_stage5_driver_migration",
    "test_session_persistence",
)
_ARTIFACT_MODULES = {
    "test_artifact_catalog.py",
    "test_paper_evidence.py",
    "test_paper_renderers.py",
    "test_result_infrastructure.py",
}
_REPRODUCTION_MODULES = {
    "test_finalization_contracts.py",
    "test_reproducibility_foundation.py",
    "test_workload_metadata.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--native-required",
        action="store_true",
        help="fail if native binaries are missing or no native gate executes",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Classify existing tests without scattering bookkeeping decorators."""
    root = Path(str(config.rootpath))
    missing = [str(path) for path in _NATIVE_BINARIES if not (root / path).is_file()]
    if config.getoption("--native-required") and missing:
        raise pytest.UsageError(
            "native-required tier is missing compiled libraries: " + ", ".join(missing)
        )

    for item in items:
        filename = Path(str(item.path)).name
        assigned = False
        if filename in _NATIVE_MODULES:
            item.add_marker("native")
            assigned = True
        if filename in _NATIVE_GATE_MODULES or (
            filename == "test_native_validation.py"
            and item.name.startswith(
                (
                    "test_reusable_validated_corpora",
                    "test_cpp_entry_points_return_error",
                )
            )
        ):
            item.add_marker("native_gate")
        if filename.startswith(_INTEGRATION_PREFIXES):
            item.add_marker("integration")
            assigned = True
        if filename in _ARTIFACT_MODULES or filename.startswith("test_research_audit"):
            item.add_marker("artifact")
            item.add_marker("research_audit")
            assigned = True
        if filename in _REPRODUCTION_MODULES:
            item.add_marker("reproduction")
            assigned = True
        if filename == "test_native_build_profiles.py":
            item.add_marker("sanitizer")
        if not assigned:
            item.add_marker("unit")

    config._bondmaxsim_native_gate_reports = 0  # type: ignore[attr-defined]


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]):
    """Count native gate call phases on the config shared with sessionfinish."""
    skipped = call.excinfo is not None and isinstance(
        call.excinfo.value, pytest.skip.Exception
    )
    if call.when == "call" and "native_gate" in item.keywords and not skipped:
        item.config._bondmaxsim_native_gate_reports += 1  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("--native-required"):
        return
    executed = session.config._bondmaxsim_native_gate_reports  # type: ignore[attr-defined]
    if executed == 0 and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write_line(
                "ERROR: native-required tier executed zero native gate tests",
                red=True,
            )
