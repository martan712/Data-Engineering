from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.render.paper.generator import (
    StalePaperOutputError,
    check_generated_outputs,
    generate_paper_assets,
)


def test_tracked_generated_assets_are_current_and_canonical() -> None:
    paths = check_generated_outputs(workspace=REPO_ROOT)
    relative = {path.relative_to(REPO_ROOT) for path in paths}
    assert Path("report/generated/tables/dense-kernel-rows.tex") in relative
    assert Path("report/generated/tables/exact-safe-rows.tex") in relative
    assert Path("report/generated/tables/ir-quality-rows.tex") in relative
    assert Path("report/generated/headline-macros.tex") in relative
    dense = (REPO_ROOT / "report/generated/tables/dense-kernel-rows.tex").read_text(encoding="utf-8")
    exact = (REPO_ROOT / "report/generated/tables/exact-safe-rows.tex").read_text(encoding="utf-8")
    macros = (REPO_ROOT / "report/generated/headline-macros.tex").read_text(encoding="utf-8")
    assert "SciFact & 54.2 & 121.3 & 13.9 & 45.8" in dense
    assert "ArguAna & 33.6 & $+8.7\\%$ & 97\\%" in exact
    assert "\\newcommand{\\DenseMedianSpeedup}{3.1\\ensuremath{\\times}}" in macros
    assert "\\newcommand{\\ExactSafeWinningDatasets}{1}" in macros


def test_plot_data_is_deterministic_normalized_csv() -> None:
    first = (REPO_ROOT / "report/generated/plot-data/bound-slack-scifact.csv").read_bytes()
    generate_paper_assets(workspace=REPO_ROOT)
    second = (REPO_ROOT / "report/generated/plot-data/bound-slack-scifact.csv").read_bytes()
    assert first == second
    assert first.startswith(b"dataset,dimension_order,dimensions_scanned,slack_mean")
    assert b"timestamp" not in first.lower()


def test_check_mode_detects_stale_output_without_rewriting(tmp_path: Path) -> None:
    (tmp_path / "artifacts").mkdir()
    shutil.copy2(REPO_ROOT / "artifacts/catalog.yaml", tmp_path / "artifacts/catalog.yaml")
    (tmp_path / "results").symlink_to(REPO_ROOT / "results", target_is_directory=True)
    paths = generate_paper_assets(workspace=tmp_path)
    target = next(path for path in paths if path.name == "headline-macros.tex")
    target.write_text("stale\n", encoding="utf-8")
    with pytest.raises(StalePaperOutputError, match="stale report/generated/headline-macros.tex"):
        check_generated_outputs(workspace=tmp_path)
    assert target.read_text(encoding="utf-8") == "stale\n"
