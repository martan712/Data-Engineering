"""Static and functional checks for the three native build profiles."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
KERNEL_DIRS = (
    "cpp/per_document_oracle",
    "cpp/wide_block_maxsim_bond",
    "cpp/fused_panel_maxsim",
)


def test_each_makefile_defines_consistent_profiles():
    for relative in KERNEL_DIRS:
        source = (REPO_ROOT / relative / "Makefile").read_text(encoding="utf-8")
        assert "BUILD ?= portable" in source
        assert "$(origin CXX),default" in source
        assert "paper-native" in source
        assert "sanitize" in source
        assert "-march=x86-64-v2" in source
        assert "-march=native" in source
        assert "-fsanitize=address,undefined" in source
        assert "native_build_manifest.py" in source


def test_manifest_writer_records_binary_and_flags(tmp_path):
    binary = tmp_path / "tiny.so"
    binary.write_bytes(b"native fixture")
    output = tmp_path / "manifest.json"
    compiler = shutil.which("g++")
    assert compiler is not None
    subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts/native_build_manifest.py"),
            "--binary",
            str(binary),
            "--compiler",
            compiler,
            "--flags",
            "-O3 -fopenmp",
            "--profile",
            "portable",
            "--output",
            str(output),
        ],
        check=True,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["profile"] == "portable"
    assert manifest["flags"] == ["-O3", "-fopenmp"]
    assert manifest["openmp_enabled"] is True
    assert len(manifest["binary_sha256"]) == 64
