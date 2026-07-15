"""Tests for the locked-environment and deterministic fixture foundation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bondmaxsim.data.fixture import generate_fixture, validate_fixture
from bondmaxsim.experiments.environment import capture_environment, write_environment


REPO_ROOT = Path(__file__).parents[1]


def test_fixture_is_byte_deterministic_and_valid(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = generate_fixture(first)
    generate_fixture(second)

    assert manifest["counts"] == {
        "documents": 6,
        "document_tokens": 15,
        "queries": 4,
        "query_tokens": 8,
    }
    for filename in ("small.npz", "ids.json", "qrels.tsv", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert validate_fixture(first) == manifest


def test_fixture_validation_rejects_tampering(tmp_path):
    output = tmp_path / "fixture"
    generate_fixture(output)
    with (output / "qrels.tsv").open("ab") as target:
        target.write(b"q0\td5\t1\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_fixture(output)


def test_environment_capture_has_required_provenance(tmp_path):
    snapshot = capture_environment("portable")
    assert snapshot["host"]["machine"]
    assert snapshot["python"]["version"] == "3.12.13"
    assert snapshot["toolchain"]["build_profile"] == "portable"
    assert snapshot["code"]["git_revision"]
    assert snapshot["code"]["uv_lock_sha256"]

    output = tmp_path / "environment.json"
    write_environment(output, "sanitize")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["toolchain"]["build_profile"] == "sanitize"
    assert not list(tmp_path.glob(".*.tmp"))


def test_data_config_freezes_recovered_sources_and_encoder_behavior():
    config = json.loads((REPO_ROOT / "configs/data/v1.json").read_text())
    assert config["resolution_status"] == "frozen_for_regeneration"
    assert set(config["datasets"]) == {"scifact", "nfcorpus", "arguana", "scidocs"}
    assert config["text"]["document_fields"] == ["text"]
    assert config["text"]["include_title"] is False
    assert all(
        len(dataset[revision]) == 40
        for dataset in config["datasets"].values()
        for revision in ("corpus_queries_revision", "qrels_revision")
    )
    assert config["sources"]["model"]["revision"] == config["sources"]["tokenizer"]["revision"]
    assert config["encoding"]["document_length"] == 300
    assert config["encoding"]["query_length"] == 48
    assert config["encoding"]["truncation"] is True
    assert config["encoding"]["padding_tokens_retained"] is False


def test_setup_script_is_syntactically_valid_and_requires_a_mode():
    subprocess.run(["bash", "-n", "setup.sh"], cwd=REPO_ROOT, check=True)
    result = subprocess.run(
        ["bash", "setup.sh"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stdout


def test_top_level_makefile_exposes_required_targets():
    source = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in (
        "setup-paper",
        "data-small",
        "verify",
        "reproduce-core",
        "paper",
        "artifact-smoke",
        "data-full",
        "reproduce-full",
    ):
        assert f"{target}:" in source
