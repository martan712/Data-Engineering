from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bondmaxsim.config import CORECT_COMMIT, PDX_COMMIT, PDX_SIGMOD_COMMIT
from bondmaxsim.data.config import DATASETS
from bondmaxsim.release.candidate import (
    BUILD_PROFILES,
    KERNELS,
    PREPARATION_ITEMS,
    CandidateProbes,
    ReleaseCandidateError,
    create_release_candidate,
    validate_release_candidate,
)


_REVISION = "a" * 40
_COMPILER_VERSION = "fixture-clang version 1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preparation():
    return {
        item: {"completed": True, "note": f"fixture confirmed {item}"}
        for item in PREPARATION_ITEMS
    }


def _freeze_document(*, fixture: bool = True):
    digest = "d" * 64
    return {
        "schema_name": "bondmaxsim.data-freeze",
        "schema_version": "1.0.0",
        "fixture": fixture,
        "configuration": {"path": "configs/data/v1.json", "sha256": "c" * 64},
        "datasets": {
            dataset: {
                "data_sha256": digest,
                "indexes": {
                    "faiss": {"index_sha256": "e" * 64},
                    "plaid": {"index_sha256": "f" * 64},
                },
                "workloads": {
                    "mechanism": {
                        "workload_id": f"{dataset}-mechanism",
                        "sha256": "1" * 64,
                    },
                    "quality": {
                        "workload_id": f"{dataset}-quality",
                        "sha256": "2" * 64,
                    },
                },
            }
            for dataset in DATASETS
        },
    }


def _write_native_fixture(workspace: Path):
    compiler = workspace / "toolchain/clang++"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("fixture compiler", encoding="utf-8")
    active_hashes = {}
    for kernel, active_relative in KERNELS.items():
        active = workspace / active_relative
        active.parent.mkdir(parents=True, exist_ok=True)
        for profile in BUILD_PROFILES:
            profile_dir = active.parent / "build" / profile
            profile_dir.mkdir(parents=True, exist_ok=True)
            binary = profile_dir / active.name
            content = (
                f"{kernel}-paper-native" if profile == "paper-native" else f"{kernel}-{profile}"
            ).encode()
            binary.write_bytes(content)
            manifest = {
                "schema_name": "bondmaxsim.native-build-manifest",
                "schema_version": "1.0.0",
                "profile": profile,
                "binary_name": active.name,
                "binary_sha256": _sha256(binary),
                "compiler_path": str(compiler),
                "compiler_version": _COMPILER_VERSION,
                "flags": ["-std=c++20", f"-DPROFILE={profile}"],
            }
            (profile_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        active.write_bytes((active.parent / "build/paper-native" / active.name).read_bytes())
        active_hashes[str(active_relative)] = _sha256(active)
    return compiler, active_hashes


def _submodule_status():
    return "\n".join(
        (
            f" {CORECT_COMMIT.ljust(40, '0')} extern/CoRECT (heads/main)",
            f" {PDX_COMMIT.ljust(40, '0')} extern/PDX (heads/main)",
            f" {PDX_SIGMOD_COMMIT.ljust(40, '0')} extern/PDX-sigmod (heads/main)",
        )
    )


def _fixture_setup(tmp_path: Path, *, dirty: bool = False, freeze=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "uv.lock").write_text("fixture-lock", encoding="utf-8")
    compiler, active_hashes = _write_native_fixture(workspace)
    freeze_path = workspace / "artifacts/data/freeze-v1.json"
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text("{}\n", encoding="utf-8")
    freeze_document = freeze or _freeze_document()

    def command(arguments, _workspace):
        if arguments == ("git", "rev-parse", "HEAD"):
            return _REVISION
        if arguments == ("git", "status", "--porcelain", "--untracked-files=all"):
            return " M src/dirty.py" if dirty else ""
        if arguments == ("git", "submodule", "status", "--recursive"):
            return _submodule_status()
        if arguments == (str(compiler), "--version"):
            return _COMPILER_VERSION + "\nfixture target"
        raise AssertionError(f"unexpected command {arguments}")

    lock_sha = _sha256(workspace / "uv.lock")
    environment = {
        "captured_at_utc": "2026-07-15T12:00:00+00:00",
        "session": {"pid": 1, "hostname": "fixture-host"},
        "host": {
            "system": "Linux",
            "release": "fixture",
            "machine": "x86_64",
            "cpu_model": "fixture-cpu",
            "logical_cpu_count": 8,
            "cpu_governor": "performance",
            "ac_online": True,
        },
        "python": {
            "version": "3.12",
            "implementation": "CPython",
            "executable": "/fixture/python",
            "packages": {"bondmaxsim": "fixture"},
        },
        "toolchain": {
            "uv": "uv fixture",
            "compiler": _COMPILER_VERSION,
            "build_profile": "paper-native",
            "native_binary_sha256": active_hashes,
        },
        "threads": {"OMP_NUM_THREADS": "1"},
        "code": {
            "git_revision": _REVISION,
            "git_dirty": False,
            "git_status": [],
            "submodules": _submodule_status().splitlines(),
            "uv_lock_sha256": lock_sha,
        },
    }
    probes = CandidateProbes(
        command=command,
        environment=lambda: environment,
        data_freeze=lambda _path, _root: freeze_document,
    )
    return workspace, freeze_path, probes


def test_fixture_release_candidate_records_complete_preflight_atomically(tmp_path: Path):
    workspace, freeze_path, probes = _fixture_setup(tmp_path)
    output = workspace / "artifacts/release/rc-v1.json"
    document = create_release_candidate(
        output,
        preparation=_preparation(),
        workspace=workspace,
        freeze_path=freeze_path,
        generated_root=workspace / "generated",
        fixture=True,
        probes=probes,
    )

    assert document["ready"] is True
    assert document["benchmark"] == {"commit": _REVISION, "dirty": False, "status": []}
    assert set(document["submodules"]) == {
        "extern/CoRECT",
        "extern/PDX",
        "extern/PDX-sigmod",
    }
    assert document["lockfile"]["sha256"] == _sha256(workspace / "uv.lock")
    assert document["native"]["required_profiles"] == list(BUILD_PROFILES)
    assert set(document["native"]["kernels"]) == set(KERNELS)
    assert set(document["data_freeze"]["datasets"]) == set(DATASETS)
    assert document["environment"]["host"]["cpu_model"] == "fixture-cpu"
    assert validate_release_candidate(output, allow_fixture=True) == document
    assert not list(output.parent.glob(".*.tmp"))


def test_fixture_candidate_cannot_authorize_production(tmp_path: Path):
    workspace, freeze_path, probes = _fixture_setup(tmp_path)
    output = workspace / "rc.json"
    create_release_candidate(
        output,
        preparation=_preparation(),
        workspace=workspace,
        freeze_path=freeze_path,
        fixture=True,
        probes=probes,
    )
    with pytest.raises(ReleaseCandidateError, match="cannot authorize production"):
        validate_release_candidate(output)


def test_dirty_worktree_is_release_blocking(tmp_path: Path):
    workspace, freeze_path, probes = _fixture_setup(tmp_path, dirty=True)
    output = workspace / "rc.json"
    with pytest.raises(ReleaseCandidateError, match="worktree must be clean"):
        create_release_candidate(
            output,
            preparation=_preparation(),
            workspace=workspace,
            freeze_path=freeze_path,
            fixture=True,
            probes=probes,
        )
    assert not output.exists()


def test_production_missing_data_freeze_fails_before_any_probe(tmp_path: Path):
    with pytest.raises(ReleaseCandidateError, match="production data freeze is missing"):
        create_release_candidate(
            tmp_path / "rc.json",
            preparation=_preparation(),
            workspace=tmp_path,
            freeze_path=tmp_path / "missing-freeze.json",
        )


def test_invalid_data_freeze_scope_is_refused(tmp_path: Path):
    invalid = _freeze_document()
    invalid["datasets"].pop("scidocs")
    workspace, freeze_path, probes = _fixture_setup(tmp_path, freeze=invalid)
    with pytest.raises(ReleaseCandidateError, match="dataset scope"):
        create_release_candidate(
            workspace / "rc.json",
            preparation=_preparation(),
            workspace=workspace,
            freeze_path=freeze_path,
            fixture=True,
            probes=probes,
        )


def test_active_binary_must_be_paper_native(tmp_path: Path):
    workspace, freeze_path, probes = _fixture_setup(tmp_path)
    active = workspace / KERNELS["per_document_oracle"]
    active.write_bytes(b"wrong-active-profile")
    with pytest.raises(ReleaseCandidateError, match="not the paper-native build"):
        create_release_candidate(
            workspace / "rc.json",
            preparation=_preparation(),
            workspace=workspace,
            freeze_path=freeze_path,
            fixture=True,
            probes=probes,
        )


def test_missing_build_profile_is_refused(tmp_path: Path):
    workspace, freeze_path, probes = _fixture_setup(tmp_path)
    manifest = (
        workspace
        / KERNELS["wide_block_maxsim_bond"].parent
        / "build/sanitize/manifest.json"
    )
    manifest.unlink()
    with pytest.raises(ReleaseCandidateError, match="cannot read"):
        create_release_candidate(
            workspace / "rc.json",
            preparation=_preparation(),
            workspace=workspace,
            freeze_path=freeze_path,
            fixture=True,
            probes=probes,
        )


def test_incomplete_preparation_is_refused(tmp_path: Path):
    workspace, freeze_path, probes = _fixture_setup(tmp_path)
    preparation = _preparation()
    preparation["machine_quiescent"]["completed"] = False
    with pytest.raises(ReleaseCandidateError, match="machine_quiescent is incomplete"):
        create_release_candidate(
            workspace / "rc.json",
            preparation=preparation,
            workspace=workspace,
            freeze_path=freeze_path,
            fixture=True,
            probes=probes,
        )
