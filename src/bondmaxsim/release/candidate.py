"""Fail-closed Stage 7 release-candidate provenance snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bondmaxsim.config import CORECT_COMMIT, PDX_COMMIT, PDX_SIGMOD_COMMIT, REPO_ROOT
from bondmaxsim.data.config import DATASETS
from bondmaxsim.data.freeze import (
    DEFAULT_FREEZE_MANIFEST,
    DEFAULT_GENERATED_ROOT,
    validate_freeze_bundle,
)
from bondmaxsim.experiments.environment import capture_environment


SCHEMA_NAME = "bondmaxsim.release-candidate"
SCHEMA_VERSION = "1.0.0"
BUILD_PROFILES = ("portable", "paper-native", "sanitize")
KERNELS = {
    "per_document_oracle": Path("cpp/per_document_oracle/per_document_oracle.so"),
    "wide_block_maxsim_bond": Path(
        "cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.so"
    ),
    "fused_panel_maxsim": Path("cpp/fused_panel_maxsim/fused_panel_maxsim.so"),
}
PREPARATION_ITEMS = (
    "unrelated_applications_closed",
    "background_jobs_stopped",
    "machine_quiescent",
    "power_and_governor_recorded",
    "thermal_state_stable",
)
_PINNED_SUBMODULES = {
    "extern/CoRECT": CORECT_COMMIT,
    "extern/PDX": PDX_COMMIT,
    "extern/PDX-sigmod": PDX_SIGMOD_COMMIT,
}
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReleaseCandidateError(RuntimeError):
    """The Stage 7 release candidate is incomplete or has drifted."""


CommandProbe = Callable[[tuple[str, ...], Path], str]
EnvironmentProbe = Callable[[], Mapping[str, Any]]
DataFreezeProbe = Callable[[Path, Path], Mapping[str, Any]]


@dataclass(frozen=True)
class CandidateProbes:
    """Injected read-only probes for deterministic fixture-scale tests."""

    command: CommandProbe
    environment: EnvironmentProbe
    data_freeze: DataFreezeProbe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReleaseCandidateError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseCandidateError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ReleaseCandidateError(f"{label} must contain a JSON object")
    return value


def _relative(path: Path, workspace: Path, label: str) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError as error:
        raise ReleaseCandidateError(f"{label} must be inside the workspace") from error


def _production_command(arguments: tuple[str, ...], workspace: Path) -> str:
    try:
        return subprocess.run(
            arguments,
            cwd=workspace,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.rstrip("\r\n")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseCandidateError(
            f"command failed: {' '.join(arguments)}: {error}"
        ) from error


def _production_data_freeze(
    freeze_path: Path, generated_root: Path
) -> Mapping[str, Any]:
    try:
        return validate_freeze_bundle(
            freeze_path, generated_root=generated_root, fixture=False
        )
    except Exception as error:
        raise ReleaseCandidateError(f"production data freeze is invalid: {error}") from error


def _production_probes() -> CandidateProbes:
    return CandidateProbes(
        command=_production_command,
        environment=lambda: capture_environment("paper-native"),
        data_freeze=_production_data_freeze,
    )


def _preparation_record(preparation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(preparation, Mapping) or set(preparation) != set(PREPARATION_ITEMS):
        raise ReleaseCandidateError(
            f"preparation checklist must contain exactly {list(PREPARATION_ITEMS)}"
        )
    record: dict[str, Any] = {}
    for item in PREPARATION_ITEMS:
        row = preparation[item]
        if not isinstance(row, Mapping) or set(row) != {"completed", "note"}:
            raise ReleaseCandidateError(
                f"preparation.{item} must contain completed and note"
            )
        if row["completed"] is not True:
            raise ReleaseCandidateError(f"preparation.{item} is incomplete")
        if not isinstance(row["note"], str) or not row["note"].strip():
            raise ReleaseCandidateError(f"preparation.{item}.note must be nonempty")
        record[item] = {"completed": True, "note": row["note"].strip()}
    return record


def _git_record(workspace: Path, command: CommandProbe) -> dict[str, Any]:
    revision = command(("git", "rev-parse", "HEAD"), workspace)
    if not _COMMIT.fullmatch(revision):
        raise ReleaseCandidateError("benchmark Git revision is not a full commit")
    status_text = command(
        ("git", "status", "--porcelain", "--untracked-files=all"), workspace
    )
    status = status_text.splitlines() if status_text else []
    if status:
        raise ReleaseCandidateError("benchmark worktree must be clean")
    return {"commit": revision, "dirty": False, "status": []}


def _submodule_record(workspace: Path, command: CommandProbe) -> dict[str, str]:
    raw = command(("git", "submodule", "status", "--recursive"), workspace)
    rows: dict[str, str] = {}
    for line in raw.splitlines():
        if len(line) < 42:
            raise ReleaseCandidateError("invalid recursive submodule status")
        state = line[0]
        fields = line[1:].strip().split()
        if state != " " or len(fields) < 2 or not _COMMIT.fullmatch(fields[0]):
            raise ReleaseCandidateError(
                "all recursive submodules must be initialized at recorded commits"
            )
        path = fields[1]
        if path in rows:
            raise ReleaseCandidateError(f"duplicate submodule path {path}")
        rows[path] = fields[0]
    if not rows:
        raise ReleaseCandidateError("no recursive submodule commits were recorded")
    for path, abbreviated in _PINNED_SUBMODULES.items():
        if path not in rows or not rows[path].startswith(abbreviated):
            raise ReleaseCandidateError(f"submodule {path} is not at its pinned commit")
    return dict(sorted(rows.items()))


def _lockfile_record(workspace: Path) -> dict[str, Any]:
    path = workspace / "uv.lock"
    if not path.is_file():
        raise ReleaseCandidateError("uv.lock is missing")
    return {"path": "uv.lock", "sha256": _sha256(path), "bytes": path.stat().st_size}


def _native_record(
    workspace: Path, command: CommandProbe
) -> tuple[dict[str, Any], dict[str, str]]:
    kernels: dict[str, Any] = {}
    compiler_identities: set[tuple[str, str]] = set()
    active_hashes: dict[str, str] = {}
    for kernel, active_relative in KERNELS.items():
        active_path = workspace / active_relative
        if not active_path.is_file():
            raise ReleaseCandidateError(f"missing active native binary {active_relative}")
        active_sha256 = _sha256(active_path)
        active_hashes[str(active_relative)] = active_sha256
        profile_records: dict[str, Any] = {}
        for profile in BUILD_PROFILES:
            directory = active_relative.parent / "build" / profile
            manifest_path = workspace / directory / "manifest.json"
            manifest = _read_json(manifest_path, f"{kernel} {profile} build manifest")
            if (
                manifest.get("schema_name") != "bondmaxsim.native-build-manifest"
                or manifest.get("schema_version") != "1.0.0"
                or manifest.get("profile") != profile
            ):
                raise ReleaseCandidateError(
                    f"{kernel} has an invalid {profile} build manifest"
                )
            binary_name = manifest.get("binary_name")
            compiler_path = manifest.get("compiler_path")
            compiler_version = manifest.get("compiler_version")
            binary_sha256 = manifest.get("binary_sha256")
            flags = manifest.get("flags")
            if (
                not isinstance(binary_name, str)
                or binary_name != active_relative.name
                or not isinstance(compiler_path, str)
                or not compiler_path
                or not isinstance(compiler_version, str)
                or not compiler_version
                or not isinstance(binary_sha256, str)
                or not _SHA256.fullmatch(binary_sha256)
                or not isinstance(flags, list)
                or not flags
                or any(not isinstance(flag, str) or not flag for flag in flags)
            ):
                raise ReleaseCandidateError(
                    f"{kernel} {profile} build provenance is incomplete"
                )
            profile_binary = workspace / directory / binary_name
            if not profile_binary.is_file() or _sha256(profile_binary) != binary_sha256:
                raise ReleaseCandidateError(
                    f"{kernel} {profile} binary does not match its manifest"
                )
            version_output = command((compiler_path, "--version"), workspace)
            if not version_output or version_output.splitlines()[0] != compiler_version:
                raise ReleaseCandidateError(
                    f"{kernel} {profile} compiler identity no longer matches"
                )
            compiler_identities.add((compiler_path, compiler_version))
            profile_records[profile] = {
                "manifest_path": str(directory / "manifest.json"),
                "manifest_sha256": _sha256(manifest_path),
                "binary_path": str(directory / binary_name),
                "binary_sha256": binary_sha256,
                "flags": list(flags),
            }
        paper_native_sha = profile_records["paper-native"]["binary_sha256"]
        if active_sha256 != paper_native_sha:
            raise ReleaseCandidateError(
                f"active {kernel} binary is not the paper-native build"
            )
        kernels[kernel] = {
            "active_binary": {
                "path": str(active_relative),
                "sha256": active_sha256,
            },
            "profiles": profile_records,
        }
    if len(compiler_identities) != 1:
        raise ReleaseCandidateError("native profiles do not share one compiler identity")
    compiler_path, compiler_version = next(iter(compiler_identities))
    return (
        {
            "required_profiles": list(BUILD_PROFILES),
            "active_profile": "paper-native",
            "compiler": {"path": compiler_path, "version": compiler_version},
            "kernels": kernels,
        },
        active_hashes,
    )


def _data_freeze_record(
    workspace: Path,
    freeze_path: Path,
    generated_root: Path,
    validator: DataFreezeProbe,
    *,
    fixture: bool,
) -> dict[str, Any]:
    if not freeze_path.is_file():
        kind = "fixture" if fixture else "production"
        raise ReleaseCandidateError(f"{kind} data freeze is missing: {freeze_path}")
    freeze = validator(freeze_path, generated_root)
    if (
        not isinstance(freeze, Mapping)
        or freeze.get("schema_name") != "bondmaxsim.data-freeze"
        or freeze.get("schema_version") != "1.0.0"
        or freeze.get("fixture") is not fixture
    ):
        raise ReleaseCandidateError("validated data freeze identity is invalid")
    configuration = freeze.get("configuration")
    datasets = freeze.get("datasets")
    if (
        not isinstance(configuration, Mapping)
        or not _SHA256.fullmatch(str(configuration.get("sha256", "")))
        or not isinstance(datasets, Mapping)
        or set(datasets) != set(DATASETS)
    ):
        raise ReleaseCandidateError("data freeze configuration/dataset scope is incomplete")
    dataset_records: dict[str, Any] = {}
    for dataset in DATASETS:
        row = datasets[dataset]
        if not isinstance(row, Mapping) or not _SHA256.fullmatch(
            str(row.get("data_sha256", ""))
        ):
            raise ReleaseCandidateError(f"{dataset}: frozen data provenance is invalid")
        indexes = row.get("indexes")
        workloads = row.get("workloads")
        if not isinstance(indexes, Mapping) or not indexes:
            raise ReleaseCandidateError(f"{dataset}: frozen indexes are missing")
        if not isinstance(workloads, Mapping) or set(workloads) != {"mechanism", "quality"}:
            raise ReleaseCandidateError(f"{dataset}: frozen workloads are incomplete")
        index_hashes = {}
        for backend, index in sorted(indexes.items()):
            if not isinstance(index, Mapping) or not _SHA256.fullmatch(
                str(index.get("index_sha256", ""))
            ):
                raise ReleaseCandidateError(
                    f"{dataset}: invalid {backend} index provenance"
                )
            index_hashes[str(backend)] = index["index_sha256"]
        workload_records = {}
        for regime, workload in sorted(workloads.items()):
            if (
                not isinstance(workload, Mapping)
                or not isinstance(workload.get("workload_id"), str)
                or not workload["workload_id"]
                or not _SHA256.fullmatch(str(workload.get("sha256", "")))
            ):
                raise ReleaseCandidateError(
                    f"{dataset}: invalid {regime} workload provenance"
                )
            workload_records[str(regime)] = {
                "workload_id": workload["workload_id"],
                "sha256": workload["sha256"],
            }
        dataset_records[dataset] = {
            "data_sha256": row["data_sha256"],
            "indexes": index_hashes,
            "workloads": workload_records,
        }
    return {
        "path": _relative(freeze_path, workspace, "data freeze"),
        "sha256": _sha256(freeze_path),
        "validated_document_sha256": _canonical_sha256(freeze),
        "configuration_sha256": configuration["sha256"],
        "datasets": dataset_records,
    }


def _environment_record(
    snapshot: Mapping[str, Any],
    git_record: Mapping[str, Any],
    lockfile: Mapping[str, Any],
    native_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        raise ReleaseCandidateError("environment snapshot must be an object")
    for field in ("captured_at_utc", "session", "host", "python", "toolchain", "threads", "code"):
        if field not in snapshot:
            raise ReleaseCandidateError(f"environment snapshot is missing {field}")
    try:
        captured_at = datetime.fromisoformat(
            str(snapshot["captured_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError:
        raise ReleaseCandidateError("environment capture time is not ISO-8601") from None
    if captured_at.tzinfo is None:
        raise ReleaseCandidateError("environment capture time must include a timezone")
    session = snapshot["session"]
    host = snapshot["host"]
    python = snapshot["python"]
    code = snapshot["code"]
    toolchain = snapshot["toolchain"]
    threads = snapshot["threads"]
    if (
        not isinstance(session, Mapping)
        or not isinstance(session.get("pid"), int)
        or session["pid"] <= 0
        or not isinstance(session.get("hostname"), str)
        or not session["hostname"]
    ):
        raise ReleaseCandidateError("environment session snapshot is incomplete")
    if (
        not isinstance(host, Mapping)
        or not host.get("system")
        or not host.get("machine")
        or not host.get("cpu_model")
        or not isinstance(host.get("logical_cpu_count"), int)
        or host["logical_cpu_count"] <= 0
        or not isinstance(host.get("cpu_governor"), str)
        or not host["cpu_governor"]
        or not isinstance(host.get("ac_online"), bool)
    ):
        raise ReleaseCandidateError("machine snapshot is incomplete")
    if (
        not isinstance(python, Mapping)
        or not isinstance(python.get("version"), str)
        or not python["version"]
        or not isinstance(python.get("implementation"), str)
        or not python["implementation"]
        or not isinstance(python.get("executable"), str)
        or not python["executable"]
        or not isinstance(python.get("packages"), Mapping)
        or not python["packages"]
        or not isinstance(threads, Mapping)
    ):
        raise ReleaseCandidateError("Python/thread environment snapshot is incomplete")
    if (
        not isinstance(code, Mapping)
        or code.get("git_revision") != git_record["commit"]
        or code.get("git_dirty") is not False
        or code.get("uv_lock_sha256") != lockfile["sha256"]
    ):
        raise ReleaseCandidateError("environment code provenance does not match preflight")
    if (
        not isinstance(toolchain, Mapping)
        or toolchain.get("build_profile") != "paper-native"
        or not isinstance(toolchain.get("uv"), str)
        or not toolchain["uv"]
        or not isinstance(toolchain.get("compiler"), str)
        or not toolchain["compiler"]
        or toolchain.get("native_binary_sha256") != native_hashes
    ):
        raise ReleaseCandidateError("environment native provenance does not match preflight")
    return json.loads(json.dumps(snapshot, sort_keys=True, allow_nan=False))


def _build_document(
    *,
    workspace: Path,
    freeze_path: Path,
    generated_root: Path,
    preparation: Mapping[str, Any],
    probes: CandidateProbes,
    fixture: bool,
) -> dict[str, Any]:
    git_record = _git_record(workspace, probes.command)
    submodules = _submodule_record(workspace, probes.command)
    lockfile = _lockfile_record(workspace)
    native, active_hashes = _native_record(workspace, probes.command)
    data_freeze = _data_freeze_record(
        workspace,
        freeze_path,
        generated_root,
        probes.data_freeze,
        fixture=fixture,
    )
    environment = _environment_record(
        probes.environment(), git_record, lockfile, active_hashes
    )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "fixture": fixture,
        "ready": True,
        "captured_at_utc": environment["captured_at_utc"],
        "benchmark": git_record,
        "submodules": submodules,
        "lockfile": lockfile,
        "native": native,
        "data_freeze": data_freeze,
        "environment": environment,
        "preparation": _preparation_record(preparation),
    }


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_release_candidate_document(
    document: Mapping[str, Any], *, allow_fixture: bool = False
) -> Mapping[str, Any]:
    """Validate the closed RC schema before it is accepted or serialized."""
    required = {
        "schema_name",
        "schema_version",
        "fixture",
        "ready",
        "captured_at_utc",
        "benchmark",
        "submodules",
        "lockfile",
        "native",
        "data_freeze",
        "environment",
        "preparation",
    }
    if not isinstance(document, Mapping) or set(document) != required:
        raise ReleaseCandidateError("release-candidate fields are not exact")
    if document.get("schema_name") != SCHEMA_NAME or document.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseCandidateError("unsupported release-candidate schema")
    if document.get("fixture") is True and not allow_fixture:
        raise ReleaseCandidateError("fixture release candidate cannot authorize production runs")
    if document.get("fixture") not in {True, False} or document.get("ready") is not True:
        raise ReleaseCandidateError("release candidate is not ready")
    if (
        not isinstance(document.get("captured_at_utc"), str)
        or not document["captured_at_utc"]
    ):
        raise ReleaseCandidateError("release-candidate capture time is invalid")
    benchmark = document.get("benchmark")
    if (
        not isinstance(benchmark, Mapping)
        or set(benchmark) != {"commit", "dirty", "status"}
        or not _COMMIT.fullmatch(str(benchmark.get("commit", "")))
        or benchmark.get("dirty") is not False
        or benchmark.get("status") != []
    ):
        raise ReleaseCandidateError("release-candidate Git state is invalid")
    lockfile = document.get("lockfile")
    if (
        not isinstance(lockfile, Mapping)
        or set(lockfile) != {"path", "sha256", "bytes"}
        or lockfile.get("path") != "uv.lock"
        or not _SHA256.fullmatch(str(lockfile.get("sha256", "")))
        or isinstance(lockfile.get("bytes"), bool)
        or not isinstance(lockfile.get("bytes"), int)
        or lockfile["bytes"] <= 0
    ):
        raise ReleaseCandidateError("release-candidate lockfile hash is invalid")
    submodules = document.get("submodules")
    if not isinstance(submodules, Mapping) or not submodules:
        raise ReleaseCandidateError("release-candidate submodules are invalid")
    if any(
        not isinstance(path, str)
        or not path
        or not _COMMIT.fullmatch(str(commit))
        for path, commit in submodules.items()
    ):
        raise ReleaseCandidateError("release-candidate submodule commit is invalid")
    for path, abbreviated in _PINNED_SUBMODULES.items():
        if path not in submodules or not submodules[path].startswith(abbreviated):
            raise ReleaseCandidateError(f"release-candidate submodule {path} is unpinned")
    native = document.get("native")
    if (
        not isinstance(native, Mapping)
        or set(native) != {
            "required_profiles",
            "active_profile",
            "compiler",
            "kernels",
        }
        or native.get("required_profiles") != list(BUILD_PROFILES)
        or native.get("active_profile") != "paper-native"
        or not isinstance(native.get("kernels"), Mapping)
        or set(native["kernels"]) != set(KERNELS)
    ):
        raise ReleaseCandidateError("release-candidate native profile scope is invalid")
    compiler = native.get("compiler")
    if (
        not isinstance(compiler, Mapping)
        or set(compiler) != {"path", "version"}
        or not isinstance(compiler.get("path"), str)
        or not compiler["path"]
        or not isinstance(compiler.get("version"), str)
        or not compiler["version"]
    ):
        raise ReleaseCandidateError("release-candidate compiler identity is invalid")
    active_hashes: dict[str, str] = {}
    for kernel, expected_active_path in KERNELS.items():
        row = native["kernels"][kernel]
        if not isinstance(row, Mapping) or set(row) != {"active_binary", "profiles"}:
            raise ReleaseCandidateError(f"release-candidate {kernel} record is invalid")
        active = row["active_binary"]
        profiles = row["profiles"]
        if (
            not isinstance(active, Mapping)
            or set(active) != {"path", "sha256"}
            or active.get("path") != str(expected_active_path)
            or not _SHA256.fullmatch(str(active.get("sha256", "")))
            or not isinstance(profiles, Mapping)
            or set(profiles) != set(BUILD_PROFILES)
        ):
            raise ReleaseCandidateError(
                f"release-candidate {kernel} binary/profile record is invalid"
            )
        active_hashes[str(expected_active_path)] = active["sha256"]
        for profile in BUILD_PROFILES:
            profile_row = profiles[profile]
            if (
                not isinstance(profile_row, Mapping)
                or set(profile_row)
                != {
                    "manifest_path",
                    "manifest_sha256",
                    "binary_path",
                    "binary_sha256",
                    "flags",
                }
                or not _SHA256.fullmatch(
                    str(profile_row.get("manifest_sha256", ""))
                )
                or not _SHA256.fullmatch(str(profile_row.get("binary_sha256", "")))
                or not isinstance(profile_row.get("manifest_path"), str)
                or not profile_row["manifest_path"]
                or not isinstance(profile_row.get("binary_path"), str)
                or not profile_row["binary_path"]
                or not isinstance(profile_row.get("flags"), list)
                or not profile_row["flags"]
                or any(
                    not isinstance(flag, str) or not flag
                    for flag in profile_row["flags"]
                )
            ):
                raise ReleaseCandidateError(
                    f"release-candidate {kernel} {profile} provenance is invalid"
                )
        if active["sha256"] != profiles["paper-native"]["binary_sha256"]:
            raise ReleaseCandidateError(
                f"release-candidate {kernel} active binary is not paper-native"
            )
    _preparation_record(document.get("preparation", {}))
    data_freeze = document.get("data_freeze")
    if (
        not isinstance(data_freeze, Mapping)
        or set(data_freeze)
        != {
            "path",
            "sha256",
            "validated_document_sha256",
            "configuration_sha256",
            "datasets",
        }
        or not isinstance(data_freeze.get("path"), str)
        or not data_freeze["path"]
        or any(
            not _SHA256.fullmatch(str(data_freeze.get(field, "")))
            for field in (
                "sha256",
                "validated_document_sha256",
                "configuration_sha256",
            )
        )
        or not isinstance(data_freeze.get("datasets"), Mapping)
        or set(data_freeze["datasets"]) != set(DATASETS)
    ):
        raise ReleaseCandidateError("release-candidate data scope is invalid")
    for dataset, row in data_freeze["datasets"].items():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"data_sha256", "indexes", "workloads"}
            or not _SHA256.fullmatch(str(row.get("data_sha256", "")))
            or not isinstance(row.get("indexes"), Mapping)
            or not row["indexes"]
            or any(
                not isinstance(backend, str)
                or not backend
                or not _SHA256.fullmatch(str(value))
                for backend, value in row["indexes"].items()
            )
            or not isinstance(row.get("workloads"), Mapping)
            or set(row["workloads"]) != {"mechanism", "quality"}
        ):
            raise ReleaseCandidateError(
                f"release-candidate {dataset} data/index provenance is invalid"
            )
        for regime, workload in row["workloads"].items():
            if (
                not isinstance(workload, Mapping)
                or set(workload) != {"workload_id", "sha256"}
                or not isinstance(workload.get("workload_id"), str)
                or not workload["workload_id"]
                or not _SHA256.fullmatch(str(workload.get("sha256", "")))
            ):
                raise ReleaseCandidateError(
                    f"release-candidate {dataset} {regime} workload is invalid"
                )
    environment = document.get("environment")
    normalized_environment = _environment_record(
        environment, benchmark, lockfile, active_hashes
    )
    if normalized_environment.get("captured_at_utc") != document["captured_at_utc"]:
        raise ReleaseCandidateError("release-candidate capture times disagree")
    return document


def create_release_candidate(
    output_path: Path,
    *,
    preparation: Mapping[str, Any],
    workspace: Path = REPO_ROOT,
    freeze_path: Path = DEFAULT_FREEZE_MANIFEST,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    fixture: bool = False,
    probes: CandidateProbes | None = None,
) -> Mapping[str, Any]:
    """Capture, validate, and atomically write one Stage 7 RC record."""
    if fixture != (probes is not None):
        raise ReleaseCandidateError(
            "fixture mode requires injected probes; production forbids injected probes"
        )
    if not Path(freeze_path).is_file():
        kind = "fixture" if fixture else "production"
        raise ReleaseCandidateError(f"{kind} data freeze is missing: {freeze_path}")
    selected_probes = probes or _production_probes()
    document = _build_document(
        workspace=Path(workspace),
        freeze_path=Path(freeze_path),
        generated_root=Path(generated_root),
        preparation=preparation,
        probes=selected_probes,
        fixture=fixture,
    )
    validate_release_candidate_document(document, allow_fixture=fixture)
    _atomic_json(output_path, document)
    return document


def validate_release_candidate(
    path: Path,
    *,
    allow_fixture: bool = False,
) -> Mapping[str, Any]:
    """Read and structurally validate an existing RC record."""
    return validate_release_candidate_document(
        _read_json(path, "release candidate"), allow_fixture=allow_fixture
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--preparation", type=Path, required=True)
    arguments = parser.parse_args(argv)
    preparation = _read_json(arguments.preparation, "preparation checklist")
    create_release_candidate(
        arguments.output,
        preparation=preparation,
        freeze_path=arguments.freeze,
        generated_root=arguments.generated_root,
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
