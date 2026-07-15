"""Capture software, host, code, and native-binary provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bondmaxsim.config import REPO_ROOT


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "FAISS_NUM_THREADS",
    "TORCH_NUM_THREADS",
)

NATIVE_BINARIES = (
    "cpp/per_document_oracle/per_document_oracle.so",
    "cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.so",
    "cpp/fused_panel_maxsim/fused_panel_maxsim.so",
)


def _command(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_first(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _cpu_model() -> str | None:
    source = _read_first(Path("/proc/cpuinfo"))
    if source is None:
        return platform.processor() or None
    for line in source.splitlines():
        if line.startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _packages() -> dict[str, str]:
    return dict(
        sorted(
            {
                (distribution.metadata.get("Name") or "unknown", distribution.version)
                for distribution in importlib.metadata.distributions()
            },
            key=lambda item: item[0].lower(),
        )
    )


def capture_environment(build_profile: str | None = None) -> dict[str, Any]:
    """Return a JSON-compatible snapshot without mutating host state."""
    git_revision = _command("git", "rev-parse", "HEAD")
    git_status = _command("git", "status", "--short")
    submodules = _command("git", "submodule", "status", "--recursive")
    lockfile = REPO_ROOT / "uv.lock"
    native = {}
    for relative in NATIVE_BINARIES:
        path = REPO_ROOT / relative
        native[relative] = _sha256(path) if path.is_file() else None

    governor = _read_first(
        Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    )
    ac_online = None
    for candidate in sorted(Path("/sys/class/power_supply").glob("*/online")):
        value = _read_first(candidate)
        if value in {"0", "1"}:
            ac_online = value == "1"
            break

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "session": {"pid": os.getpid(), "hostname": socket.gethostname()},
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_governor": governor,
            "ac_online": ac_online,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(os.sys.executable).resolve()),
            "packages": _packages(),
        },
        "toolchain": {
            "uv": _command("uv", "--version"),
            "compiler": (_command("g++", "--version") or "").splitlines()[0] or None,
            "build_profile": build_profile,
            "native_binary_sha256": native,
        },
        "threads": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES},
        "code": {
            "git_revision": git_revision,
            "git_dirty": bool(git_status),
            "git_status": git_status.splitlines() if git_status else [],
            "submodules": submodules.splitlines() if submodules else [],
            "uv_lock_sha256": _sha256(lockfile) if lockfile.is_file() else None,
        },
    }


def write_environment(path: Path, build_profile: str | None = None) -> None:
    """Atomically write one environment snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(capture_environment(build_profile), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-profile")
    arguments = parser.parse_args()
    write_environment(arguments.output, arguments.build_profile)
    print(arguments.output)


if __name__ == "__main__":
    main()
