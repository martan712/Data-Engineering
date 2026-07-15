"""Write deterministic metadata for one native build product."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_line(*command: str) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.splitlines()[0]


def cpu_flags() -> list[str]:
    try:
        source = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return []
    for line in source.splitlines():
        if line.startswith("flags") and ":" in line:
            return sorted(line.split(":", 1)[1].split())
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--flags", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    payload = {
        "schema_name": "bondmaxsim.native-build-manifest",
        "schema_version": "1.0.0",
        "profile": arguments.profile,
        "compiler_path": shutil.which(arguments.compiler) or str(Path(arguments.compiler).resolve()),
        "compiler_version": first_line(arguments.compiler, "--version"),
        "flags": arguments.flags.split(),
        "machine": platform.machine(),
        "cpu_flags": cpu_flags(),
        "openmp_enabled": "-fopenmp" in arguments.flags.split(),
        "binary_name": arguments.binary.name,
        "binary_sha256": sha256(arguments.binary),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
