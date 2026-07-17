"""Shared timing and provenance helpers for controlled benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import numpy as np

INITIAL_CPU_AFFINITY = (
    sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
)

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOMP_CPU_AFFINITY",
    "RAYON_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)


@dataclass(frozen=True)
class BenchmarkObservation:
    """One benchmark result plus optional non-overlapping component timings."""

    value: Any = None
    stage_seconds: Mapping[str, float] = field(default_factory=dict)


def summarize_samples(samples: Iterable[float]) -> dict[str, float | int]:
    """Summarize raw timing samples without discarding their original values."""
    values = np.asarray(list(samples), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one timing sample is required")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("Timing samples must be finite and non-negative")

    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def run_trials(
    action: Callable[[], BenchmarkObservation],
    *,
    warmup_runs: int = 1,
    measured_runs: int = 5,
) -> tuple[Any, dict[str, Any]]:
    """Run warm-ups and measured trials around one complete online operation.

    The outer timer is authoritative for end-to-end latency. Component timings
    returned by the action are diagnostic and are stored independently.
    """
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive")

    for _ in range(warmup_runs):
        action()

    end_to_end_samples: list[float] = []
    stage_samples: dict[str, list[float]] = {}
    last_value: Any = None

    for _ in range(measured_runs):
        start = perf_counter()
        observation = action()
        elapsed = perf_counter() - start
        if not isinstance(observation, BenchmarkObservation):
            raise TypeError("Benchmark action must return BenchmarkObservation")

        end_to_end_samples.append(elapsed)
        last_value = observation.value
        for stage, seconds in observation.stage_seconds.items():
            numeric_seconds = float(seconds)
            if numeric_seconds < 0.0 or not np.isfinite(numeric_seconds):
                raise ValueError(f"Invalid timing for stage {stage!r}: {seconds}")
            stage_samples.setdefault(str(stage), []).append(numeric_seconds)

    missing_stage_samples = {
        stage: len(samples)
        for stage, samples in stage_samples.items()
        if len(samples) != measured_runs
    }
    if missing_stage_samples:
        raise ValueError(
            "Every reported stage must be present in every measured run: "
            f"{missing_stage_samples}"
        )

    timing = {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "end_to_end_seconds": end_to_end_samples,
        "end_to_end_summary": summarize_samples(end_to_end_samples),
        "stage_seconds": stage_samples,
        "stage_summaries": {
            stage: summarize_samples(samples)
            for stage, samples in sorted(stage_samples.items())
        },
    }
    return last_value, timing


def run_interleaved_trials(
    actions: Mapping[str, Callable[[], BenchmarkObservation]],
    *,
    warmup_runs: int = 1,
    measured_runs: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark complete method arms in a deterministic rotating order."""
    if not actions:
        raise ValueError("At least one benchmark action is required")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive")

    names = list(actions)
    for warmup_index in range(warmup_runs):
        shift = warmup_index % len(names)
        for name in names[shift:] + names[:shift]:
            actions[name]()

    samples = {
        name: {"end_to_end": [], "stages": {}, "last_value": None}
        for name in names
    }
    measured_order: list[list[str]] = []

    for repetition in range(measured_runs):
        shift = repetition % len(names)
        order = names[shift:] + names[:shift]
        measured_order.append(order)
        for name in order:
            start = perf_counter()
            observation = actions[name]()
            elapsed = perf_counter() - start
            if not isinstance(observation, BenchmarkObservation):
                raise TypeError("Benchmark action must return BenchmarkObservation")

            arm = samples[name]
            arm["end_to_end"].append(elapsed)
            arm["last_value"] = observation.value
            for stage, seconds in observation.stage_seconds.items():
                numeric_seconds = float(seconds)
                if numeric_seconds < 0.0 or not np.isfinite(numeric_seconds):
                    raise ValueError(f"Invalid timing for stage {stage!r}: {seconds}")
                arm["stages"].setdefault(str(stage), []).append(numeric_seconds)

    values: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    for name in names:
        arm = samples[name]
        incomplete = {
            stage: len(stage_values)
            for stage, stage_values in arm["stages"].items()
            if len(stage_values) != measured_runs
        }
        if incomplete:
            raise ValueError(
                f"Every stage for arm {name!r} must occur in every measured run: "
                f"{incomplete}"
            )

        values[name] = arm["last_value"]
        timings[name] = {
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "end_to_end_seconds": arm["end_to_end"],
            "end_to_end_summary": summarize_samples(arm["end_to_end"]),
            "stage_seconds": arm["stages"],
            "stage_summaries": {
                stage: summarize_samples(stage_values)
                for stage, stage_values in sorted(arm["stages"].items())
            },
        }

    return values, {
        "arm_order": names,
        "measured_execution_order": measured_order,
        "arms": timings,
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest for an input or result artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Hash a directory tree, including relative paths and file contents."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Expected a directory, got {root}")

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        size = file_path.stat().st_size
        digest.update(size.to_bytes(8, "little"))
        total_bytes += size
        file_count += 1
        with file_path.open("rb") as file:
            while chunk := file.read(chunk_size):
                digest.update(chunk)

    return {
        "path": str(root.resolve()),
        "files": file_count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def package_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    """Resolve selected package versions without serializing the entire venv."""
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unknown"


def collect_runtime_metadata(
    project_root: str | Path,
    *,
    input_paths: Iterable[str | Path] = (),
    packages: Iterable[str] = ("numpy", "faiss-cpu", "pylate", "pdxearch"),
) -> dict[str, Any]:
    """Capture enough provenance to interpret and reproduce one result file."""
    root = Path(project_root).resolve()
    commit = _git_value(root, "rev-parse", "HEAD")
    status = _git_value(root, "status", "--porcelain")

    inputs = {}
    for path in input_paths:
        resolved = Path(path).resolve()
        inputs[str(resolved)] = {
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": commit,
            "dirty": bool(status) if status is not None else None,
        },
        "runtime": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "initial_cpu_affinity": INITIAL_CPU_AFFINITY,
            "process_cpu_affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
            "affinity_note": (
                "initial_cpu_affinity is captured when benchmarking.py is imported; "
                "process_cpu_affinity is the calling thread at metadata capture and "
                "may be narrowed to one OpenMP place after a native kernel runs"
            ),
            "python": sys.version,
            "python_executable": sys.executable,
            "command": [sys.executable, *sys.argv],
            "working_directory": str(Path.cwd()),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
            },
            "packages": package_versions(packages),
        },
        "inputs": inputs,
    }
