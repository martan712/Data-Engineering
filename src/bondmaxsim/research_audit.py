"""Fail-closed research artifact and clean-clone audit commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.fixture import generate_fixture, validate_fixture
from bondmaxsim.results.catalog import validate_catalog, validate_evidence_manifest
from bondmaxsim.results.migrations import load_result


NATIVE_BINARIES = (
    Path("cpp/per_document_oracle/per_document_oracle.so"),
    Path("cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.so"),
    Path("cpp/fused_panel_maxsim/fused_panel_maxsim.so"),
)
_FIXTURE_DRIVERS = (
    "experiments.stage3_mechanism.e08_checkpoint_ablation",
    "experiments.stage4_integration.e01_fixed_candidate_arms",
    "experiments.stage5_corect.e01_ir_evaluation",
)


class ResearchAuditError(RuntimeError):
    """A publication/reproduction invariant is not satisfied."""


def _git_tracked_files(workspace: Path) -> tuple[Path, ...]:
    try:
        completed = subprocess.run(
            ("git", "ls-files", "-z"),
            cwd=workspace,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ResearchAuditError(f"cannot enumerate tracked files: {error}") from error
    return tuple(
        Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    )


def tracked_result_paths(
    workspace: Path = REPO_ROOT,
    *,
    tracked_files: Iterable[Path] | None = None,
) -> tuple[Path, ...]:
    """Discover every tracked JSON below ``results/``; never accept zero."""
    candidates = tracked_files if tracked_files is not None else _git_tracked_files(workspace)
    paths = tuple(
        sorted(
            workspace / relative
            for relative in candidates
            if relative.suffix == ".json" and relative.parts[:1] == ("results",)
        )
    )
    if not paths:
        raise ResearchAuditError("schema discovery found zero tracked result JSON files")
    missing = [str(path.relative_to(workspace)) for path in paths if not path.is_file()]
    if missing:
        raise ResearchAuditError("tracked result JSON files are missing: " + ", ".join(missing))
    return paths


def audit_result_schemas(
    workspace: Path = REPO_ROOT,
    *,
    tracked_files: Iterable[Path] | None = None,
) -> dict[str, object]:
    """Load every discovered result through the current/compatibility registry."""
    paths = tracked_result_paths(workspace, tracked_files=tracked_files)
    schemas: Counter[str] = Counter()
    failures: list[str] = []
    for path in paths:
        try:
            result = load_result(path)
        except Exception as error:  # aggregate every bad tracked artifact
            failures.append(f"{path.relative_to(workspace)}: {error}")
            continue
        historical = result.payload.get("historical_schema_name")
        schemas[str(historical or result.schema_name)] += 1
    if failures:
        raise ResearchAuditError("result schema audit failed:\n" + "\n".join(failures))
    return {
        "result_count": len(paths),
        "schemas": dict(sorted(schemas.items())),
    }


def audit_governance(workspace: Path = REPO_ROOT) -> dict[str, int]:
    """Invoke the authoritative catalog and evidence validators."""
    try:
        catalog = validate_catalog(
            workspace / "artifacts/catalog.yaml", workspace=workspace
        )
        manifest = validate_evidence_manifest(
            workspace / "artifacts/paper_evidence.yaml", catalog
        )
    except Exception as error:
        raise ResearchAuditError(f"catalog/evidence validation failed: {error}") from error
    return {
        "artifact_count": len(catalog.get("artifacts", [])),
        "evidence_count": len(manifest.get("evidence", [])),
    }


def audit_generated_paper_outputs(workspace: Path = REPO_ROOT) -> dict[str, object]:
    """Delegate stale-content detection to the canonical paper generator."""
    try:
        from bondmaxsim.render.paper.generator import check_generated_outputs
    except ImportError as error:
        raise ResearchAuditError(
            "paper generator does not expose check_generated_outputs"
        ) from error
    try:
        checked = check_generated_outputs(workspace=workspace)
    except Exception as error:
        raise ResearchAuditError(f"generated paper outputs are stale: {error}") from error
    paths = tuple(Path(path) for path in checked)
    return {
        "generated_output_count": len(paths),
        "generated_outputs": [
            str(path.relative_to(workspace) if path.is_absolute() else path)
            for path in paths
        ],
    }


def _require_native_binaries(workspace: Path) -> None:
    missing = [str(path) for path in NATIVE_BINARIES if not (workspace / path).is_file()]
    if missing:
        raise ResearchAuditError(
            "clean-clone fixture smoke requires compiled native libraries: "
            + ", ".join(missing)
        )


def clean_clone_fixture_smoke(workspace: Path = REPO_ROOT) -> dict[str, int]:
    """Exercise deterministic data plus one fixture driver from Stages 3--5."""
    _require_native_binaries(workspace)
    with tempfile.TemporaryDirectory(prefix="bondmaxsim-clean-clone-") as directory:
        output_root = Path(directory)
        fixture = output_root / "data"
        generate_fixture(fixture)
        validate_fixture(fixture)
        result_dir = output_root / "results"
        for module in _FIXTURE_DRIVERS:
            try:
                subprocess.run(
                    (
                        sys.executable,
                        "-m",
                        module,
                        "--fixture",
                        "--session-id",
                        "clean-clone-smoke",
                        "--output-dir",
                        str(result_dir),
                    ),
                    cwd=workspace,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except subprocess.CalledProcessError as error:
                detail = error.stderr.strip() or error.stdout.strip()
                raise ResearchAuditError(f"fixture driver {module} failed: {detail}") from error
        results = tuple(sorted(result_dir.glob("*.json")))
        if len(results) < len(_FIXTURE_DRIVERS):
            raise ResearchAuditError("fixture drivers did not persist all expected result artifacts")
        for path in results:
            load_result(path)
        return {"fixture_drivers": len(_FIXTURE_DRIVERS), "result_artifacts": len(results)}


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("results", "governance", "paper", "clean-clone", "full"),
    )
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT)
    arguments = parser.parse_args(argv)
    workspace = arguments.workspace.resolve()
    try:
        if arguments.command == "results":
            payload: object = audit_result_schemas(workspace)
        elif arguments.command == "governance":
            payload = audit_governance(workspace)
        elif arguments.command == "paper":
            payload = audit_generated_paper_outputs(workspace)
        elif arguments.command == "clean-clone":
            payload = clean_clone_fixture_smoke(workspace)
        else:
            payload = {
                "results": audit_result_schemas(workspace),
                "governance": audit_governance(workspace),
                "paper": audit_generated_paper_outputs(workspace),
            }
    except ResearchAuditError as error:
        parser.exit(1, f"research audit failed: {error}\n")
    _print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
