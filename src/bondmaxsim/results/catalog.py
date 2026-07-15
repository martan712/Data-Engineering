"""Artifact catalog generation and relationship validation.

The catalog is deliberately stored as the JSON subset of YAML.  That keeps the
tracked governance file readable without adding another runtime dependency and
makes regeneration byte-for-byte deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from bondmaxsim.results.migrations import load_result
from bondmaxsim.results.models import ResultValidationError

CATALOG_STATUSES = {"current", "superseded", "historical", "diagnostic", "invalid"}
_ARTIFACT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POINTER = re.compile(r"^(?:/(?:[^~/]|~0|~1)*)*$")
_RESULT_GLOB = "results/json/*.json"
_VALIDATION_REPORTS = ("artifacts/baseline/agreement-cases.json",)
_REQUIRED_ENTRY_FIELDS = {
    "artifact_id",
    "path",
    "sha256",
    "schema",
    "experiment_id",
    "dataset_id",
    "workload_id",
    "method_configuration",
    "command",
    "inputs",
    "dependencies",
    "outputs",
    "consumers",
    "status",
    "field_status",
    "regeneration_cost",
}


def _read_json_subset(path: Path | str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultValidationError(f"cannot read catalog {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ResultValidationError("catalog root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_artifact_paths(workspace: Path) -> list[Path]:
    paths = list(workspace.glob(_RESULT_GLOB))
    paths.extend(workspace / relative for relative in _VALIDATION_REPORTS)
    return sorted(path for path in paths if path.is_file())


def _json_pointer_exists(document: Any, pointer: str) -> bool:
    if pointer == "":
        return True
    current = document
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                return False
            current = current[token]
        elif isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError):
                return False
        else:
            return False
    return True


def _pointer_overlaps(left: str, right: str) -> bool:
    """Whether selecting either RFC 6901 pointer also selects the other."""
    if left in {"", "/"} or right in {"", "/"}:
        return True
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def superseded_scopes(entry: Mapping[str, Any]) -> tuple[str, ...]:
    scopes: list[str] = []
    for relationship in entry.get("supersessions", []):
        if not isinstance(relationship, Mapping):
            continue
        scope = relationship.get("scope", [])
        scopes.extend([scope] if isinstance(scope, str) else scope)
    return tuple(scopes)


def pointer_status(entry: Mapping[str, Any], pointer: str) -> str:
    """Return the effective status of a catalog selection.

    Partial supersession takes precedence over the artifact-wide status.  A
    parent pointer is superseded when it includes a superseded descendant, so a
    headline consumer cannot bypass field governance by selecting ``/arms``.
    """
    if any(_pointer_overlaps(pointer, scope) for scope in superseded_scopes(entry)):
        return "superseded"
    for qualification in entry.get("field_status", []):
        if not isinstance(qualification, Mapping):
            continue
        scope = qualification.get("scope", [])
        pointers = [scope] if isinstance(scope, str) else scope
        if any(_pointer_overlaps(pointer, candidate) for candidate in pointers):
            return str(qualification.get("status", entry.get("status")))
    return str(entry.get("status"))


def _validate_string_list(entry: Mapping[str, Any], field: str, artifact_id: str) -> None:
    value = entry.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ResultValidationError(f"{artifact_id}.{field} must be a list of non-empty strings")


def _relationships(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    legacy = entry.get("supersession")
    modern = entry.get("supersessions", [])
    relationships: list[Mapping[str, Any]] = []
    if legacy:
        relationships.append(legacy)
    if modern:
        if not isinstance(modern, list):
            raise ResultValidationError("supersessions must be a list")
        relationships.extend(modern)
    return relationships


def _validate_legacy_catalog(catalog: Mapping[str, Any], workspace: Path | None) -> Mapping[str, Any]:
    """Keep the accepted Stage 3 catalog skeleton readable during migration."""
    entries = catalog.get("artifacts")
    if not isinstance(entries, list):
        raise ResultValidationError("catalog.artifacts must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    graph: dict[str, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("artifact_id"), str):
            raise ResultValidationError("catalog entries require artifact_id")
        artifact_id = entry["artifact_id"]
        if artifact_id in by_id:
            raise ResultValidationError(f"duplicate artifact_id {artifact_id}")
        if entry.get("status") not in CATALOG_STATUSES:
            raise ResultValidationError(f"invalid catalog status for {artifact_id}")
        if workspace is not None and entry.get("path") and not (workspace / entry["path"]).exists():
            raise ResultValidationError(f"catalog path does not exist: {entry['path']}")
        by_id[artifact_id] = entry
        graph[artifact_id] = []
    for artifact_id, entry in by_id.items():
        for relationship in _relationships(entry):
            successor = relationship.get("successor")
            if successor not in by_id:
                raise ResultValidationError(f"unknown successor {successor}")
            scope = relationship.get("scope")
            pointers = [scope] if isinstance(scope, str) else scope
            if not isinstance(pointers, list) or not pointers or not all(
                isinstance(pointer, str) and _POINTER.fullmatch(pointer) for pointer in pointers
            ):
                raise ResultValidationError(f"invalid supersession scope for {artifact_id}")
            graph[artifact_id].append(successor)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ResultValidationError("catalog supersession graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for successor in graph[node]:
            visit(successor)
        visiting.remove(node)
        visited.add(node)

    for artifact_id in graph:
        visit(artifact_id)
    return catalog


def validate_catalog(path: Path | str, *, workspace: Path | None = None) -> Mapping[str, Any]:
    catalog = _read_json_subset(path)
    if "catalog_version" not in catalog:
        return _validate_legacy_catalog(catalog, workspace)
    if catalog.get("catalog_version") != "1.0.0":
        raise ResultValidationError("unsupported catalog_version")
    entries = catalog.get("artifacts")
    if not isinstance(entries, list):
        raise ResultValidationError("catalog.artifacts must be a list")

    by_id: dict[str, Mapping[str, Any]] = {}
    by_path: dict[str, str] = {}
    documents: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ResultValidationError("catalog entries must be objects")
        missing = sorted(_REQUIRED_ENTRY_FIELDS - set(entry))
        if missing:
            raise ResultValidationError(f"catalog entry missing fields: {', '.join(missing)}")
        artifact_id = entry.get("artifact_id")
        if not isinstance(artifact_id, str) or not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ResultValidationError("catalog entries require a stable artifact_id")
        if artifact_id in by_id:
            raise ResultValidationError(f"duplicate artifact_id {artifact_id}")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ResultValidationError(f"invalid catalog path for {artifact_id}")
        if relative in by_path:
            raise ResultValidationError(f"duplicate catalog path {relative}")
        checksum = entry.get("sha256")
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise ResultValidationError(f"invalid checksum for {artifact_id}")
        schema = entry.get("schema")
        if not isinstance(schema, Mapping) or not all(
            isinstance(schema.get(key), str) and schema.get(key) for key in ("name", "version")
        ):
            raise ResultValidationError(f"invalid schema for {artifact_id}")
        for field in ("experiment_id", "dataset_id", "workload_id", "command", "regeneration_cost"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ResultValidationError(f"{artifact_id}.{field} must be a non-empty string")
        if not isinstance(entry.get("method_configuration"), Mapping):
            raise ResultValidationError(f"{artifact_id}.method_configuration must be an object")
        for field in ("inputs", "dependencies", "outputs", "consumers"):
            _validate_string_list(entry, field, artifact_id)
        if entry.get("status") not in CATALOG_STATUSES:
            raise ResultValidationError(f"invalid catalog status for {artifact_id}")
        if not isinstance(entry.get("field_status"), list):
            raise ResultValidationError(f"{artifact_id}.field_status must be a list")

        if workspace is not None:
            source = workspace / relative
            if not source.is_file():
                raise ResultValidationError(f"catalog path does not exist: {relative}")
            if _sha256(source) != checksum:
                raise ResultValidationError(f"catalog checksum mismatch: {relative}")
            try:
                documents[artifact_id] = json.loads(source.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ResultValidationError(f"catalog JSON is invalid: {relative}") from error
        by_id[artifact_id] = entry
        by_path[relative] = artifact_id

    if workspace is not None:
        expected = {
            str(path.relative_to(workspace)) for path in _tracked_artifact_paths(workspace)
        }
        missing = sorted(expected - set(by_path))
        extra = sorted(set(by_path) - expected)
        if missing or extra:
            raise ResultValidationError(
                f"catalog coverage mismatch; missing={missing}, extra={extra}"
            )

    graph = {artifact_id: [] for artifact_id in by_id}
    for artifact_id, entry in by_id.items():
        for dependency in entry["dependencies"]:
            if dependency not in by_id:
                raise ResultValidationError(f"unknown dependency {dependency} for {artifact_id}")
        for qualification in entry["field_status"]:
            if not isinstance(qualification, Mapping) or qualification.get("status") not in CATALOG_STATUSES:
                raise ResultValidationError(f"invalid field status for {artifact_id}")
            scope = qualification.get("scope")
            pointers = [scope] if isinstance(scope, str) else scope
            if not isinstance(pointers, list) or not pointers or not all(
                isinstance(pointer, str) and _POINTER.fullmatch(pointer) for pointer in pointers
            ):
                raise ResultValidationError(f"invalid field-status scope for {artifact_id}")
            if workspace is not None and not all(
                _json_pointer_exists(documents[artifact_id], pointer) for pointer in pointers
            ):
                raise ResultValidationError(f"unknown field-status pointer for {artifact_id}")

        for supersession in _relationships(entry):
            if not isinstance(supersession, Mapping):
                raise ResultValidationError(f"invalid supersession for {artifact_id}")
            successor = supersession.get("successor")
            if successor not in by_id:
                raise ResultValidationError(f"unknown successor {successor}")
            scope = supersession.get("scope")
            pointers = [scope] if isinstance(scope, str) else scope
            if not isinstance(pointers, list) or not pointers or not all(
                isinstance(pointer, str) and _POINTER.fullmatch(pointer) for pointer in pointers
            ):
                raise ResultValidationError(f"invalid supersession scope for {artifact_id}")
            for field in ("reason", "date"):
                if not isinstance(supersession.get(field), str) or not supersession[field]:
                    raise ResultValidationError(f"supersession {artifact_id} requires {field}")
            if workspace is not None and not all(
                _json_pointer_exists(documents[artifact_id], pointer) for pointer in pointers
            ):
                raise ResultValidationError(f"unknown supersession pointer for {artifact_id}")
            if artifact_id not in by_id[successor].get("supersedes", []):
                raise ResultValidationError(
                    f"successor {successor} lacks supersedes backlink to {artifact_id}"
                )
            graph[artifact_id].append(successor)

        for predecessor in entry.get("supersedes", []):
            if predecessor not in by_id:
                raise ResultValidationError(f"unknown predecessor {predecessor}")
            if artifact_id not in {
                relationship.get("successor") for relationship in _relationships(by_id[predecessor])
            }:
                raise ResultValidationError(
                    f"predecessor {predecessor} lacks supersession edge to {artifact_id}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ResultValidationError("catalog supersession graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for successor in graph[node]:
            visit(successor)
        visiting.remove(node)
        visited.add(node)

    for artifact_id in graph:
        visit(artifact_id)
    return catalog


def _experiment_module(stem: str) -> str:
    parts = stem.split("_")
    if stem.startswith("stage2_testbed_exact_agreement"):
        return "experiments.stage2_testbed.s02_exact_agreement"
    if stem.startswith("stage2_testbed_two_mode_smoke"):
        return "experiments.stage2_testbed.s03_two_mode_smoke"
    if stem.startswith("stage3_mechanism_r12c"):
        return "experiments.stage3_mechanism.r12c_interleaved_exact_safe"
    if stem.startswith("stage3_mechanism_"):
        return "experiments.stage3_mechanism." + "_".join(parts[2:-1])
    if stem.startswith("stage4_integration_"):
        return "experiments.stage4_integration." + "_".join(parts[2:-1])
    if stem.startswith("stage5_corect_e01"):
        return "experiments.stage5_corect.e01_ir_evaluation"
    if stem.startswith("stage5_corect_e02"):
        return "experiments.stage5_corect.e02_paired_significance"
    if stem.startswith("stage5_corect_e03"):
        return "experiments.stage5_corect.e03_bm25_baseline"
    raise ResultValidationError(f"no producing module registered for {stem}")


def _command(stem: str, dataset: str) -> str:
    module = _experiment_module(stem)
    command = f"uv run python -m {module}"
    if "two_mode_smoke" in stem:
        return command
    if stem.endswith("_1t"):
        return command + " --threads 1"
    if stem.endswith("_mt"):
        return command + " --threads 0"
    if "stage5_corect_e02" in stem:
        return command + " --input <stage5-e01-artifact>"
    if dataset not in {"multi-dataset", "unknown"}:
        command += f" --dataset {dataset}"
    return command


def _status(stem: str) -> str:
    if "two_mode_smoke" in stem or any(
        f"stage3_mechanism_e0{number}_" in stem for number in range(4, 8)
    ):
        return "diagnostic"
    if "stage3_mechanism_r12c_" in stem:
        return "current"
    return "historical"


def _result_inputs(dataset: str) -> list[str]:
    if dataset == "multi-dataset":
        return ["configs/data/v1.json", "dataset:beir-multi-dataset"]
    return ["configs/data/v1.json", f"dataset:beir-{dataset}"]


def _dependencies(stem: str) -> list[str]:
    if stem == "stage5_corect_e02_paired_significance":
        return [
            f"stage5_corect_e01_ir_evaluation_{dataset}"
            for dataset in ("arguana", "nfcorpus", "scidocs", "scifact")
        ]
    return []


def _figure_output(workspace: Path, stem: str) -> str | None:
    # Result prefixes contain two stage components (for example
    # ``stage3_mechanism``), while figure directories use that full prefix.
    stage_name = "_".join(stem.split("_")[:2])
    figure_stem = stem[len(stage_name) + 1 :]
    candidate = workspace / "results" / "figures" / stage_name / f"{figure_stem}.png"
    return str(candidate.relative_to(workspace)) if candidate.is_file() else None


def _consumers(workspace: Path, stem: str) -> list[str]:
    consumers = ["artifacts/paper_evidence.yaml"]
    if _figure_output(workspace, stem):
        stage_name = "_".join(stem.split("_")[:2])
        experiment = stem[len(stage_name) + 1 :].rsplit("_", 1)[0]
        consumers.append(f"renderer:{stage_name}.{experiment}")
    return consumers


def _timing_pointers(data: Mapping[str, Any], *, experiment: str, thread: str) -> list[str]:
    pointers: list[str] = []
    if experiment == "e08_checkpoint_ablation":
        suffix = "1t" if thread == "single" else "mt"
        for index, arm in enumerate(data.get("arms", [])):
            for metric in (f"ms_per_query_{suffix}",):
                if metric in arm:
                    pointers.append(f"/arms/{index}/{metric}")
        baseline = data.get("dense_fused_baseline", {})
        metric = f"ms_per_query_{suffix}"
        if metric in baseline:
            pointers.append(f"/dense_fused_baseline/{metric}")
    elif experiment == "e09_bound_tightness_ablation":
        for index, arm in enumerate(data.get("arms", [])):
            for metric in ("ms_per_query", "qps"):
                if metric in arm:
                    pointers.append(f"/arms/{index}/{metric}")
        for metric in ("ms_per_query", "qps"):
            if metric in data.get("dense_arm", {}):
                pointers.append(f"/dense_arm/{metric}")
    return pointers


def _partial_supersessions(data: Mapping[str, Any], experiment: str) -> list[dict[str, Any]]:
    if experiment == "e08_checkpoint_ablation":
        return [
            {
                "successor": f"stage3_mechanism_r12c_interleaved_exact_safe_{suffix}",
                "scope": _timing_pointers(data, experiment=experiment, thread=thread),
                "reason": "standalone latency was replaced by counterbalanced R12c timing; accounting fields remain diagnostic",
                "date": "2026-07-15",
            }
            for thread, suffix in (("single", "1t"), ("all_cores", "mt"))
        ]
    if experiment == "e09_bound_tightness_ablation":
        return [
            {
                "successor": "stage3_mechanism_r12c_interleaved_exact_safe_mt",
                "scope": _timing_pointers(data, experiment=experiment, thread="all_cores"),
                "reason": "standalone all-core latency was replaced by counterbalanced R12c timing; bound/pruning fields remain diagnostic",
                "date": "2026-07-15",
            }
        ]
    return []


def _field_status(data: Mapping[str, Any], experiment: str) -> list[dict[str, Any]]:
    relationships = _partial_supersessions(data, experiment)
    if not relationships:
        return []
    scopes = [pointer for relationship in relationships for pointer in relationship["scope"]]
    return [
        {"status": "superseded", "scope": scopes, "reason": "see supersession relationships"},
        {
            "status": "diagnostic",
            "scope": ["/arms"],
            "reason": "non-latency mechanism and accounting observations are retained for diagnosis",
        },
    ]


def _result_entry(workspace: Path, path: Path) -> dict[str, Any]:
    relative = str(path.relative_to(workspace))
    data = json.loads(path.read_text(encoding="utf-8"))
    normalized = load_result(path)
    provenance = normalized.provenance
    schema_name = str(provenance.get("historical_schema_name", normalized.schema_name))
    schema_version = normalized.schema_version if schema_name == normalized.schema_name else "1"
    stem = path.stem
    dataset = normalized.dataset_id
    figure = _figure_output(workspace, stem)
    outputs = [relative] + ([figure] if figure else [])
    supersessions = _partial_supersessions(data, normalized.experiment_id)
    entry: dict[str, Any] = {
        "artifact_id": stem,
        "path": relative,
        "sha256": _sha256(path),
        "schema": {"name": schema_name, "version": schema_version},
        "experiment_id": normalized.experiment_id,
        "dataset_id": dataset,
        "workload_id": normalized.workload_id,
        "method_configuration": dict(normalized.method_configuration),
        "command": _command(stem, dataset),
        "inputs": _result_inputs(dataset),
        "dependencies": _dependencies(stem),
        "outputs": outputs,
        "consumers": _consumers(workspace, stem),
        "status": _status(stem),
        "field_status": _field_status(data, normalized.experiment_id),
        "regeneration_cost": "full-dataset" if "two_mode_smoke" not in stem else "fixture",
    }
    if supersessions:
        entry["supersessions"] = supersessions
    if "stage3_mechanism_r12c_" in stem:
        predecessors = [
            path.stem
            for path in sorted((workspace / "results" / "json").glob("stage3_mechanism_e0[89]_*.json"))
            if stem.endswith("_mt") or "e08_checkpoint" in path.stem
        ]
        entry["supersedes"] = predecessors
    return entry


def _validation_entry(workspace: Path, path: Path) -> dict[str, Any]:
    relative = str(path.relative_to(workspace))
    return {
        "artifact_id": "validation.native-agreement-baseline",
        "path": relative,
        "sha256": _sha256(path),
        "schema": {"name": "bondmaxsim.validation.agreement-cases", "version": "1"},
        "experiment_id": "native-agreement-baseline",
        "dataset_id": "synthetic",
        "workload_id": "agreement-regression-cases-v1",
        "method_configuration": {"validator": "historical-exact-agreement"},
        "command": "uv run --no-sync pytest tests/test_agreement.py",
        "inputs": [],
        "dependencies": [],
        "outputs": [relative],
        "consumers": ["tests/test_agreement.py", "artifacts/paper_evidence.yaml"],
        "status": "current",
        "field_status": [],
        "regeneration_cost": "fixture",
    }


def generate_catalog(workspace: Path) -> dict[str, Any]:
    """Build the complete deterministic catalog from tracked artifact sources."""
    entries: list[dict[str, Any]] = []
    for path in _tracked_artifact_paths(workspace):
        if str(path.relative_to(workspace)) in _VALIDATION_REPORTS:
            entries.append(_validation_entry(workspace, path))
        else:
            entries.append(_result_entry(workspace, path))
    entries.sort(key=lambda entry: entry["artifact_id"])
    return {"catalog_version": "1.0.0", "artifacts": entries}


def render_status_inventory(catalog: Mapping[str, Any]) -> str:
    entries = catalog.get("artifacts", [])
    counts = Counter(entry["status"] for entry in entries)
    lines = [
        "# Artifact status inventory",
        "",
        "Generated from `artifacts/catalog.yaml`; edit the catalog generator, not this table.",
        "",
        f"Catalog version: `{catalog.get('catalog_version')}`. Tracked artifacts: **{len(entries)}**.",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status in sorted(CATALOG_STATUSES):
        lines.append(f"| {status} | {counts[status]} |")
    lines.extend(["", "## Artifacts", "", "| Artifact ID | Status | Schema | Workload | Supersession | Reproduction command |", "|---|---|---|---|---|---|"])
    for entry in entries:
        successors = ", ".join(
            sorted({relationship["successor"] for relationship in _relationships(entry)})
        ) or "—"
        schema = entry["schema"]
        lines.append(
            f"| `{entry['artifact_id']}` | {entry['status']} | "
            f"`{schema['name']}@{schema['version']}` | `{entry['workload_id']}` | "
            f"{successors} | `{entry['command']}` |"
        )
    lines.extend(
        [
            "",
            "## Supersession policy",
            "",
            "The e08/e09 standalone latency leaves are superseded by the matching R12c "
            "counterbalanced artifacts. Their non-latency pruning, bound, and cell observations "
            "remain retained as diagnostic fields. Catalog validation rejects a current evidence "
            "reference to a superseded leaf or to a parent that contains one.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_generated_catalog(workspace: Path) -> tuple[Path, Path]:
    catalog = generate_catalog(workspace)
    catalog_path = workspace / "artifacts" / "catalog.yaml"
    inventory_path = workspace / "docs" / "artifact-status.md"
    _atomic_write(catalog_path, json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    _atomic_write(inventory_path, render_status_inventory(catalog))
    validate_catalog(catalog_path, workspace=workspace)
    return catalog_path, inventory_path


def validate_evidence_manifest(
    path: Path | str, catalog: Mapping[str, Any]
) -> Mapping[str, Any]:
    manifest = _read_json_subset(path)
    evidence = manifest.get("evidence")
    if not isinstance(evidence, list):
        raise ResultValidationError("manifest.evidence must be a list")
    artifacts = {entry["artifact_id"]: entry for entry in catalog.get("artifacts", [])}
    seen: set[str] = set()
    for entry in evidence:
        evidence_id = entry.get("evidence_id") if isinstance(entry, Mapping) else None
        if not isinstance(evidence_id, str) or evidence_id in seen:
            raise ResultValidationError("evidence IDs must be unique strings")
        seen.add(evidence_id)
        references = entry.get("references")
        if not isinstance(references, list) or not references:
            raise ResultValidationError(f"evidence {evidence_id} needs references")
        for reference in references:
            artifact = artifacts.get(reference.get("artifact_id"))
            pointer = reference.get("pointer")
            if artifact is None:
                raise ResultValidationError(f"evidence {evidence_id} references unknown artifact")
            if not isinstance(pointer, str) or not _POINTER.fullmatch(pointer):
                raise ResultValidationError(f"evidence {evidence_id} has invalid JSON pointer")
            effective_status = pointer_status(artifact, pointer)
            if effective_status in {"invalid", "superseded"} and entry.get("status") == "current":
                raise ResultValidationError(
                    f"current evidence {evidence_id} references {effective_status} evidence"
                )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the artifact catalog and status inventory")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    arguments = parser.parse_args(argv)
    workspace = arguments.workspace.resolve()
    catalog = generate_catalog(workspace)
    catalog_text = json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    inventory_text = render_status_inventory(catalog)
    if arguments.check:
        expected = {
            workspace / "artifacts" / "catalog.yaml": catalog_text,
            workspace / "docs" / "artifact-status.md": inventory_text,
        }
        stale = [str(path.relative_to(workspace)) for path, text in expected.items() if not path.is_file() or path.read_text(encoding="utf-8") != text]
        if stale:
            raise ResultValidationError(f"generated catalog outputs are stale: {', '.join(stale)}")
        validate_catalog(workspace / "artifacts" / "catalog.yaml", workspace=workspace)
        return 0
    write_generated_catalog(workspace)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
