"""Reproducible derived-index builds tied to frozen generated data.

FAISS and PLAID are derived artifacts: their identity includes the complete
generated-data manifest, the document embedding/ID checksums, and every
effective index setting.  Building remains resource-serial; injectable backend
builders keep the provenance and resume contract testable on tiny fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.config import DATASETS, FrozenDataConfiguration, load_data_configuration
from bondmaxsim.data.generation import validate_generated_dataset


class DerivedIndexError(RuntimeError):
    """A derived index cannot be built or validated reproducibly."""


@dataclass(frozen=True)
class IndexBuildSpec:
    backend: str
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class DerivedIndexes:
    dataset: str
    output_root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    resumed: bool


BackendBuilder = Callable[
    [Path, np.ndarray, np.ndarray, IndexBuildSpec],
    Sequence[Path],
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _faiss_n_lists(total_tokens: int) -> int:
    if total_tokens <= 0:
        raise DerivedIndexError("document token count must be positive")
    return int(2 ** np.clip(np.round(np.log2(4.0 * np.sqrt(total_tokens))), 8, 13))


def effective_index_specs(
    frozen: FrozenDataConfiguration,
    *,
    total_tokens: int,
) -> tuple[IndexBuildSpec, IndexBuildSpec]:
    """Resolve all index settings, including adapter defaults, explicitly."""
    configured = frozen.document["indexes"]
    faiss = configured["faiss"]
    plaid = configured["plaid"]
    return (
        IndexBuildSpec(
            "faiss",
            {
                "implementation": "faiss.IndexIVFFlat",
                "metric": faiss["metric"],
                "seed": int(faiss["seed"]),
                "n_lists_policy": faiss["n_lists_policy"],
                "n_lists": _faiss_n_lists(total_tokens),
                "nprobe": 32,
                "kmeans_niters": 10,
                "train_points_per_centroid": 64,
            },
        ),
        IndexBuildSpec(
            "plaid",
            {
                "implementation": plaid["implementation"],
                "seed": int(plaid["seed"]),
                "nbits": 4,
                "kmeans_niters": 4,
                "n_ivf_probe": int(plaid["n_ivf_probe"]),
                "n_full_scores": 8192,
                "num_threads": None,
                "device": "cpu",
                "use_fast": True,
            },
        ),
    )


def _build_faiss(
    destination: Path,
    doc_values: np.ndarray,
    doc_starts: np.ndarray,
    spec: IndexBuildSpec,
) -> Sequence[Path]:
    from bondmaxsim.baselines.faiss_ivf import FaissIVFBaseline

    settings = spec.settings
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    baseline = FaissIVFBaseline(
        doc_values,
        doc_starts,
        n_lists=int(settings["n_lists"]),
        nprobe=int(settings["nprobe"]),
        kmeans_niters=int(settings["kmeans_niters"]),
        train_points_per_centroid=int(settings["train_points_per_centroid"]),
        seed=int(settings["seed"]),
    )
    try:
        baseline.build(cache_path=str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return (destination,)


def _build_plaid(
    destination: Path,
    doc_values: np.ndarray,
    doc_starts: np.ndarray,
    spec: IndexBuildSpec,
) -> Sequence[Path]:
    from bondmaxsim.baselines.plaid import PLAIDBaseline

    settings = spec.settings
    destination.parent.mkdir(parents=True, exist_ok=True)
    baseline = PLAIDBaseline(
        destination.name,
        index_root=destination.parent,
        nbits=int(settings["nbits"]),
        kmeans_niters=int(settings["kmeans_niters"]),
        n_ivf_probe=int(settings["n_ivf_probe"]),
        n_full_scores=int(settings["n_full_scores"]),
        num_threads=settings["num_threads"],
        seed=int(settings["seed"]),
    )
    ends = np.append(doc_starts[1:], len(doc_values))
    documents = [doc_values[start:end] for start, end in zip(doc_starts, ends, strict=True)]
    baseline.build(documents)
    return tuple(sorted(path for path in destination.rglob("*") if path.is_file()))


DEFAULT_BUILDERS: Mapping[str, BackendBuilder] = {
    "faiss": _build_faiss,
    "plaid": _build_plaid,
}


def _source_record(
    output_root: Path,
    dataset: str,
    data_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output_root / "manifests" / f"{dataset}.json"
    outputs = {
        relative: {"sha256": record["sha256"], "bytes": int(record["bytes"])}
        for relative, record in sorted(data_manifest["outputs"].items())
    }
    return {
        "data_manifest": {
            "path": str(manifest_path.relative_to(output_root)),
            "sha256": _sha256(manifest_path),
        },
        "outputs": outputs,
    }


def _artifact_records(output_root: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    unique = sorted(set(Path(path).resolve() for path in paths))
    if not unique:
        raise DerivedIndexError("index builder produced no files")
    root = output_root.resolve()
    records = []
    for path in unique:
        if not path.is_file() or path.is_symlink():
            raise DerivedIndexError(f"index artifact is not a regular file: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise DerivedIndexError(f"index artifact escapes output root: {path}") from error
        records.append({"path": str(relative), "sha256": _sha256(path), "bytes": path.stat().st_size})
    return records


def build_derived_indexes(
    dataset: str,
    *,
    output_root: Path,
    frozen: FrozenDataConfiguration | None = None,
    builders: Mapping[str, BackendBuilder] = DEFAULT_BUILDERS,
    resume: bool = True,
) -> DerivedIndexes:
    """Build FAISS then PLAID for one validated generated dataset."""
    if dataset not in DATASETS:
        raise DerivedIndexError(f"unsupported dataset {dataset!r}")
    frozen = frozen or load_data_configuration()
    manifest_path = output_root / "index_manifests" / f"{dataset}.json"
    if resume and manifest_path.is_file():
        manifest = validate_derived_indexes(output_root, dataset, frozen=frozen)
        return DerivedIndexes(dataset, output_root, manifest_path, manifest, True)
    missing = {"faiss", "plaid"} - set(builders)
    if missing:
        raise DerivedIndexError(f"missing index builders: {', '.join(sorted(missing))}")

    data_manifest = validate_generated_dataset(output_root, dataset, frozen=frozen)
    embedding_path = output_root / "embeddings" / f"{dataset}.npz"
    with np.load(embedding_path) as arrays:
        doc_values = np.ascontiguousarray(arrays["doc_values"], dtype=np.float32)
        doc_starts = np.asarray(arrays["doc_starts"], dtype=np.int64)
    specs = effective_index_specs(frozen, total_tokens=len(doc_values))
    source = _source_record(output_root, dataset, data_manifest)
    source_key = source["outputs"][f"embeddings/{dataset}.npz"]["sha256"][:16]
    index_records: dict[str, Any] = {}
    for spec in specs:
        suffix = ".faiss" if spec.backend == "faiss" else ""
        destination = output_root / "indexes" / spec.backend / dataset / f"{source_key}{suffix}"
        paths = builders[spec.backend](destination, doc_values, doc_starts, spec)
        index_records[spec.backend] = {
            "settings": dict(spec.settings),
            "artifacts": _artifact_records(output_root, paths),
        }
    manifest = {
        "schema_name": "bondmaxsim.index-manifest",
        "schema_version": "1.0.0",
        "dataset": dataset,
        "configuration_sha256": frozen.sha256,
        "source_data": source,
        "document_shape": [int(value) for value in doc_values.shape],
        "document_count": int(len(doc_starts)),
        "indexes": index_records,
    }
    _atomic_json(manifest_path, manifest)
    validated = validate_derived_indexes(output_root, dataset, frozen=frozen)
    return DerivedIndexes(dataset, output_root, manifest_path, validated, False)


def validate_derived_indexes(
    output_root: Path,
    dataset: str,
    *,
    frozen: FrozenDataConfiguration | None = None,
) -> Mapping[str, Any]:
    frozen = frozen or load_data_configuration()
    data_manifest = validate_generated_dataset(output_root, dataset, frozen=frozen)
    manifest_path = output_root / "index_manifests" / f"{dataset}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DerivedIndexError(f"cannot read {dataset} index manifest: {error}") from error
    if (
        manifest.get("schema_name") != "bondmaxsim.index-manifest"
        or manifest.get("schema_version") != "1.0.0"
    ):
        raise DerivedIndexError(f"{dataset}: unsupported index manifest")
    if manifest.get("dataset") != dataset:
        raise DerivedIndexError(f"{dataset}: index manifest dataset mismatch")
    if manifest.get("configuration_sha256") != frozen.sha256:
        raise DerivedIndexError(f"{dataset}: index configuration hash mismatch")
    expected_source = _source_record(output_root, dataset, data_manifest)
    if manifest.get("source_data") != expected_source:
        raise DerivedIndexError(f"{dataset}: source data hash mismatch")
    expected_shape = [
        int(data_manifest["document_token_statistics"]["total_tokens"]),
        int(frozen.document["encoding"]["dimension"]),
    ]
    shape = manifest.get("document_shape")
    if shape != expected_shape:
        raise DerivedIndexError(f"{dataset}: document shape/source mismatch")
    if manifest.get("document_count") != int(data_manifest["counts"]["documents"]):
        raise DerivedIndexError(f"{dataset}: document count/source mismatch")
    expected_specs = effective_index_specs(frozen, total_tokens=int(shape[0]))
    indexes = manifest.get("indexes")
    if not isinstance(indexes, Mapping) or set(indexes) != {spec.backend for spec in expected_specs}:
        raise DerivedIndexError(f"{dataset}: incomplete index records")
    for spec in expected_specs:
        record = indexes[spec.backend]
        if record.get("settings") != dict(spec.settings):
            raise DerivedIndexError(f"{dataset}: {spec.backend} settings mismatch")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise DerivedIndexError(f"{dataset}: {spec.backend} has no artifacts")
        for artifact in artifacts:
            path = output_root / artifact["path"]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != artifact.get("bytes")
                or _sha256(path) != artifact.get("sha256")
            ):
                raise DerivedIndexError(f"{dataset}: checksum mismatch for {artifact['path']}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "generated" / "v1")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    frozen = load_data_configuration()
    for dataset in tuple(arguments.dataset or DATASETS):
        if arguments.validate_only:
            validate_derived_indexes(arguments.output_root, dataset, frozen=frozen)
            print(f"{dataset}: validated")
        else:
            result = build_derived_indexes(
                dataset,
                output_root=arguments.output_root,
                frozen=frozen,
                resume=not arguments.no_resume,
            )
            print(f"{dataset}: {'resumed' if result.resumed else 'built'} {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
