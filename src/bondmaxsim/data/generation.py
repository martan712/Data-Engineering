"""Pinned, atomic, dataset-serial generation of final embedding artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.config import DATASETS, FrozenDataConfiguration, load_data_configuration


class DataGenerationError(RuntimeError):
    """Pinned source generation or validation failed."""


class Encoder(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        batch_size: int,
        show_progress_bar: bool,
        **kwargs: Any,
    ) -> Sequence[np.ndarray]: ...


@dataclass(frozen=True)
class PublicSource:
    document_ids: tuple[str, ...]
    document_texts: tuple[str, ...]
    query_ids: tuple[str, ...]
    query_texts: tuple[str, ...]
    qrels: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class GeneratedDataset:
    dataset: str
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    resumed: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _flatten_embeddings(
    embeddings: Sequence[np.ndarray], *, dimension: int, label: str
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    chunks: list[np.ndarray] = []
    starts = np.empty(len(embeddings), dtype=np.int64)
    lengths: list[int] = []
    offset = 0
    for index, embedding in enumerate(embeddings):
        matrix = np.ascontiguousarray(embedding, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != dimension or matrix.shape[0] == 0:
            raise DataGenerationError(
                f"{label}[{index}] must be nonempty [tokens,{dimension}], got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise DataGenerationError(f"{label}[{index}] contains non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise DataGenerationError(f"{label}[{index}] contains non-unit token vectors")
        starts[index] = offset
        offset += len(matrix)
        lengths.append(len(matrix))
        chunks.append(matrix)
    if not chunks:
        raise DataGenerationError(f"{label} contains no embeddings")
    return np.concatenate(chunks, axis=0), starts, lengths


def _token_statistics(lengths: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "count": int(len(values)),
        "total_tokens": int(values.sum()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": int(values.min()),
        "max": int(values.max()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def load_public_source(dataset: str, frozen: FrozenDataConfiguration) -> PublicSource:
    """Load immutable BEIR rows using only revisions recorded in the config."""
    from datasets import load_dataset

    row = frozen.dataset(dataset)
    corpus = load_dataset(
        f"BeIR/{dataset}",
        "corpus",
        split="corpus",
        revision=row["corpus_queries_revision"],
    )
    queries = load_dataset(
        f"BeIR/{dataset}",
        "queries",
        split="queries",
        revision=row["corpus_queries_revision"],
    )
    qrels_rows = load_dataset(
        f"BeIR/{dataset}-qrels",
        split="test",
        revision=row["qrels_revision"],
    )
    qrels: dict[str, dict[str, int]] = {}
    for value in qrels_rows:
        query_id = str(value["query-id"])
        qrels.setdefault(query_id, {})[str(value["corpus-id"])] = int(value["score"])
    source = PublicSource(
        tuple(str(value) for value in corpus["_id"]),
        tuple(str(value) for value in corpus["text"]),
        tuple(str(value) for value in queries["_id"]),
        tuple(str(value) for value in queries["text"]),
        qrels,
    )
    expected = frozen.dataset(dataset)
    counts = (len(source.document_ids), len(source.query_ids), len(source.qrels))
    wanted = (
        expected["expected_corpus_rows"],
        expected["expected_query_rows"],
        expected["expected_test_qrels_queries"],
    )
    if counts != wanted:
        raise DataGenerationError(f"{dataset}: pinned source counts {counts} != {wanted}")
    return source


def load_encoder(frozen: FrozenDataConfiguration) -> Encoder:
    from pylate import models

    source = frozen.document["sources"]["model"]
    encoding = frozen.document["encoding"]
    return models.ColBERT(
        source["repository"],
        revision=source["revision"],
        device=encoding["device"],
        trust_remote_code=source["trust_remote_code"],
        query_prefix=encoding["query_prefix"],
        document_prefix=encoding["document_prefix"],
        query_length=encoding["query_length"],
        document_length=encoding["document_length"],
        truncation=encoding["truncation"],
        do_query_expansion=encoding["do_query_expansion"],
        attend_to_expansion_tokens=encoding["attend_to_expansion_tokens"],
    )


def _encode(
    encoder: Encoder,
    texts: Sequence[str],
    *,
    is_query: bool,
    batch_size: int,
) -> Sequence[np.ndarray]:
    return encoder.encode(
        texts,
        is_query=is_query,
        batch_size=batch_size,
        show_progress_bar=True,
        precision="float32",
        convert_to_numpy=True,
        padding=False,
        normalize_embeddings=True,
        pool_factor=1,
        protected_tokens=1,
        output_value="token_embeddings",
    )


def _qrels_bytes(qrels: Mapping[str, Mapping[str, int]]) -> bytes:
    lines = ["query-id\tcorpus-id\tscore"]
    for query_id in sorted(qrels):
        for document_id, relevance in sorted(qrels[query_id].items()):
            lines.append(f"{query_id}\t{document_id}\t{int(relevance)}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _selected_quality_queries(source: PublicSource) -> tuple[list[str], list[str]]:
    pairs = [
        (query_id, text)
        for query_id, text in zip(source.query_ids, source.query_texts, strict=True)
        if query_id in source.qrels
    ]
    if len(pairs) != len(source.qrels):
        missing = sorted(set(source.qrels) - set(source.query_ids))
        raise DataGenerationError(f"qrels query IDs missing from source queries: {missing[:3]}")
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def generate_dataset(
    dataset: str,
    *,
    output_root: Path,
    frozen: FrozenDataConfiguration | None = None,
    source_loader: Callable[[str, FrozenDataConfiguration], PublicSource] = load_public_source,
    encoder: Encoder | None = None,
    resume: bool = True,
) -> GeneratedDataset:
    """Generate one dataset completely; callers serialize datasets themselves."""
    if dataset not in DATASETS:
        raise DataGenerationError(f"unsupported dataset {dataset!r}")
    frozen = frozen or load_data_configuration()
    manifest_path = output_root / "manifests" / f"{dataset}.json"
    if resume and manifest_path.is_file():
        manifest = validate_generated_dataset(output_root, dataset, frozen=frozen)
        return GeneratedDataset(dataset, output_root, manifest_path, manifest, True)
    source = source_loader(dataset, frozen)
    if set(source.qrels) - set(source.query_ids):
        raise DataGenerationError(f"{dataset}: qrels/query ID mismatch")
    if {
        document_id for judged in source.qrels.values() for document_id in judged
    } - set(source.document_ids):
        raise DataGenerationError(f"{dataset}: qrels/document ID mismatch")
    encoder = encoder or load_encoder(frozen)
    encoding = frozen.document["encoding"]
    mechanism = frozen.document["workloads"]["mechanism"]
    population_count = int(mechanism["population_query_count"])
    mechanism_ids = list(source.query_ids[:population_count])
    mechanism_texts = list(source.query_texts[:population_count])
    quality_ids, quality_texts = _selected_quality_queries(source)

    documents, document_starts, document_lengths = _flatten_embeddings(
        _encode(
            encoder,
            source.document_texts,
            is_query=False,
            batch_size=encoding["batch_size_documents"],
        ),
        dimension=encoding["dimension"],
        label=f"{dataset}.documents",
    )
    mechanism_values, mechanism_starts, mechanism_lengths = _flatten_embeddings(
        _encode(
            encoder,
            mechanism_texts,
            is_query=True,
            batch_size=encoding["batch_size_mechanism_queries"],
        ),
        dimension=encoding["dimension"],
        label=f"{dataset}.mechanism_queries",
    )
    quality_values, quality_starts, quality_lengths = _flatten_embeddings(
        _encode(
            encoder,
            quality_texts,
            is_query=True,
            batch_size=encoding["batch_size_quality_queries"],
        ),
        dimension=encoding["dimension"],
        label=f"{dataset}.quality_queries",
    )

    embedding_path = output_root / "embeddings" / f"{dataset}.npz"
    quality_path = output_root / "embeddings" / f"{dataset}_test_queries.npz"
    ids_path = output_root / "beir_ids" / f"{dataset}_ids.json"
    qrels_path = output_root / "qrels" / f"{dataset}.tsv"
    _atomic_npz(
        embedding_path,
        doc_values=documents,
        doc_starts=document_starts,
        query_values=mechanism_values,
        query_starts=mechanism_starts,
    )
    _atomic_npz(
        quality_path,
        query_values=quality_values,
        query_starts=quality_starts,
        query_ids=np.asarray(quality_ids),
    )
    _atomic_bytes(
        ids_path,
        (json.dumps(
            {
                "dataset": dataset,
                "source": f"BeIR/{dataset}@{frozen.dataset(dataset)['corpus_queries_revision']}",
                "corpus_ids": source.document_ids,
                "query_ids": source.query_ids,
                "mechanism_query_ids": mechanism_ids,
                "quality_query_ids": quality_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n").encode("utf-8"),
    )
    _atomic_bytes(qrels_path, _qrels_bytes(source.qrels))
    output_paths = (embedding_path, quality_path, ids_path, qrels_path)
    manifest = {
        "schema_name": "bondmaxsim.data-manifest",
        "schema_version": "1.0.0",
        "dataset": dataset,
        "configuration_sha256": frozen.sha256,
        "sources": {
            "corpus_queries_revision": frozen.dataset(dataset)["corpus_queries_revision"],
            "qrels_revision": frozen.dataset(dataset)["qrels_revision"],
            "model_revision": frozen.document["sources"]["model"]["revision"],
            "tokenizer_revision": frozen.document["sources"]["tokenizer"]["revision"],
        },
        "workloads": {
            "mechanism": {
                "workload_id": mechanism["workload_id_template"].format(dataset=dataset),
                "population_query_ids": mechanism_ids,
                "token_statistics": _token_statistics(mechanism_lengths),
            },
            "quality": {
                "workload_id": frozen.document["workloads"]["quality"]["workload_id_template"].format(dataset=dataset),
                "query_ids": quality_ids,
                "token_statistics": _token_statistics(quality_lengths),
            },
        },
        "counts": {
            "documents": len(source.document_ids),
            "source_queries": len(source.query_ids),
            "mechanism_population_queries": len(mechanism_ids),
            "quality_queries": len(quality_ids),
            "qrels_queries": len(source.qrels),
        },
        "document_token_statistics": _token_statistics(document_lengths),
        "outputs": {
            str(path.relative_to(output_root)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
    }
    _atomic_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    validated = validate_generated_dataset(output_root, dataset, frozen=frozen)
    return GeneratedDataset(dataset, output_root, manifest_path, validated, False)


def _validate_offsets(values: np.ndarray, starts: np.ndarray, label: str) -> list[int]:
    if values.ndim != 2 or values.shape[1] != 128 or values.dtype != np.float32:
        raise DataGenerationError(f"{label} values must be float32 [tokens,128]")
    if starts.ndim != 1 or starts.dtype != np.int64 or len(starts) == 0:
        raise DataGenerationError(f"{label} starts must be nonempty int64 vector")
    if starts[0] != 0 or np.any(starts[1:] <= starts[:-1]) or starts[-1] >= len(values):
        raise DataGenerationError(f"{label} starts do not form a complete monotone partition")
    if not np.isfinite(values).all():
        raise DataGenerationError(f"{label} contains non-finite tokens")
    for begin in range(0, len(values), 250_000):
        norms = np.linalg.norm(values[begin : begin + 250_000], axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise DataGenerationError(f"{label} contains non-unit tokens")
    ends = np.append(starts[1:], len(values))
    return (ends - starts).astype(int).tolist()


def validate_generated_dataset(
    output_root: Path,
    dataset: str,
    *,
    frozen: FrozenDataConfiguration | None = None,
) -> Mapping[str, Any]:
    frozen = frozen or load_data_configuration()
    manifest_path = output_root / "manifests" / f"{dataset}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataGenerationError(f"cannot read {dataset} manifest: {error}") from error
    if manifest.get("schema_name") != "bondmaxsim.data-manifest" or manifest.get("schema_version") != "1.0.0":
        raise DataGenerationError(f"{dataset}: unsupported data manifest")
    if manifest.get("configuration_sha256") != frozen.sha256:
        raise DataGenerationError(f"{dataset}: configuration hash mismatch")
    for relative, record in manifest.get("outputs", {}).items():
        path = output_root / relative
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise DataGenerationError(f"{dataset}: checksum mismatch for {relative}")
    embedding_path = output_root / "embeddings" / f"{dataset}.npz"
    quality_path = output_root / "embeddings" / f"{dataset}_test_queries.npz"
    with np.load(embedding_path) as values:
        document_lengths = _validate_offsets(values["doc_values"], values["doc_starts"], f"{dataset}.documents")
        mechanism_lengths = _validate_offsets(values["query_values"], values["query_starts"], f"{dataset}.mechanism")
    with np.load(quality_path) as values:
        quality_lengths = _validate_offsets(values["query_values"], values["query_starts"], f"{dataset}.quality")
        quality_ids = [str(value) for value in values["query_ids"]]
    ids = json.loads((output_root / "beir_ids" / f"{dataset}_ids.json").read_text(encoding="utf-8"))
    qrels_ids = {
        line.split("\t", 1)[0]
        for line in (output_root / "qrels" / f"{dataset}.tsv").read_text(encoding="utf-8").splitlines()[1:]
    }
    if len(ids["corpus_ids"]) != len(document_lengths):
        raise DataGenerationError(f"{dataset}: document ID/offset count mismatch")
    if ids["mechanism_query_ids"] != manifest["workloads"]["mechanism"]["population_query_ids"] or len(ids["mechanism_query_ids"]) != len(mechanism_lengths):
        raise DataGenerationError(f"{dataset}: mechanism query ID/offset mismatch")
    if quality_ids != ids["quality_query_ids"] or len(quality_ids) != len(quality_lengths):
        raise DataGenerationError(f"{dataset}: quality query ID/offset mismatch")
    if set(quality_ids) != qrels_ids:
        raise DataGenerationError(f"{dataset}: quality query/qrels overlap is incomplete")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "generated" / "v1")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    frozen = load_data_configuration()
    selected = tuple(arguments.dataset or DATASETS)
    encoder = None if arguments.validate_only else load_encoder(frozen)
    for dataset in selected:
        if arguments.validate_only:
            validate_generated_dataset(arguments.output_root, dataset, frozen=frozen)
            print(f"{dataset}: valid")
        else:
            result = generate_dataset(
                dataset,
                output_root=arguments.output_root,
                frozen=frozen,
                encoder=encoder,
                resume=not arguments.no_resume,
            )
            print(f"{dataset}: {'resumed' if result.resumed else 'generated'} {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
