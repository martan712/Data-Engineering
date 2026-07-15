"""Pinned, atomic, dataset-serial generation of final embedding artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.config import DATASETS, FrozenDataConfiguration, load_data_configuration


_MANIFEST_SCHEMA = "bondmaxsim.data-manifest"
_MANIFEST_VERSION = "1.0.0"
_VALIDATION_CONTRACT = "strict-v2"
_MECHANISM_SELECTION_SEED = 42


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
    if values.ndim != 1 or len(values) == 0 or np.any(values <= 0):
        raise DataGenerationError("token statistics require positive token lengths")
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


def _require_unique_ids(values: Sequence[str], label: str) -> None:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise DataGenerationError(f"{label} must contain nonempty string IDs")
    if len(set(values)) != len(values):
        raise DataGenerationError(f"{label} contains duplicate IDs")


def _validate_public_source(
    dataset: str,
    source: PublicSource,
    frozen: FrozenDataConfiguration,
    *,
    fixture: bool,
) -> None:
    if len(source.document_ids) != len(source.document_texts):
        raise DataGenerationError(f"{dataset}: document ID/text count mismatch")
    if len(source.query_ids) != len(source.query_texts):
        raise DataGenerationError(f"{dataset}: query ID/text count mismatch")
    _require_unique_ids(source.document_ids, f"{dataset}.document_ids")
    _require_unique_ids(source.query_ids, f"{dataset}.query_ids")
    if any(not isinstance(text, str) for text in source.document_texts):
        raise DataGenerationError(f"{dataset}: document texts must be strings")
    if any(not isinstance(text, str) for text in source.query_texts):
        raise DataGenerationError(f"{dataset}: query texts must be strings")

    document_ids = set(source.document_ids)
    query_ids = set(source.query_ids)
    for query_id, judgments in source.qrels.items():
        if query_id not in query_ids:
            raise DataGenerationError(f"{dataset}: qrels/query ID mismatch for {query_id!r}")
        if not isinstance(judgments, Mapping) or not judgments:
            raise DataGenerationError(f"{dataset}: qrels query {query_id!r} has no judgments")
        for document_id, relevance in judgments.items():
            if document_id not in document_ids:
                raise DataGenerationError(
                    f"{dataset}: qrels/document ID mismatch for {document_id!r}"
                )
            if (
                isinstance(relevance, bool)
                or not isinstance(relevance, int)
                or relevance <= 0
            ):
                raise DataGenerationError(
                    f"{dataset}: qrels relevance must be a positive integer"
                )

    if not fixture:
        expected = frozen.dataset(dataset)
        counts = (len(source.document_ids), len(source.query_ids), len(source.qrels))
        wanted = (
            expected["expected_corpus_rows"],
            expected["expected_query_rows"],
            expected["expected_test_qrels_queries"],
        )
        if counts != wanted:
            raise DataGenerationError(f"{dataset}: pinned source counts {counts} != {wanted}")


def _mechanism_selection(
    population_ids: Sequence[str],
    population_lengths: Sequence[int],
    mechanism: Mapping[str, Any],
    *,
    fixture: bool,
) -> tuple[list[int], list[str], dict[str, Any]]:
    if mechanism.get("population_selection") != "first_200_queries_in_source_order":
        raise DataGenerationError("unsupported mechanism population selection")
    if mechanism.get("selection") != "seed42_without_replacement_sorted_source_indices":
        raise DataGenerationError("unsupported mechanism query selection")
    query_count = int(mechanism["query_count"])
    if fixture:
        query_count = min(query_count, len(population_ids))
    if len(population_ids) != len(population_lengths):
        raise DataGenerationError("mechanism population ID/token count mismatch")
    if query_count > len(population_ids):
        raise DataGenerationError("mechanism query sample exceeds its population")
    rng = np.random.default_rng(_MECHANISM_SELECTION_SEED)
    selected_indices = sorted(
        int(value)
        for value in rng.choice(len(population_ids), size=query_count, replace=False)
    )
    selected_ids = [population_ids[index] for index in selected_indices]
    selected_lengths = [population_lengths[index] for index in selected_indices]
    return selected_indices, selected_ids, _token_statistics(selected_lengths)


def _encoder_settings(frozen: FrozenDataConfiguration) -> dict[str, Any]:
    """Return the exact frozen settings executed by this generator."""
    encoding = frozen.document["encoding"]
    tokenizer = frozen.document["sources"]["tokenizer"]
    model = frozen.document["sources"]["model"]
    if tokenizer["repository"] != model["repository"] or tokenizer["revision"] != model["revision"]:
        raise DataGenerationError("model/tokenizer repositories and revisions must match")
    expected = {
        "dtype": "float32",
        "precision": "float32",
        "normalize_embeddings": True,
        "is_query_documents": False,
        "is_query_queries": True,
        "padding_argument": False,
        "padding_tokens_retained": False,
        "output_value": "token_embeddings",
        "convert_to_numpy": True,
        "pool_factor": 1,
        "protected_tokens": 1,
        "truncation_strategy": "longest_first",
        "effective_tokenizer_padding": "batch_longest",
        "document_skiplist": "ascii_punctuation",
        "query_skiplist": "none",
        "special_tokens": {
            "cls": "[CLS]",
            "sep": "[SEP]",
            "mask_and_pad": "[MASK]",
            "prefix_position": "after_cls",
        },
    }
    mismatches = {
        key: (encoding.get(key), value)
        for key, value in expected.items()
        if encoding.get(key) != value
    }
    if mismatches:
        raise DataGenerationError(f"unsupported executable encoder settings: {mismatches}")
    return {
        "model_repository": model["repository"],
        "model_revision": model["revision"],
        "tokenizer_repository": tokenizer["repository"],
        "tokenizer_revision": tokenizer["revision"],
        "tokenizer_use_fast": tokenizer["use_fast"],
        **dict(encoding),
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
        document_id = str(value["corpus-id"])
        raw_relevance = value["score"]
        if (
            isinstance(raw_relevance, bool)
            or not isinstance(raw_relevance, (int, np.integer))
            or int(raw_relevance) <= 0
        ):
            raise DataGenerationError(
                f"{dataset}: source qrels relevance must be a positive integer"
            )
        if document_id in qrels.setdefault(query_id, {}):
            raise DataGenerationError(
                f"{dataset}: duplicate qrels row for {(query_id, document_id)!r}"
            )
        qrels[query_id][document_id] = int(raw_relevance)
    source = PublicSource(
        tuple(str(value) for value in corpus["_id"]),
        tuple(str(value) for value in corpus["text"]),
        tuple(str(value) for value in queries["_id"]),
        tuple(str(value) for value in queries["text"]),
        qrels,
    )
    _validate_public_source(dataset, source, frozen, fixture=False)
    return source


def load_encoder(frozen: FrozenDataConfiguration) -> Encoder:
    from pylate import models

    _encoder_settings(frozen)
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
        skiplist_words=list(string.punctuation),
        tokenizer_kwargs={
            "use_fast": bool(frozen.document["sources"]["tokenizer"]["use_fast"])
        },
    )


def _encode(
    encoder: Encoder,
    texts: Sequence[str],
    *,
    is_query: bool,
    batch_size: int,
    encoding: Mapping[str, Any],
) -> Sequence[np.ndarray]:
    return encoder.encode(
        texts,
        is_query=is_query,
        batch_size=batch_size,
        show_progress_bar=True,
        precision=encoding["precision"],
        convert_to_numpy=encoding["convert_to_numpy"],
        padding=encoding["padding_argument"],
        normalize_embeddings=encoding["normalize_embeddings"],
        pool_factor=encoding["pool_factor"],
        protected_tokens=encoding["protected_tokens"],
        output_value=encoding["output_value"],
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


def _expected_sources(dataset: str, frozen: FrozenDataConfiguration) -> dict[str, str]:
    dataset_config = frozen.dataset(dataset)
    return {
        "corpus_queries_revision": dataset_config["corpus_queries_revision"],
        "qrels_revision": dataset_config["qrels_revision"],
        "model_revision": frozen.document["sources"]["model"]["revision"],
        "tokenizer_revision": frozen.document["sources"]["tokenizer"]["revision"],
    }


def _output_paths(output_root: Path, dataset: str) -> dict[str, Path]:
    relatives = (
        f"embeddings/{dataset}.npz",
        f"embeddings/{dataset}_test_queries.npz",
        f"beir_ids/{dataset}_ids.json",
        f"qrels/{dataset}.tsv",
    )
    return {relative: output_root / relative for relative in relatives}


def _output_records(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative, path in paths.items():
        if not path.is_file():
            raise DataGenerationError(f"missing generated output {relative}")
        records[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return records


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataGenerationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise DataGenerationError(f"{label} must be a JSON object")
    return value


def _read_qrels(
    path: Path,
    *,
    dataset: str,
    query_ids: set[str],
    document_ids: set[str],
) -> tuple[dict[str, dict[str, int]], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DataGenerationError(f"cannot read {dataset} qrels: {error}") from error
    if not lines or lines[0] != "query-id\tcorpus-id\tscore":
        raise DataGenerationError(f"{dataset}: invalid qrels header")
    qrels: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str]] = set()
    for row_number, line in enumerate(lines[1:], start=2):
        columns = line.split("\t")
        if len(columns) != 3:
            raise DataGenerationError(f"{dataset}: invalid qrels row {row_number}")
        query_id, document_id, raw_relevance = columns
        pair = (query_id, document_id)
        if not query_id or query_id not in query_ids:
            raise DataGenerationError(f"{dataset}: unknown qrels query at row {row_number}")
        if not document_id or document_id not in document_ids:
            raise DataGenerationError(f"{dataset}: unknown qrels document at row {row_number}")
        if pair in seen:
            raise DataGenerationError(f"{dataset}: duplicate qrels row for {pair!r}")
        seen.add(pair)
        try:
            relevance = int(raw_relevance)
        except ValueError:
            raise DataGenerationError(
                f"{dataset}: invalid qrels relevance at row {row_number}"
            ) from None
        if str(relevance) != raw_relevance or relevance <= 0:
            raise DataGenerationError(
                f"{dataset}: qrels relevance must be a canonical positive integer"
            )
        qrels.setdefault(query_id, {})[document_id] = relevance
    if not seen:
        raise DataGenerationError(f"{dataset}: qrels contain no judgments")
    return qrels, len(seen)


def _manifest_from_completed_files(
    output_root: Path,
    dataset: str,
    *,
    frozen: FrozenDataConfiguration,
    fixture: bool,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise DataGenerationError(f"unsupported dataset {dataset!r}")
    paths = _output_paths(output_root, dataset)
    output_records = _output_records(paths)
    ids = _read_json_object(paths[f"beir_ids/{dataset}_ids.json"], f"{dataset} IDs")
    required_id_fields = {
        "dataset",
        "source",
        "corpus_ids",
        "query_ids",
        "mechanism_query_ids",
        "quality_query_ids",
    }
    if set(ids) != required_id_fields:
        raise DataGenerationError(f"{dataset}: ID sidecar fields are not exact")
    if ids["dataset"] != dataset:
        raise DataGenerationError(f"{dataset}: ID sidecar dataset mismatch")
    expected_source = (
        f"BeIR/{dataset}@{frozen.dataset(dataset)['corpus_queries_revision']}"
    )
    if ids["source"] != expected_source:
        raise DataGenerationError(f"{dataset}: ID sidecar source revision mismatch")
    sequences: dict[str, list[str]] = {}
    for field in (
        "corpus_ids",
        "query_ids",
        "mechanism_query_ids",
        "quality_query_ids",
    ):
        value = ids[field]
        if not isinstance(value, list):
            raise DataGenerationError(f"{dataset}: {field} must be an ordered list")
        sequences[field] = value
        _require_unique_ids(value, f"{dataset}.{field}")

    corpus_ids = sequences["corpus_ids"]
    source_query_ids = sequences["query_ids"]
    mechanism_ids = sequences["mechanism_query_ids"]
    sidecar_quality_ids = sequences["quality_query_ids"]
    qrels, qrels_rows = _read_qrels(
        paths[f"qrels/{dataset}.tsv"],
        dataset=dataset,
        query_ids=set(source_query_ids),
        document_ids=set(corpus_ids),
    )

    encoding = frozen.document["encoding"]
    dimension = int(encoding["dimension"])
    embedding_path = paths[f"embeddings/{dataset}.npz"]
    try:
        with np.load(embedding_path) as values:
            if set(values.files) != {
                "doc_values",
                "doc_starts",
                "query_values",
                "query_starts",
            }:
                raise DataGenerationError(f"{dataset}: embedding NPZ fields are not exact")
            document_lengths = _validate_offsets(
                values["doc_values"], values["doc_starts"], f"{dataset}.documents", dimension
            )
            mechanism_lengths = _validate_offsets(
                values["query_values"], values["query_starts"], f"{dataset}.mechanism", dimension
            )
    except (OSError, KeyError, ValueError) as error:
        raise DataGenerationError(f"cannot read {dataset} embeddings: {error}") from error

    quality_path = paths[f"embeddings/{dataset}_test_queries.npz"]
    try:
        with np.load(quality_path) as values:
            if set(values.files) != {"query_values", "query_starts", "query_ids"}:
                raise DataGenerationError(f"{dataset}: quality NPZ fields are not exact")
            quality_lengths = _validate_offsets(
                values["query_values"], values["query_starts"], f"{dataset}.quality", dimension
            )
            raw_quality_ids = values["query_ids"]
            if raw_quality_ids.ndim != 1:
                raise DataGenerationError(f"{dataset}: quality query IDs must be a vector")
            quality_ids = [str(value) for value in raw_quality_ids]
    except (OSError, KeyError, ValueError) as error:
        raise DataGenerationError(f"cannot read {dataset} quality embeddings: {error}") from error

    _require_unique_ids(quality_ids, f"{dataset}.quality_npz_query_ids")
    mechanism = frozen.document["workloads"]["mechanism"]
    population_count = int(mechanism["population_query_count"])
    if fixture:
        population_count = min(population_count, len(source_query_ids))
    if mechanism_ids != source_query_ids[:population_count]:
        raise DataGenerationError(f"{dataset}: mechanism IDs are not the frozen source prefix")
    if len(mechanism_ids) != len(mechanism_lengths):
        raise DataGenerationError(f"{dataset}: mechanism query ID/offset count mismatch")
    if len(corpus_ids) != len(document_lengths):
        raise DataGenerationError(f"{dataset}: document ID/offset count mismatch")
    expected_quality_ids = [query_id for query_id in source_query_ids if query_id in qrels]
    if quality_ids != sidecar_quality_ids or quality_ids != expected_quality_ids:
        raise DataGenerationError(f"{dataset}: quality query ID/order mismatch")
    if len(quality_ids) != len(quality_lengths):
        raise DataGenerationError(f"{dataset}: quality query ID/offset count mismatch")

    if not fixture:
        dataset_config = frozen.dataset(dataset)
        actual_counts = (len(corpus_ids), len(source_query_ids), len(qrels))
        expected_counts = (
            dataset_config["expected_corpus_rows"],
            dataset_config["expected_query_rows"],
            dataset_config["expected_test_qrels_queries"],
        )
        if actual_counts != expected_counts:
            raise DataGenerationError(
                f"{dataset}: generated counts {actual_counts} != {expected_counts}"
            )
        if len(mechanism_ids) != int(mechanism["population_query_count"]):
            raise DataGenerationError(f"{dataset}: mechanism population count mismatch")

    selected_indices, selected_ids, selected_statistics = _mechanism_selection(
        mechanism_ids, mechanism_lengths, mechanism, fixture=fixture
    )
    quality = frozen.document["workloads"]["quality"]
    return {
        "schema_name": _MANIFEST_SCHEMA,
        "schema_version": _MANIFEST_VERSION,
        "validation_contract": _VALIDATION_CONTRACT,
        "dataset": dataset,
        "configuration_sha256": frozen.sha256,
        "sources": _expected_sources(dataset, frozen),
        "encoder_settings": _encoder_settings(frozen),
        "workloads": {
            "mechanism": {
                "workload_id": mechanism["workload_id_template"].format(dataset=dataset),
                "population_query_ids": mechanism_ids,
                "population_token_statistics": _token_statistics(mechanism_lengths),
                "selection_seed": _MECHANISM_SELECTION_SEED,
                "selected_source_indices": selected_indices,
                "selected_query_ids": selected_ids,
                "token_statistics": selected_statistics,
            },
            "quality": {
                "workload_id": quality["workload_id_template"].format(dataset=dataset),
                "query_ids": quality_ids,
                "token_statistics": _token_statistics(quality_lengths),
            },
        },
        "counts": {
            "documents": len(corpus_ids),
            "source_queries": len(source_query_ids),
            "mechanism_population_queries": len(mechanism_ids),
            "mechanism_selected_queries": len(selected_ids),
            "quality_queries": len(quality_ids),
            "qrels_queries": len(qrels),
            "qrels_rows": qrels_rows,
        },
        "document_token_statistics": _token_statistics(document_lengths),
        "outputs": output_records,
    }


def generate_dataset(
    dataset: str,
    *,
    output_root: Path,
    frozen: FrozenDataConfiguration | None = None,
    source_loader: Callable[[str, FrozenDataConfiguration], PublicSource] = load_public_source,
    encoder: Encoder | None = None,
    resume: bool = True,
    fixture: bool = False,
) -> GeneratedDataset:
    """Generate one dataset completely; callers serialize datasets themselves."""
    if dataset not in DATASETS:
        raise DataGenerationError(f"unsupported dataset {dataset!r}")
    frozen = frozen or load_data_configuration()
    _encoder_settings(frozen)
    manifest_path = output_root / "manifests" / f"{dataset}.json"
    if resume and manifest_path.is_file():
        existing = _read_json_object(manifest_path, f"{dataset} manifest")
        if existing.get("validation_contract") == _VALIDATION_CONTRACT:
            manifest = validate_generated_dataset(
                output_root, dataset, frozen=frozen, fixture=fixture
            )
        else:
            manifest = refreeze_generated_dataset(
                output_root, dataset, frozen=frozen, fixture=fixture
            )
        return GeneratedDataset(dataset, output_root, manifest_path, manifest, True)
    source = source_loader(dataset, frozen)
    _validate_public_source(dataset, source, frozen, fixture=fixture)
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
            encoding=encoding,
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
            encoding=encoding,
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
            encoding=encoding,
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
    manifest = _manifest_from_completed_files(
        output_root, dataset, frozen=frozen, fixture=fixture
    )
    _atomic_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    validated = validate_generated_dataset(
        output_root, dataset, frozen=frozen, fixture=fixture
    )
    return GeneratedDataset(dataset, output_root, manifest_path, validated, False)


def _validate_offsets(
    values: np.ndarray, starts: np.ndarray, label: str, dimension: int
) -> list[int]:
    if values.ndim != 2 or values.shape[1] != dimension or values.dtype != np.float32:
        raise DataGenerationError(
            f"{label} values must be float32 [tokens,{dimension}]"
        )
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
    fixture: bool = False,
) -> Mapping[str, Any]:
    """Recompute and compare every frozen manifest field and output checksum."""
    if dataset not in DATASETS:
        raise DataGenerationError(f"unsupported dataset {dataset!r}")
    frozen = frozen or load_data_configuration()
    manifest_path = output_root / "manifests" / f"{dataset}.json"
    manifest = _read_json_object(manifest_path, f"{dataset} manifest")
    if (
        manifest.get("schema_name") != _MANIFEST_SCHEMA
        or manifest.get("schema_version") != _MANIFEST_VERSION
        or manifest.get("validation_contract") != _VALIDATION_CONTRACT
    ):
        raise DataGenerationError(f"{dataset}: unsupported data manifest")
    expected = _manifest_from_completed_files(
        output_root, dataset, frozen=frozen, fixture=fixture
    )
    if manifest != expected:
        mismatches = sorted(
            key
            for key in set(manifest) | set(expected)
            if manifest.get(key) != expected.get(key)
        )
        raise DataGenerationError(
            f"{dataset}: manifest does not match recomputed data: {mismatches}"
        )
    return manifest


def refreeze_generated_dataset(
    output_root: Path,
    dataset: str,
    *,
    frozen: FrozenDataConfiguration | None = None,
    fixture: bool = False,
) -> Mapping[str, Any]:
    """Upgrade a completed legacy manifest from its files without encoding."""
    if dataset not in DATASETS:
        raise DataGenerationError(f"unsupported dataset {dataset!r}")
    frozen = frozen or load_data_configuration()
    manifest_path = output_root / "manifests" / f"{dataset}.json"
    legacy = _read_json_object(manifest_path, f"{dataset} manifest")
    if legacy.get("validation_contract") == _VALIDATION_CONTRACT:
        raise DataGenerationError(f"{dataset}: strict manifest must be validated, not refrozen")
    if legacy.get("schema_name") != _MANIFEST_SCHEMA or legacy.get("schema_version") != _MANIFEST_VERSION:
        raise DataGenerationError(f"{dataset}: unsupported legacy data manifest")
    if legacy.get("dataset") != dataset:
        raise DataGenerationError(f"{dataset}: legacy manifest dataset mismatch")
    if legacy.get("configuration_sha256") != frozen.sha256:
        raise DataGenerationError(f"{dataset}: legacy manifest configuration hash mismatch")
    if legacy.get("sources") != _expected_sources(dataset, frozen):
        raise DataGenerationError(f"{dataset}: legacy manifest source revisions mismatch")
    expected_paths = _output_paths(output_root, dataset)
    actual_outputs = legacy.get("outputs")
    if not isinstance(actual_outputs, Mapping) or set(actual_outputs) != set(expected_paths):
        raise DataGenerationError(f"{dataset}: legacy manifest output set mismatch")
    for relative, path in expected_paths.items():
        record = actual_outputs[relative]
        if not isinstance(record, Mapping) or set(record) != {"sha256", "bytes"}:
            raise DataGenerationError(f"{dataset}: invalid legacy output record for {relative}")
        if not path.is_file():
            raise DataGenerationError(f"{dataset}: missing legacy output {relative}")
        if (
            record["sha256"] != _sha256(path)
            or record["bytes"] != path.stat().st_size
        ):
            raise DataGenerationError(f"{dataset}: legacy checksum/size mismatch for {relative}")

    manifest = _manifest_from_completed_files(
        output_root, dataset, frozen=frozen, fixture=fixture
    )
    _atomic_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return validate_generated_dataset(
        output_root, dataset, frozen=frozen, fixture=fixture
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "generated" / "v1")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--refreeze-manifest",
        action="store_true",
        help="upgrade a completed pre-strict manifest without encoding",
    )
    arguments = parser.parse_args(argv)
    if arguments.validate_only and arguments.refreeze_manifest:
        parser.error("--validate-only and --refreeze-manifest are mutually exclusive")
    frozen = load_data_configuration()
    selected = tuple(arguments.dataset or DATASETS)
    encoder: Encoder | None = None
    for dataset in selected:
        if arguments.validate_only:
            validate_generated_dataset(arguments.output_root, dataset, frozen=frozen)
            print(f"{dataset}: valid")
        elif arguments.refreeze_manifest:
            refreeze_generated_dataset(arguments.output_root, dataset, frozen=frozen)
            print(f"{dataset}: manifest refrozen")
        else:
            manifest_path = arguments.output_root / "manifests" / f"{dataset}.json"
            if not manifest_path.is_file() and encoder is None:
                encoder = load_encoder(frozen)
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
