"""Frozen data-configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bondmaxsim.config import REPO_ROOT


DATASETS = ("arguana", "nfcorpus", "scidocs", "scifact")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class DataConfigurationError(ValueError):
    """The final data configuration is incomplete or ambiguous."""


@dataclass(frozen=True)
class FrozenDataConfiguration:
    path: Path
    document: Mapping[str, Any]
    sha256: str

    def dataset(self, name: str) -> Mapping[str, Any]:
        if name not in DATASETS:
            raise DataConfigurationError(f"unsupported dataset {name!r}")
        return self.document["datasets"][name]


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require(mapping: Mapping[str, Any], *names: str, context: str) -> None:
    missing = [name for name in names if mapping.get(name) in (None, "", [], {})]
    if missing:
        raise DataConfigurationError(f"{context} is missing {', '.join(missing)}")


def validate_data_configuration(document: Mapping[str, Any]) -> Mapping[str, Any]:
    if document.get("schema_name") != "bondmaxsim.data-config":
        raise DataConfigurationError("unexpected data configuration schema")
    if document.get("schema_version") != "1.0.0":
        raise DataConfigurationError("unsupported data configuration version")
    if document.get("resolution_status") != "frozen_for_regeneration":
        raise DataConfigurationError("data configuration is not frozen for regeneration")
    datasets = document.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(DATASETS):
        raise DataConfigurationError("data configuration must define exactly four datasets")
    for name in DATASETS:
        row = datasets[name]
        if not isinstance(row, Mapping):
            raise DataConfigurationError(f"datasets.{name} must be an object")
        _require(
            row,
            "corpus_queries_revision",
            "qrels_revision",
            "expected_corpus_rows",
            "expected_query_rows",
            "expected_test_qrels_queries",
            context=f"datasets.{name}",
        )
        for field in ("corpus_queries_revision", "qrels_revision"):
            if not _COMMIT.fullmatch(str(row[field])):
                raise DataConfigurationError(f"datasets.{name}.{field} is not immutable")
        if not all(
            isinstance(row[field], int) and not isinstance(row[field], bool) and row[field] > 0
            for field in (
                "expected_corpus_rows",
                "expected_query_rows",
                "expected_test_qrels_queries",
            )
        ):
            raise DataConfigurationError(f"datasets.{name} expected counts must be positive integers")
    sources = document.get("sources")
    if not isinstance(sources, Mapping):
        raise DataConfigurationError("sources must be an object")
    for component in ("model", "tokenizer"):
        row = sources.get(component)
        if not isinstance(row, Mapping) or not _COMMIT.fullmatch(str(row.get("revision", ""))):
            raise DataConfigurationError(f"sources.{component}.revision is not immutable")
    encoding = document.get("encoding")
    if not isinstance(encoding, Mapping):
        raise DataConfigurationError("encoding must be an object")
    _require(
        encoding,
        "dimension",
        "document_length",
        "query_length",
        "dtype",
        "precision",
        "query_prefix",
        "document_prefix",
        "truncation_strategy",
        "effective_tokenizer_padding",
        "special_tokens",
        "output_value",
        context="encoding",
    )
    for flag in (
        "normalize_embeddings",
        "is_query_documents",
        "is_query_queries",
        "do_query_expansion",
        "attend_to_expansion_tokens",
        "truncation",
        "padding_argument",
        "padding_tokens_retained",
        "convert_to_numpy",
    ):
        if not isinstance(encoding.get(flag), bool):
            raise DataConfigurationError(f"encoding.{flag} must be explicit boolean")
    if encoding["dimension"] != 128 or encoding["dtype"] != "float32":
        raise DataConfigurationError("final kernels require 128-dimensional float32 embeddings")
    return document


def load_data_configuration(
    path: Path = REPO_ROOT / "configs" / "data" / "v1.json",
) -> FrozenDataConfiguration:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataConfigurationError(f"cannot read data configuration {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise DataConfigurationError("data configuration root must be an object")
    validate_data_configuration(document)
    return FrozenDataConfiguration(path, document, hashlib.sha256(_canonical_bytes(document)).hexdigest())
