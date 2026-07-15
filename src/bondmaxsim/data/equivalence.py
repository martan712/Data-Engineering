"""Mechanical legacy-versus-regenerated data compatibility decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.config import DATASETS, FrozenDataConfiguration, load_data_configuration
from bondmaxsim.data.generation import DataGenerationError, validate_generated_dataset
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_topk


class EquivalenceError(RuntimeError):
    """The equivalence audit cannot make a complete mechanical decision."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ids(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EquivalenceError(f"cannot read ID sidecar {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise EquivalenceError(f"ID sidecar {path} is not an object")
    return value


def _lengths(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.append(starts[1:], len(values)) - starts


def _array_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    bitwise = bool(same_shape and same_dtype and np.array_equal(left, right))
    maximum = None
    mean = None
    if same_shape and np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        maximum = float(difference.max(initial=0.0))
        mean = float(difference.mean()) if difference.size else 0.0
    return {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "bitwise_equal": bitwise,
        "max_abs_difference": maximum,
        "mean_abs_difference": mean,
    }


def _unpack(values: np.ndarray, starts: np.ndarray) -> list[np.ndarray]:
    ends = np.append(starts[1:], len(values))
    return [values[int(start) : int(end)] for start, end in zip(starts, ends, strict=True)]


def _legacy_quality(legacy_root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    test_path = legacy_root / "embeddings" / f"{dataset}_test_queries.npz"
    if test_path.is_file():
        with np.load(test_path) as values:
            return (
                values["query_values"].copy(),
                values["query_starts"].copy(),
                [str(value) for value in values["query_ids"]],
            )
    with np.load(legacy_root / "embeddings" / f"{dataset}.npz") as values:
        query_values = values["query_values"].copy()
        query_starts = values["query_starts"].copy()
    sidecar = _ids(legacy_root / "beir_ids" / f"{dataset}_ids.json")
    return query_values, query_starts, [str(value) for value in sidecar["query_ids"][: len(query_starts)]]


def _topk_comparison(
    legacy_docs: np.ndarray,
    legacy_starts: np.ndarray,
    generated_docs: np.ndarray,
    generated_starts: np.ndarray,
    legacy_queries: np.ndarray,
    legacy_query_starts: np.ndarray,
    generated_queries: np.ndarray,
    generated_query_starts: np.ndarray,
    *,
    k: int,
    limit: int | None,
) -> dict[str, Any]:
    if len(legacy_starts) != len(generated_starts):
        return {"status": "not_comparable", "reason": "document count differs"}
    left_queries = _unpack(legacy_queries, legacy_query_starts)
    right_queries = _unpack(generated_queries, generated_query_starts)
    if len(left_queries) != len(right_queries):
        return {"status": "not_comparable", "reason": "query count differs"}
    query_count = len(left_queries) if limit is None else min(limit, len(left_queries))
    equal_sets = 0
    equal_scores = 0
    maximum_score_difference = 0.0
    failures: list[int] = []
    for index, (left_query, right_query) in enumerate(
        zip(left_queries[:query_count], right_queries[:query_count], strict=True)
    ):
        left_ids, left_scores = exact_maxsim_topk(
            left_query, legacy_docs, legacy_starts, min(k, len(legacy_starts))
        )
        right_ids, right_scores = exact_maxsim_topk(
            right_query, generated_docs, generated_starts, min(k, len(generated_starts))
        )
        set_equal = set(left_ids.tolist()) == set(right_ids.tolist())
        score_equal = np.array_equal(left_scores, right_scores)
        equal_sets += int(set_equal)
        equal_scores += int(score_equal)
        if left_scores.shape == right_scores.shape:
            maximum_score_difference = max(
                maximum_score_difference,
                float(np.max(np.abs(left_scores.astype(np.float64) - right_scores.astype(np.float64)), initial=0.0)),
            )
        if not set_equal:
            failures.append(index)
    return {
        "status": "complete" if limit is None or query_count == len(left_queries) else "sampled",
        "query_count": query_count,
        "topk_set_equal_count": equal_sets,
        "topk_score_bitwise_equal_count": equal_scores,
        "max_topk_score_abs_difference": maximum_score_difference,
        "first_set_mismatch_indices": failures[:20],
    }


def compare_dataset(
    dataset: str,
    *,
    legacy_root: Path,
    generated_root: Path,
    frozen: FrozenDataConfiguration | None = None,
    k: int = 10,
    topk_limit: int | None = None,
    representative_accounting: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Compare every identity/data axis and return the predeclared decision."""
    frozen = frozen or load_data_configuration()
    validate_generated_dataset(generated_root, dataset, frozen=frozen)
    legacy_ids = _ids(legacy_root / "beir_ids" / f"{dataset}_ids.json")
    generated_ids = _ids(generated_root / "beir_ids" / f"{dataset}_ids.json")
    with np.load(legacy_root / "embeddings" / f"{dataset}.npz") as values:
        legacy_docs = values["doc_values"].copy()
        legacy_doc_starts = values["doc_starts"].copy()
        legacy_mechanism = values["query_values"].copy()
        legacy_mechanism_starts = values["query_starts"].copy()
    with np.load(generated_root / "embeddings" / f"{dataset}.npz") as values:
        generated_docs = values["doc_values"].copy()
        generated_doc_starts = values["doc_starts"].copy()
        generated_mechanism = values["query_values"].copy()
        generated_mechanism_starts = values["query_starts"].copy()
    legacy_quality, legacy_quality_starts, legacy_quality_ids = _legacy_quality(
        legacy_root, dataset
    )
    with np.load(generated_root / "embeddings" / f"{dataset}_test_queries.npz") as values:
        generated_quality = values["query_values"].copy()
        generated_quality_starts = values["query_starts"].copy()
        generated_quality_ids = [str(value) for value in values["query_ids"]]

    identity = {
        "corpus_ids_order_equal": list(legacy_ids["corpus_ids"]) == list(generated_ids["corpus_ids"]),
        "mechanism_query_ids_order_equal": list(legacy_ids["query_ids"][: len(legacy_mechanism_starts)])
        == list(generated_ids["mechanism_query_ids"]),
        "quality_query_ids_order_equal": legacy_quality_ids == generated_quality_ids,
        "qrels_checksum_equal": _sha256(legacy_root / "qrels" / f"{dataset}.tsv")
        == _sha256(generated_root / "qrels" / f"{dataset}.tsv"),
    }
    arrays = {
        "documents": _array_comparison(legacy_docs, generated_docs),
        "document_starts": _array_comparison(legacy_doc_starts, generated_doc_starts),
        "document_token_lengths": _array_comparison(
            _lengths(legacy_docs, legacy_doc_starts),
            _lengths(generated_docs, generated_doc_starts),
        ),
        "mechanism_queries": _array_comparison(legacy_mechanism, generated_mechanism),
        "mechanism_query_starts": _array_comparison(legacy_mechanism_starts, generated_mechanism_starts),
        "mechanism_token_lengths": _array_comparison(
            _lengths(legacy_mechanism, legacy_mechanism_starts),
            _lengths(generated_mechanism, generated_mechanism_starts),
        ),
        "quality_queries": _array_comparison(legacy_quality, generated_quality),
        "quality_query_starts": _array_comparison(legacy_quality_starts, generated_quality_starts),
    }
    ranking = {"mechanism": {"status": "not_comparable", "reason": "identity or shape mismatch"}, "quality": {"status": "not_comparable", "reason": "identity or shape mismatch"}}
    if identity["corpus_ids_order_equal"] and identity["mechanism_query_ids_order_equal"] and arrays["document_starts"]["same_shape"] and arrays["mechanism_query_starts"]["same_shape"]:
        ranking["mechanism"] = _topk_comparison(
            legacy_docs,
            legacy_doc_starts,
            generated_docs,
            generated_doc_starts,
            legacy_mechanism,
            legacy_mechanism_starts,
            generated_mechanism,
            generated_mechanism_starts,
            k=k,
            limit=topk_limit,
        )
    if identity["corpus_ids_order_equal"] and identity["quality_query_ids_order_equal"] and arrays["document_starts"]["same_shape"] and arrays["quality_query_starts"]["same_shape"]:
        ranking["quality"] = _topk_comparison(
            legacy_docs,
            legacy_doc_starts,
            generated_docs,
            generated_doc_starts,
            legacy_quality,
            legacy_quality_starts,
            generated_quality,
            generated_quality_starts,
            k=k,
            limit=topk_limit,
        )
    accounting = None
    if representative_accounting is not None:
        accounting = {
            "legacy": dict(representative_accounting(legacy_root, dataset)),
            "generated": dict(representative_accounting(generated_root, dataset)),
        }
    bitwise_equivalent = bool(
        all(identity.values())
        and all(row["bitwise_equal"] for row in arrays.values())
    )
    decision = (
        "bitwise_equivalent_non_timing_evidence_may_remain_eligible"
        if bitwise_equivalent
        else "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
    )
    return {
        "schema_name": "bondmaxsim.data-equivalence",
        "schema_version": "1.0.0",
        "dataset": dataset,
        "configuration_sha256": frozen.sha256,
        "identity": identity,
        "arrays": arrays,
        "rankings": ranking,
        "representative_accounting": accounting,
        "bitwise_equivalent": bitwise_equivalent,
        "decision": decision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--legacy-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--generated-root", type=Path, default=REPO_ROOT / "data/generated/v1")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/generated/v1/equivalence.json")
    parser.add_argument("--topk-limit", type=int, help="diagnostic sampling only; omit for the final decision audit")
    arguments = parser.parse_args(argv)
    selected = tuple(arguments.dataset or DATASETS)
    reports = [
        compare_dataset(
            dataset,
            legacy_root=arguments.legacy_root,
            generated_root=arguments.generated_root,
            topk_limit=arguments.topk_limit,
        )
        for dataset in selected
    ]
    aggregate = {
        "schema_name": "bondmaxsim.data-equivalence-suite",
        "schema_version": "1.0.0",
        "reports": reports,
        "decision": (
            "bitwise_equivalent_non_timing_evidence_may_remain_eligible"
            if all(report["bitwise_equivalent"] for report in reports)
            else "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
        ),
    }
    _atomic_json(arguments.output, aggregate)
    print(json.dumps({"output": str(arguments.output), "decision": aggregate["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
