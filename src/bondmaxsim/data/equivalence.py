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
from bondmaxsim.data.beir_ids import load_qrels_tsv
from bondmaxsim.data.generation import validate_generated_dataset
from bondmaxsim.data.packing import build_qcum
from bondmaxsim.oracle.agreement import validate_boundary_tie_equivalence
from bondmaxsim.oracle.checkpoint_sim import simulate_fused_doc_pruning
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores


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


_DIFFERENCE_THRESHOLDS = (0.0, 1e-7, 1e-6, 1e-5, 1e-4)
_DIFFERENCE_CHUNK = 1_000_000
_QUANTILE_SAMPLE = 100_000


def _array_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    bitwise = bool(same_shape and same_dtype and np.array_equal(left, right))
    maximum = None
    mean = None
    quantiles = None
    difference_counts = None
    difference_sample_size = 0
    if same_shape and np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        left_flat = left.reshape(-1)
        right_flat = right.reshape(-1)
        maximum = 0.0
        total = 0.0
        counts = {threshold: 0 for threshold in _DIFFERENCE_THRESHOLDS}
        for begin in range(0, left_flat.size, _DIFFERENCE_CHUNK):
            end = min(begin + _DIFFERENCE_CHUNK, left_flat.size)
            difference = np.abs(
                left_flat[begin:end].astype(np.float64)
                - right_flat[begin:end].astype(np.float64)
            )
            maximum = max(maximum, float(difference.max(initial=0.0)))
            total += float(difference.sum(dtype=np.float64))
            for threshold in counts:
                counts[threshold] += int(np.count_nonzero(difference > threshold))
        mean = total / left_flat.size if left_flat.size else 0.0
        difference_counts = {
            f"gt_{threshold:.0e}": count for threshold, count in counts.items()
        }
        if left_flat.size:
            difference_sample_size = min(_QUANTILE_SAMPLE, left_flat.size)
            indices = np.linspace(
                0, left_flat.size - 1, difference_sample_size, dtype=np.int64
            )
            sample = np.abs(
                left_flat[indices].astype(np.float64)
                - right_flat[indices].astype(np.float64)
            )
            quantiles = {
                "p50": float(np.quantile(sample, 0.50)),
                "p90": float(np.quantile(sample, 0.90)),
                "p99": float(np.quantile(sample, 0.99)),
                "p99_9": float(np.quantile(sample, 0.999)),
            }
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
        "difference_counts": difference_counts,
        "difference_quantiles": quantiles,
        "difference_quantile_method": "evenly_spaced_deterministic_sample",
        "difference_quantile_sample_size": difference_sample_size,
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


def _per_query_quality(
    ranked_ids: np.ndarray,
    corpus_ids: Sequence[str],
    judgments: Mapping[str, int],
) -> tuple[float, float, float]:
    relevances = [int(judgments.get(str(corpus_ids[int(index)]), 0)) for index in ranked_ids]
    dcg = sum(
        (2.0**relevance - 1.0) / np.log2(rank + 2.0)
        for rank, relevance in enumerate(relevances[:10])
        if relevance > 0
    )
    ideal = sorted((int(value) for value in judgments.values()), reverse=True)[:10]
    idcg = sum(
        (2.0**relevance - 1.0) / np.log2(rank + 2.0)
        for rank, relevance in enumerate(ideal)
        if relevance > 0
    )
    relevant = sum(int(value) > 0 for value in judgments.values())
    retrieved = sum(relevance > 0 for relevance in relevances[:100])
    reciprocal_rank = next(
        (1.0 / (rank + 1.0) for rank, relevance in enumerate(relevances[:10]) if relevance > 0),
        0.0,
    )
    return (dcg / idcg if idcg else 0.0, retrieved / relevant if relevant else 0.0, reciprocal_rank)


def _mean_quality(rows: Sequence[tuple[float, float, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    values = np.asarray(rows, dtype=np.float64)
    return {
        "ndcg_at_10": float(values[:, 0].mean()),
        "recall_at_100": float(values[:, 1].mean()),
        "mrr_at_10": float(values[:, 2].mean()),
    }


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
    left_query_ids: Sequence[str] | None = None,
    right_query_ids: Sequence[str] | None = None,
    left_corpus_ids: Sequence[str] | None = None,
    right_corpus_ids: Sequence[str] | None = None,
    left_qrels: Mapping[str, Mapping[str, int]] | None = None,
    right_qrels: Mapping[str, Mapping[str, int]] | None = None,
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
    boundary_equal = 0
    maximum_score_difference = 0.0
    failures: list[int] = []
    boundary_failures: list[dict[str, Any]] = []
    left_quality: list[tuple[float, float, float]] = []
    right_quality: list[tuple[float, float, float]] = []
    for index, (left_query, right_query) in enumerate(
        zip(left_queries[:query_count], right_queries[:query_count], strict=True)
    ):
        actual_k = min(k, len(legacy_starts), len(generated_starts))
        left_all_scores = exact_maxsim_scores(left_query, legacy_docs, legacy_starts)
        right_all_scores = exact_maxsim_scores(right_query, generated_docs, generated_starts)
        left_ids, left_scores = topk_from_scores(left_all_scores, actual_k)
        right_ids, right_scores = topk_from_scores(right_all_scores, actual_k)
        set_equal = set(left_ids.tolist()) == set(right_ids.tolist())
        score_equal = np.array_equal(left_scores, right_scores)
        equal_sets += int(set_equal)
        equal_scores += int(score_equal)
        forward = validate_boundary_tie_equivalence(
            right_ids,
            left_ids,
            left_scores,
            k=actual_k,
            num_documents=len(legacy_starts),
            exact_scores_by_id=left_all_scores,
        )
        reverse = validate_boundary_tie_equivalence(
            left_ids,
            right_ids,
            right_scores,
            k=actual_k,
            num_documents=len(generated_starts),
            exact_scores_by_id=right_all_scores,
        )
        tie_equal = forward.exact_gate_passed and reverse.exact_gate_passed
        boundary_equal += int(tie_equal)
        if not tie_equal and len(boundary_failures) < 20:
            boundary_failures.append(
                {
                    "query_index": index,
                    "legacy_oracle_failures": list(forward.failure_codes),
                    "generated_oracle_failures": list(reverse.failure_codes),
                }
            )
        if left_scores.shape == right_scores.shape:
            maximum_score_difference = max(
                maximum_score_difference,
                float(np.max(np.abs(left_scores.astype(np.float64) - right_scores.astype(np.float64)), initial=0.0)),
            )
        if not set_equal:
            failures.append(index)
        if (
            left_qrels is not None
            and right_qrels is not None
            and left_query_ids is not None
            and right_query_ids is not None
            and left_corpus_ids is not None
            and right_corpus_ids is not None
        ):
            left_qid = str(left_query_ids[index])
            right_qid = str(right_query_ids[index])
            if left_qid in left_qrels and right_qid in right_qrels:
                left_quality.append(
                    _per_query_quality(left_ids, left_corpus_ids, left_qrels[left_qid])
                )
                right_quality.append(
                    _per_query_quality(right_ids, right_corpus_ids, right_qrels[right_qid])
                )
    return {
        "status": "complete" if limit is None or query_count == len(left_queries) else "sampled",
        "query_count": query_count,
        "topk_set_equal_count": equal_sets,
        "topk_score_bitwise_equal_count": equal_scores,
        "boundary_tie_equivalent_count": boundary_equal,
        "boundary_tie_complete": boundary_equal == query_count,
        "max_topk_score_abs_difference": maximum_score_difference,
        "first_set_mismatch_indices": failures[:20],
        "first_boundary_tie_failures": boundary_failures,
        "qrels_metrics": {
            "legacy": _mean_quality(left_quality),
            "generated": _mean_quality(right_quality),
            "evaluated_queries": len(left_quality),
        },
    }


def _built_in_accounting(
    documents: np.ndarray,
    document_starts: np.ndarray,
    queries: np.ndarray,
    query_starts: np.ndarray,
    *,
    k: int,
) -> dict[str, Any]:
    """Deterministic, bounded NumPy mechanism accounting for one query."""
    if len(document_starts) == 0 or len(query_starts) == 0:
        return {"status": "not_comparable", "reason": "empty documents or queries"}
    document_count = min(256, len(document_starts))
    token_end = (
        int(document_starts[document_count])
        if document_count < len(document_starts)
        else len(documents)
    )
    subset_documents = np.ascontiguousarray(documents[:token_end])
    subset_starts = np.ascontiguousarray(document_starts[:document_count])
    query_end = int(query_starts[1]) if len(query_starts) > 1 else len(queries)
    query = np.ascontiguousarray(queries[:query_end])
    dimension = int(query.shape[1])
    order = np.arange(dimension, dtype=np.int64)
    qcum = build_qcum(query, order)
    scores = exact_maxsim_scores(query, subset_documents, subset_starts)
    _, top_scores = topk_from_scores(scores, min(k, document_count))
    checkpoint = min(112, dimension - 1)
    result = simulate_fused_doc_pruning(
        query,
        subset_documents,
        subset_starts,
        order,
        qcum,
        (checkpoint,),
        float(top_scores[-1]) - 1e-3,
        shrink=1.0,
    )
    return {
        "status": "complete",
        "method": "numpy-fused-doc-oracle-tau",
        "query_index": 0,
        "document_count": document_count,
        "checkpoints": result.checkpoints,
        "documents_pruned": result.docs_pruned_total,
        "documents_pruned_per_checkpoint": result.docs_pruned_per_checkpoint.tolist(),
        "cells_scanned_pct": result.cells_scanned_pct,
        "cells_scanned_pct_padded": result.cells_scanned_pct_padded,
    }


def _accounting_comparison(
    legacy: Mapping[str, Any], generated: Mapping[str, Any]
) -> dict[str, Any]:
    complete = legacy.get("status") == "complete" and generated.get("status") == "complete"
    return {
        "status": "complete" if complete else "not_comparable",
        "legacy": dict(legacy),
        "generated": dict(generated),
        "equal": bool(complete and legacy == generated),
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
    fixture: bool = False,
) -> Mapping[str, Any]:
    """Compare every identity/data axis and return the predeclared decision."""
    frozen = frozen or load_data_configuration()
    validate_generated_dataset(
        generated_root, dataset, frozen=frozen, fixture=fixture
    )
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
    legacy_qrels_path = legacy_root / "qrels" / f"{dataset}.tsv"
    generated_qrels_path = generated_root / "qrels" / f"{dataset}.tsv"
    legacy_qrels = load_qrels_tsv(legacy_qrels_path)
    generated_qrels = load_qrels_tsv(generated_qrels_path)

    identity = {
        "corpus_ids_order_equal": list(legacy_ids["corpus_ids"]) == list(generated_ids["corpus_ids"]),
        "source_query_ids_order_equal": list(legacy_ids["query_ids"])
        == list(generated_ids["query_ids"]),
        "mechanism_query_ids_order_equal": list(legacy_ids["query_ids"][: len(legacy_mechanism_starts)])
        == list(generated_ids["mechanism_query_ids"]),
        "quality_query_ids_order_equal": legacy_quality_ids == generated_quality_ids,
        "qrels_checksum_equal": _sha256(legacy_qrels_path)
        == _sha256(generated_qrels_path),
        "qrels_semantically_equal": legacy_qrels == generated_qrels,
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
        "quality_token_lengths": _array_comparison(
            _lengths(legacy_quality, legacy_quality_starts),
            _lengths(generated_quality, generated_quality_starts),
        ),
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
            k=max(k, 100),
            limit=topk_limit,
            left_query_ids=legacy_quality_ids,
            right_query_ids=generated_quality_ids,
            left_corpus_ids=[str(value) for value in legacy_ids["corpus_ids"]],
            right_corpus_ids=[str(value) for value in generated_ids["corpus_ids"]],
            left_qrels=legacy_qrels,
            right_qrels=generated_qrels,
        )
    if representative_accounting is None:
        legacy_accounting = _built_in_accounting(
            legacy_docs,
            legacy_doc_starts,
            legacy_mechanism,
            legacy_mechanism_starts,
            k=k,
        )
        generated_accounting = _built_in_accounting(
            generated_docs,
            generated_doc_starts,
            generated_mechanism,
            generated_mechanism_starts,
            k=k,
        )
    else:
        legacy_accounting = {
            "status": "complete",
            "method": "external-callback",
            "values": dict(representative_accounting(legacy_root, dataset)),
        }
        generated_accounting = {
            "status": "complete",
            "method": "external-callback",
            "values": dict(representative_accounting(generated_root, dataset)),
        }
    accounting = _accounting_comparison(legacy_accounting, generated_accounting)

    qrels_metrics = ranking["quality"].get("qrels_metrics", {})
    metrics_complete = bool(
        ranking["quality"].get("status") == "complete"
        and qrels_metrics.get("evaluated_queries") == len(legacy_quality_ids)
        and qrels_metrics.get("legacy") is not None
        and qrels_metrics.get("generated") is not None
    )
    qrels_comparison = {
        "status": "complete" if metrics_complete else "not_comparable",
        "checksum_equal": identity["qrels_checksum_equal"],
        "semantic_equal": identity["qrels_semantically_equal"],
        "legacy_metrics": qrels_metrics.get("legacy"),
        "generated_metrics": qrels_metrics.get("generated"),
        "metrics_equal": bool(
            metrics_complete
            and qrels_metrics.get("legacy") == qrels_metrics.get("generated")
        ),
    }
    bitwise_equivalent = bool(
        all(identity.values())
        and all(row["bitwise_equal"] for row in arrays.values())
    )
    ranking_complete = all(
        row.get("status") == "complete" and row.get("boundary_tie_complete") is True
        for row in ranking.values()
    )
    audit_complete = bool(
        topk_limit is None
        and ranking_complete
        and qrels_comparison["status"] == "complete"
        and accounting["status"] == "complete"
    )
    cross_checks_equal = bool(
        ranking_complete
        and qrels_comparison["semantic_equal"]
        and qrels_comparison["metrics_equal"]
        and accounting["equal"]
    )
    if not bitwise_equivalent:
        decision = "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
    elif not audit_complete:
        decision = "diagnostic_incomplete_no_eligibility_decision"
    elif cross_checks_equal:
        decision = "bitwise_equivalent_dataset_audit_complete_aggregate_required"
    else:
        decision = "audit_inconsistency_stop_no_eligibility_decision"
    return {
        "schema_name": "bondmaxsim.data-equivalence",
        "schema_version": "1.0.0",
        "dataset": dataset,
        "configuration_sha256": frozen.sha256,
        "identity": identity,
        "arrays": arrays,
        "rankings": ranking,
        "qrels": qrels_comparison,
        "representative_accounting": accounting,
        "bitwise_equivalent": bitwise_equivalent,
        "audit_complete": audit_complete,
        "cross_checks_equal": cross_checks_equal,
        "final_eligibility": False,
        "decision": decision,
    }


def aggregate_equivalence_reports(
    reports: Sequence[Mapping[str, Any]], selected_datasets: Sequence[str]
) -> Mapping[str, Any]:
    """Issue final eligibility only for one complete, unsampled four-dataset audit."""
    selected = tuple(selected_datasets)
    report_datasets = tuple(str(report.get("dataset")) for report in reports)
    scope_complete = (
        len(selected) == len(DATASETS)
        and len(set(selected)) == len(DATASETS)
        and set(selected) == set(DATASETS)
        and len(report_datasets) == len(DATASETS)
        and set(report_datasets) == set(DATASETS)
    )
    audits_complete = bool(
        scope_complete and all(report.get("audit_complete") is True for report in reports)
    )
    bitwise_equivalent = bool(
        audits_complete and all(report.get("bitwise_equivalent") is True for report in reports)
    )
    cross_checks_equal = bool(
        audits_complete and all(report.get("cross_checks_equal") is True for report in reports)
    )
    final_eligibility = bitwise_equivalent and cross_checks_equal
    if not scope_complete:
        decision = "diagnostic_subset_no_eligibility_decision"
    elif any(report.get("bitwise_equivalent") is False for report in reports):
        decision = "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
    elif not audits_complete:
        decision = "diagnostic_incomplete_no_eligibility_decision"
    elif final_eligibility:
        decision = "bitwise_equivalent_non_timing_evidence_may_remain_eligible"
    else:
        decision = "not_bitwise_equivalent_rerun_all_data_dependent_evidence"
    return {
        "schema_name": "bondmaxsim.data-equivalence-suite",
        "schema_version": "1.0.0",
        "selected_datasets": list(selected),
        "scope_complete": scope_complete,
        "audits_complete": audits_complete,
        "bitwise_equivalent": bitwise_equivalent,
        "cross_checks_equal": cross_checks_equal,
        "final_eligibility": final_eligibility,
        "reports": list(reports),
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
    aggregate = aggregate_equivalence_reports(reports, selected)
    _atomic_json(arguments.output, aggregate)
    print(json.dumps({"output": str(arguments.output), "decision": aggregate["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
