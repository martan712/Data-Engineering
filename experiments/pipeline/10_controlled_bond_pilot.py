"""Controlled exact-versus-BOND MaxSim decision pilot.

Both arms run in one WSL process over the same normalized packed embeddings.
The exact arm returns the full score matrix and extracts top-k; the BOND arm
returns exact-safe top-k results from a vertically laid-out document index.
Index construction and input loading are offline. The online boundary starts
with resident query embeddings and ends with ranked document IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PROJECT_ROOT, setup_imports

setup_imports()
from benchmarking import (  # noqa: E402
    BenchmarkObservation,
    collect_runtime_metadata,
    run_interleaved_trials,
)
from kernels import (  # noqa: E402
    BondMaxSimIndex,
    bond_maxsim_extension_available,
    exact_maxsim_scores_f64,
    extension_available,
)
from utils_colbert import (  # noqa: E402
    compute_qrels_metrics,
    l2_normalize,
    load_packed_embeddings,
    save_json,
)

DEFAULT_CHECKPOINTS = (8, 16, 32, 64, 96, 128)
DEFAULT_SEED_COUNTS = (10, 50, 200)
REPORT_K_VALUES = (1, 3, 5, 10)
SCORE_TIE_ATOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--embeddings-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument(
        "--seed-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEED_COUNTS),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", "1")),
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--rotation", choices=("none", "pca"), default="none")
    parser.add_argument("--pca-batch-size", type=int, default=32768)
    parser.add_argument(
        "--include-oracle-seeds",
        action="store_true",
        help="Also time BOND with the exact top-k supplied as free seed IDs",
    )
    parser.add_argument(
        "--skip-prefix-seeds",
        action="store_true",
        help="Do not add deterministic-prefix BOND arms",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_documents < 0 or args.query_start < 0 or args.max_queries < 0:
        raise SystemExit("document/query limits and query-start must be non-negative")
    if args.k < max(REPORT_K_VALUES) or args.threads <= 0:
        raise SystemExit(f"k must be at least {max(REPORT_K_VALUES)} and threads positive")
    if args.warmup_runs < 0 or args.measured_runs <= 0 or args.pca_batch_size <= 0:
        raise SystemExit("warmup-runs must be non-negative and measured-runs positive")
    if not args.checkpoints or any(value <= 0 for value in args.checkpoints):
        raise SystemExit("checkpoints must be positive")
    if not args.seed_counts or any(value <= 0 for value in args.seed_counts):
        raise SystemExit("seed-counts must be positive")


def load_inputs(
    corpus_dir: Path,
    embeddings_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    document_pack = load_packed_embeddings(embeddings_dir / "documents_packed.npz")
    query_pack = load_packed_embeddings(embeddings_dir / "queries_packed.npz")

    document_count = len(document_pack["ids"])
    if args.max_documents:
        document_count = min(document_count, args.max_documents)
    if document_count < args.k:
        raise SystemExit(f"Need at least k={args.k} documents, found {document_count}")

    available_queries = len(query_pack["ids"])
    if args.query_start >= available_queries:
        raise SystemExit(
            f"query-start={args.query_start} exceeds available queries={available_queries}"
        )
    query_stop = available_queries
    if args.max_queries:
        query_stop = min(query_stop, args.query_start + args.max_queries)
    if query_stop == args.query_start:
        raise SystemExit("No queries selected")

    document_offsets = np.ascontiguousarray(
        document_pack["offsets"][: document_count + 1],
        dtype=np.int64,
    )
    document_token_count = int(document_offsets[-1])
    document_values = l2_normalize(document_pack["values"][:document_token_count])

    source_query_offsets = query_pack["offsets"][args.query_start : query_stop + 1]
    query_token_start = int(source_query_offsets[0])
    query_token_stop = int(source_query_offsets[-1])
    query_offsets = np.ascontiguousarray(
        source_query_offsets - query_token_start,
        dtype=np.int64,
    )
    query_values = l2_normalize(
        query_pack["values"][query_token_start:query_token_stop]
    )

    document_ids = document_pack["ids"][:document_count]
    query_ids = query_pack["ids"][args.query_start:query_stop]
    with (corpus_dir / "qrels.json").open("r", encoding="utf-8") as file:
        raw_qrels = json.load(file)
    retained_documents = set(document_ids)
    qrels = {
        query_id: [
            document_id
            for document_id in raw_qrels.get(query_id, [])
            if document_id in retained_documents
        ]
        for query_id in query_ids
    }

    return {
        "document_values": document_values,
        "document_offsets": document_offsets,
        "query_values": query_values,
        "query_offsets": query_offsets,
        "document_ids": document_ids,
        "query_ids": query_ids,
        "qrels": qrels,
    }


def rank_score_matrix(scores: np.ndarray, k: int) -> np.ndarray:
    """Rank each score row by descending score and ascending document index."""
    document_indices = np.arange(scores.shape[1], dtype=np.int64)
    rankings = np.empty((scores.shape[0], k), dtype=np.int64)
    for query_index, row in enumerate(scores):
        rankings[query_index] = document_indices[
            np.lexsort((document_indices, -row))[:k]
        ]
    return rankings


def apply_pca_rotation(inputs: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Apply deterministic uncentered PCA as offline index preprocessing."""
    documents = inputs["document_values"]
    queries = inputs["query_values"]
    dimension = documents.shape[1]

    start = perf_counter()
    second_moment = np.zeros((dimension, dimension), dtype=np.float64)
    for batch_start in range(0, documents.shape[0], batch_size):
        block = documents[batch_start : batch_start + batch_size].astype(
            np.float64,
            copy=False,
        )
        second_moment += block.T @ block
    eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    rotation64 = np.ascontiguousarray(eigenvectors[:, order], dtype=np.float64)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    rotation = np.ascontiguousarray(rotation64, dtype=np.float32)
    rotated_documents = np.empty_like(documents)
    for batch_start in range(0, documents.shape[0], batch_size):
        batch_stop = min(batch_start + batch_size, documents.shape[0])
        np.matmul(
            documents[batch_start:batch_stop],
            rotation,
            out=rotated_documents[batch_start:batch_stop],
        )
    rotated_queries = np.ascontiguousarray(queries @ rotation, dtype=np.float32)
    transform_seconds = perf_counter() - start

    energy = np.clip(eigenvalues, 0.0, None)
    total_energy = float(energy.sum())
    energy_fractions = {
        str(checkpoint): float(energy[:checkpoint].sum() / total_energy)
        for checkpoint in (8, 16, 32, 64, 96, dimension)
        if checkpoint <= dimension and total_energy > 0.0
    }
    orthogonality_error = float(
        np.max(
            np.abs(
                rotation.astype(np.float64).T @ rotation.astype(np.float64)
                - np.eye(dimension, dtype=np.float64)
            )
        )
    )
    inputs["document_values"] = rotated_documents
    inputs["query_values"] = rotated_queries
    return {
        "rotation": "uncentered_pca_descending_second_moment",
        "fit_seconds": fit_seconds,
        "transform_seconds": transform_seconds,
        "batch_size": batch_size,
        "rotation_sha256": hashlib.sha256(rotation.tobytes(order="C")).hexdigest(),
        "float32_orthogonality_max_abs_error": orthogonality_error,
        "cumulative_energy_fraction": energy_fractions,
    }


def ranking_ids(
    rankings: np.ndarray,
    document_ids: list[str],
    query_ids: list[str],
) -> dict[str, list[str]]:
    return {
        query_id: [document_ids[int(index)] for index in row]
        for query_id, row in zip(query_ids, rankings)
    }


def exact_action(inputs: dict[str, Any], k: int) -> Callable[[], BenchmarkObservation]:
    def run() -> BenchmarkObservation:
        start = perf_counter()
        scores = exact_maxsim_scores_f64(
            inputs["document_values"],
            inputs["document_offsets"],
            inputs["query_values"],
            inputs["query_offsets"],
        )
        score_seconds = perf_counter() - start
        start = perf_counter()
        rankings = rank_score_matrix(scores, k)
        topk_seconds = perf_counter() - start
        return BenchmarkObservation(
            value={"rankings": rankings, "scores": scores},
            stage_seconds={"exact_score": score_seconds, "topk": topk_seconds},
        )

    return run


def bond_action(
    index: BondMaxSimIndex,
    inputs: dict[str, Any],
    k: int,
    seed_count: int,
) -> Callable[[], BenchmarkObservation]:
    def run() -> BenchmarkObservation:
        start = perf_counter()
        native_result = index.search(
            inputs["query_values"],
            inputs["query_offsets"],
            k,
            seed_count,
        )
        search_seconds = perf_counter() - start
        return BenchmarkObservation(
            value=native_result,
            stage_seconds={"bond_search_topk": search_seconds},
        )

    return run


def bond_explicit_seed_action(
    index: BondMaxSimIndex,
    inputs: dict[str, Any],
    k: int,
    seed_ids: np.ndarray,
) -> Callable[[], BenchmarkObservation]:
    def run() -> BenchmarkObservation:
        start = perf_counter()
        native_result = index.search_with_seed_ids(
            inputs["query_values"],
            inputs["query_offsets"],
            k,
            seed_ids,
        )
        search_seconds = perf_counter() - start
        return BenchmarkObservation(
            value=native_result,
            stage_seconds={"bond_search_topk": search_seconds},
        )

    return run


def exact_recovery(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for k in REPORT_K_VALUES:
        if k > reference.shape[1]:
            continue
        recalls = [
            len(set(left[:k].tolist()) & set(right[:k].tolist())) / k
            for left, right in zip(reference, candidate)
        ]
        output[f"mean_recall@{k}"] = float(np.mean(recalls))
        output[f"all_top{k}_recovered_queries"] = int(
            sum(value == 1.0 for value in recalls)
        )
    output["rankings_identical"] = bool(np.array_equal(reference, candidate))
    output["topk_sets_identical"] = bool(
        all(set(left.tolist()) == set(right.tolist()) for left, right in zip(reference, candidate))
    )
    return output


def score_validation(
    exact_scores: np.ndarray,
    rankings: np.ndarray,
    bond_scores: np.ndarray,
) -> dict[str, float]:
    selected_exact = np.take_along_axis(exact_scores, rankings, axis=1)
    absolute = np.abs(selected_exact.astype(np.float64) - bond_scores)
    return {
        "max_abs_error": float(np.max(absolute)),
        "mean_abs_error": float(np.mean(absolute)),
    }


def mismatch_diagnostic(
    exact_rankings: np.ndarray,
    exact_scores: np.ndarray,
    bond_ids: np.ndarray,
    bond_scores: np.ndarray,
) -> dict[str, Any]:
    """Describe the first numerical/ranking disagreements before aborting."""
    rows = np.flatnonzero(np.any(exact_rankings != bond_ids, axis=1))
    details = []
    for row_index in rows[:5]:
        exact_row = exact_rankings[row_index]
        bond_row = bond_ids[row_index]
        union = np.unique(np.concatenate((exact_row, bond_row)))
        details.append(
            {
                "query_row": int(row_index),
                "exact_ids": exact_row.astype(int).tolist(),
                "bond_ids": bond_row.astype(int).tolist(),
                "float_exact_scores_on_union": {
                    str(int(document_id)): float(exact_scores[row_index, document_id])
                    for document_id in union
                },
                "bond_topk_scores": bond_scores[row_index].astype(float).tolist(),
            }
        )
    return {"mismatched_query_rows": rows.astype(int).tolist(), "details": details}


def numerical_order_validation(
    exact_rankings: np.ndarray,
    exact_scores: np.ndarray,
    bond_ids: np.ndarray,
    *,
    tie_atol: float = SCORE_TIE_ATOL,
) -> dict[str, Any]:
    """Accept only set-identical order changes among float-score near-ties."""
    mismatched_rows = np.flatnonzero(np.any(exact_rankings != bond_ids, axis=1))
    non_tie_inversions: list[dict[str, Any]] = []
    sets_identical = True
    for row_index in mismatched_rows:
        exact_row = exact_rankings[row_index].tolist()
        bond_row = bond_ids[row_index].tolist()
        if set(exact_row) != set(bond_row):
            sets_identical = False
            continue
        exact_positions = {document_id: rank for rank, document_id in enumerate(exact_row)}
        bond_positions = {document_id: rank for rank, document_id in enumerate(bond_row)}
        for left_index, left_id in enumerate(exact_row):
            for right_id in exact_row[left_index + 1 :]:
                if bond_positions[left_id] < bond_positions[right_id]:
                    continue
                score_gap = abs(
                    float(exact_scores[row_index, left_id])
                    - float(exact_scores[row_index, right_id])
                )
                if score_gap > tie_atol:
                    non_tie_inversions.append(
                        {
                            "query_row": int(row_index),
                            "left_id": int(left_id),
                            "right_id": int(right_id),
                            "float_score_gap": score_gap,
                        }
                    )
    return {
        "score_tie_atol": tie_atol,
        "rankings_identical": bool(mismatched_rows.size == 0),
        "topk_sets_identical": sets_identical,
        "tie_equivalent_order": sets_identical and not non_tie_inversions,
        "mismatched_query_rows": mismatched_rows.astype(int).tolist(),
        "non_tie_inversions": non_tie_inversions,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not extension_available():
        raise SystemExit("Build cpp/exact_maxsim/build_wsl.sh before this runner")
    if not bond_maxsim_extension_available():
        raise SystemExit("Build cpp/bond_maxsim/build_wsl.sh before this runner")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    corpus_dir = resolve(args.corpus_dir)
    embeddings_dir = resolve(args.embeddings_dir)
    output_path = resolve(args.output)
    inputs = load_inputs(corpus_dir, embeddings_dir, args)

    preprocessing: dict[str, Any] = {"rotation": "none"}
    if args.rotation == "pca":
        print("Fitting and applying offline uncentered PCA rotation...")
        preprocessing = apply_pca_rotation(inputs, args.pca_batch_size)

    dimension = int(inputs["document_values"].shape[1])
    checkpoints = sorted(set(args.checkpoints))
    if checkpoints[-1] != dimension or any(value > dimension for value in checkpoints):
        raise SystemExit(
            f"checkpoints must be increasing and end at embedding dimension {dimension}"
        )
    document_count = len(inputs["document_ids"])
    seed_counts = [] if args.skip_prefix_seeds else sorted(
        set(value for value in args.seed_counts if args.k <= value <= document_count)
    )
    if not seed_counts and not args.include_oracle_seeds:
        raise SystemExit("No seed-count lies in the valid range [k, documents]")

    checkpoint_array = np.ascontiguousarray(checkpoints, dtype=np.int64)
    print(
        f"documents={document_count} queries={len(inputs['query_ids'])} "
        f"doc_tokens={inputs['document_values'].shape[0]} dim={dimension} "
        f"checkpoints={checkpoints} seeds={seed_counts}"
    )
    print("Building offline vertical BOND-MaxSim index...")
    start = perf_counter()
    bond_index = BondMaxSimIndex(
        inputs["document_values"],
        inputs["document_offsets"],
        checkpoint_array,
    )
    index_build_seconds = perf_counter() - start

    actions: dict[str, Callable[[], BenchmarkObservation]] = {
        "compiled_exact": exact_action(inputs, args.k)
    }
    bond_configs: dict[str, dict[str, Any]] = {}
    for seed_count in seed_counts:
        name = f"bond_seed{seed_count}"
        actions[name] = bond_action(
            bond_index,
            inputs,
            args.k,
            seed_count,
        )
        bond_configs[name] = {
            "seed_count": seed_count,
            "seed_policy": "deterministic_document_prefix",
            "implementable_without_external_candidates": True,
        }
    if args.include_oracle_seeds:
        reference = exact_action(inputs, args.k)().value
        oracle_seed_ids = np.ascontiguousarray(
            reference["rankings"],
            dtype=np.int64,
        )
        name = "bond_oracle_topk_seeds"
        actions[name] = bond_explicit_seed_action(
            bond_index,
            inputs,
            args.k,
            oracle_seed_ids,
        )
        bond_configs[name] = {
            "seed_count": args.k,
            "seed_policy": "free_exact_topk_oracle",
            "implementable_without_external_candidates": False,
        }

    # Correctness is a hard gate before any timed comparison is accepted.
    validation_values, _ = run_interleaved_trials(
        actions,
        warmup_runs=0,
        measured_runs=1,
    )
    exact_rankings = validation_values["compiled_exact"]["rankings"]
    exact_scores = validation_values["compiled_exact"]["scores"]
    for name in bond_configs:
        order_validation = numerical_order_validation(
            exact_rankings,
            exact_scores,
            validation_values[name]["ids"],
        )
        if not order_validation["tie_equivalent_order"]:
            diagnostic = mismatch_diagnostic(
                exact_rankings,
                exact_scores,
                validation_values[name]["ids"],
                validation_values[name]["scores"],
            )
            raise SystemExit(
                f"Correctness gate failed: {name} differs beyond float-score ties: "
                f"{json.dumps(diagnostic, sort_keys=True)}"
            )
        validation_error = score_validation(
            exact_scores,
            exact_rankings,
            validation_values[name]["scores"],
        )["max_abs_error"]
        if validation_error > 1e-4:
            raise SystemExit(
                f"Correctness gate failed: {name} max score error={validation_error}"
            )

    print(
        f"Running {args.warmup_runs} warm-up and {args.measured_runs} measured "
        "interleaved rounds..."
    )
    values, timing = run_interleaved_trials(
        actions,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )
    exact_rankings = values["compiled_exact"]["rankings"]
    exact_scores = values["compiled_exact"]["scores"]
    exact_ranking_ids = ranking_ids(
        exact_rankings,
        inputs["document_ids"],
        inputs["query_ids"],
    )
    exact_qrels = compute_qrels_metrics(
        exact_ranking_ids,
        inputs["qrels"],
        k_values=REPORT_K_VALUES,
        mrr_k=10,
    )["macro"]

    bond_results: dict[str, Any] = {}
    total_query_documents = len(inputs["query_ids"]) * document_count
    for name, bond_config in bond_configs.items():
        value = values[name]
        rankings = value["ids"]
        ids = ranking_ids(rankings, inputs["document_ids"], inputs["query_ids"])
        recovery = exact_recovery(exact_rankings, rankings)
        order_validation = numerical_order_validation(
            exact_rankings,
            exact_scores,
            rankings,
        )
        if not order_validation["tie_equivalent_order"]:
            raise SystemExit(f"Measured correctness gate failed beyond ties: {name}")
        bond_results[name] = {
            **bond_config,
            "timing": timing["arms"][name],
            "exact_recovery": recovery,
            "numerical_order_validation": order_validation,
            "score_validation": score_validation(
                exact_scores,
                rankings,
                value["scores"],
            ),
            "qrels_metrics": compute_qrels_metrics(
                ids,
                inputs["qrels"],
                k_values=REPORT_K_VALUES,
                mrr_k=10,
            )["macro"],
            "work": {
                "documents_pruned": int(value["documents_pruned"]),
                "documents_exactly_scored": int(value["documents_exactly_scored"]),
                "total_query_document_pairs": total_query_documents,
                "pruned_document_fraction": (
                    int(value["documents_pruned"]) / total_query_documents
                ),
                "component_products": int(value["component_products"]),
                "full_component_products": int(value["full_component_products"]),
                "component_product_ratio": float(value["work_ratio"]),
                "pruned_by_checkpoint": {
                    str(checkpoint): int(count)
                    for checkpoint, count in zip(
                        checkpoints,
                        value["pruned_by_checkpoint"].tolist(),
                    )
                },
            },
            "topk_rankings": ids,
        }

    result = {
        "schema_version": "controlled_bond_pilot_v1",
        "status": "pilot_not_final_claim",
        "timing_boundary": (
            "resident normalized packed query token vectors to exact top-k document IDs; "
            "BOND query preparation, bounds, pruning, scoring, and top-k are included"
        ),
        "metadata": collect_runtime_metadata(
            PROJECT_ROOT,
            input_paths=(
                corpus_dir / "qrels.json",
                embeddings_dir / "documents_packed.npz",
                embeddings_dir / "queries_packed.npz",
                PROJECT_ROOT / "cpp" / "exact_maxsim" / "exact_maxsim.cpp",
                PROJECT_ROOT / "cpp" / "bond_maxsim" / "bond_maxsim.cpp",
            ),
        ),
        "dataset": {
            "corpus_dir": str(corpus_dir),
            "embeddings_dir": str(embeddings_dir),
            "query_start": args.query_start,
            "documents": document_count,
            "queries": len(inputs["query_ids"]),
            "document_token_vectors": int(inputs["document_values"].shape[0]),
            "query_token_vectors": int(inputs["query_values"].shape[0]),
            "embedding_dimension": dimension,
            "retained_qrels_labels": int(sum(map(len, inputs["qrels"].values()))),
        },
        "config": {
            "k": args.k,
            "checkpoints": checkpoints,
            "seed_counts": seed_counts,
            "include_oracle_seeds": args.include_oracle_seeds,
            "skip_prefix_seeds": args.skip_prefix_seeds,
            "threads": args.threads,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "metric": "inner product on independently L2-normalized token vectors",
            "dimension_order": (
                "natural model order"
                if args.rotation == "none"
                else "uncentered PCA eigenvectors by descending eigenvalue"
            ),
            "threshold": "kth-largest conservative lower bound from exactly scored docs",
            "score_tie_atol": SCORE_TIE_ATOL,
            "exact_accumulation": "float64 products and accumulation",
        },
        "offline_index_build": {
            "preprocessing": preprocessing,
            "bond_vertical_index_seconds": index_build_seconds,
            "estimated_vertical_values_bytes": int(inputs["document_values"].nbytes),
            "estimated_residual_norm_bytes": int(
                inputs["document_values"].shape[0] * len(checkpoints) * 8
            ),
        },
        "interleaving": {
            "arm_order": timing["arm_order"],
            "measured_execution_order": timing["measured_execution_order"],
        },
        "exact_reference": {
            "timing": timing["arms"]["compiled_exact"],
            "qrels_metrics": exact_qrels,
            "topk_rankings": exact_ranking_ids,
        },
        "bond_arms": bond_results,
    }
    save_json(output_path, result)

    exact_median = timing["arms"]["compiled_exact"]["end_to_end_summary"]["median"]
    print(f"compiled_exact median={exact_median:.6f}s")
    for name, arm in bond_results.items():
        median = arm["timing"]["end_to_end_summary"]["median"]
        work = arm["work"]
        print(
            f"{name:>18}: median={median:.6f}s "
            f"pruned={work['pruned_document_fraction']:.3f} "
            f"component_ratio={work['component_product_ratio']:.3f} "
            f"exact_set={arm['exact_recovery']['topk_sets_identical']} "
            f"tie_equivalent={arm['numerical_order_validation']['tie_equivalent_order']}"
        )
    print(f"Saved pilot result: {output_path}")


if __name__ == "__main__":
    main()
