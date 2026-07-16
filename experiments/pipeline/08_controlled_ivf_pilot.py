"""Controlled exact, FAISS-IVF, PDX-IVF, and optional PLAID benchmark.

This runner fixes the main measurement problems in the historical pipeline:
all arms run in one Linux process, rotate execution order, and time the complete
online path from resident query token vectors to final document IDs. The IVF
arms use matched exact-rerank budgets; PLAID's configured full-scoring budget is
reported separately because its API does not expose the realized candidate
count. Index build and input loading remain offline.

Use a small SciFact subset first. A run with `--max-documents 0` and
`--max-queries 0` uses all prepared inputs but should only be treated as final
after the pilot result and protocol have been reviewed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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
from candidate_pipeline import (  # noqa: E402
    CandidateFeatures,
    actual_rerank_count,
    build_candidate_features,
    select_candidates,
)
from kernels import exact_maxsim_scores, extension_available  # noqa: E402
from plaid_benchmark import (  # noqa: E402
    PlaidConfig,
    build_plaid_indexes,
    parse_plaid_config,
    plaid_action,
)
from utils_colbert import (  # noqa: E402
    compute_qrels_metrics,
    l2_normalize,
    load_packed_embeddings,
    maxsim_score,
    save_json,
    topk_overlap,
    unpack_embeddings,
)

PDX_COMMIT = "fdc62f2d22b3793060abf633cb5407438c7f739b"
DEFAULT_PDX_SOURCE = Path("/home/telle/data-engineering-pdx-clean/external/PDX")
DEFAULT_C_VALUES = (20, 50)
REPORT_K_VALUES = (1, 3, 5, 10)
RANDOM_SEED = 0
TRAINING_POINTS_PER_BUCKET = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--embeddings-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-documents", type=int, default=500)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--top-l", type=int, default=50)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--c-values", nargs="+", type=int, default=list(DEFAULT_C_VALUES))
    parser.add_argument("--selection-policy", default="approx_score")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nbuckets", type=int, default=0)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--pdx-source", type=Path, default=DEFAULT_PDX_SOURCE)
    parser.add_argument("--skip-ivf", action="store_true")
    parser.add_argument(
        "--plaid-config",
        action="append",
        type=parse_plaid_config,
        default=[],
        metavar="NPROBE:FULL_SCORES",
        help="Repeat to add PLAID arms, for example 16:100",
    )
    parser.add_argument("--plaid-index-folder", type=Path)
    parser.add_argument("--plaid-index-name", default="")
    parser.add_argument("--plaid-nbits", type=int, default=4)
    parser.add_argument("--plaid-kmeans-niters", type=int, default=4)
    parser.add_argument("--plaid-rebuild-index", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def git_snapshot(path: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(path), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "path": str(path),
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.max_documents < 0 or args.query_start < 0 or args.max_queries < 0:
        raise SystemExit("max-documents, query-start, and max-queries must be non-negative")
    if args.top_l <= 0 or args.nprobe <= 0 or args.k <= 0:
        raise SystemExit("top-l, nprobe, and k must be positive")
    if args.threads <= 0 or args.warmup_runs < 0 or args.measured_runs <= 0:
        raise SystemExit("invalid thread, warm-up, or measured-run count")
    if not args.c_values or any(value < 0 for value in args.c_values):
        raise SystemExit("all C values must be non-negative; C=0 means full pool")
    if args.plaid_nbits <= 0 or args.plaid_kmeans_niters <= 0:
        raise SystemExit("PLAID nbits and kmeans iterations must be positive")
    if args.plaid_rebuild_index and not args.plaid_config:
        raise SystemExit("--plaid-rebuild-index requires at least one --plaid-config")
    if args.plaid_config and (args.plaid_index_folder is None or not args.plaid_index_name):
        raise SystemExit("PLAID arms require --plaid-index-folder and --plaid-index-name")
    if any(config.n_full_scores < args.k for config in args.plaid_config):
        raise SystemExit("every PLAID full-score budget must be at least k")
    labels = [config.label for config in args.plaid_config]
    if len(labels) != len(set(labels)):
        raise SystemExit("duplicate PLAID configurations are not allowed")


def load_inputs(corpus_dir: Path, embeddings_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    document_pack = load_packed_embeddings(embeddings_dir / "documents_packed.npz")
    query_pack = load_packed_embeddings(embeddings_dir / "queries_packed.npz")

    document_count = len(document_pack["ids"])
    available_query_count = len(query_pack["ids"])
    if args.query_start >= available_query_count:
        raise SystemExit(
            f"query-start={args.query_start} exceeds available queries={available_query_count}"
        )
    query_stop = available_query_count
    if args.max_documents:
        document_count = min(document_count, args.max_documents)
    if args.max_queries:
        query_stop = min(query_stop, args.query_start + args.max_queries)
    query_count = query_stop - args.query_start
    if document_count < args.k:
        raise SystemExit(f"Need at least k={args.k} documents, found {document_count}")
    if query_count == 0:
        raise SystemExit("No queries selected")

    document_offsets = np.ascontiguousarray(
        document_pack["offsets"][: document_count + 1],
        dtype=np.int64,
    )
    document_token_count = int(document_offsets[-1])
    document_values = l2_normalize(document_pack["values"][:document_token_count])

    original_query_offsets = query_pack["offsets"][args.query_start : query_stop + 1]
    query_token_start = int(original_query_offsets[0])
    query_token_stop = int(original_query_offsets[-1])
    query_offsets = np.ascontiguousarray(
        original_query_offsets - query_token_start,
        dtype=np.int64,
    )
    query_values = l2_normalize(
        query_pack["values"][query_token_start:query_token_stop]
    )

    document_ids = document_pack["ids"][:document_count]
    query_ids = query_pack["ids"][args.query_start:query_stop]
    documents = unpack_embeddings(document_values, document_offsets)
    queries = unpack_embeddings(query_values, query_offsets)
    token_to_doc_index = np.repeat(
        np.arange(document_count, dtype=np.int64),
        np.diff(document_offsets),
    )

    with (corpus_dir / "qrels.json").open("r", encoding="utf-8") as file:
        raw_qrels = json.load(file)
    selected_documents = set(document_ids)
    qrels = {
        query_id: [doc_id for doc_id in raw_qrels.get(query_id, []) if doc_id in selected_documents]
        for query_id in query_ids
    }

    return {
        "document_values": document_values,
        "document_offsets": document_offsets,
        "query_values": query_values,
        "query_offsets": query_offsets,
        "documents": documents,
        "queries": queries,
        "document_ids": document_ids,
        "query_ids": query_ids,
        "token_to_doc_index": token_to_doc_index,
        "qrels": qrels,
    }


def pack_matrices(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(matrices) + 1, dtype=np.int64)
    for index, matrix in enumerate(matrices):
        offsets[index + 1] = offsets[index] + matrix.shape[0]
    values = np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)
    return values, offsets


def rank_scores(scores: np.ndarray, document_indices: np.ndarray, k: int) -> list[int]:
    order = np.lexsort((document_indices, -scores))
    return document_indices[order[:k]].astype(int).tolist()


def validate_compiled_kernel(inputs: dict[str, Any]) -> dict[str, Any]:
    documents = inputs["documents"][: min(8, len(inputs["documents"]))]
    queries = inputs["queries"][: min(2, len(inputs["queries"]))]
    document_values, document_offsets = pack_matrices(documents)
    query_values, query_offsets = pack_matrices(queries)
    actual = exact_maxsim_scores(
        document_values,
        document_offsets,
        query_values,
        query_offsets,
    )
    expected = np.empty_like(actual)
    for query_index, query in enumerate(queries):
        for document_index, document in enumerate(documents):
            expected[query_index, document_index] = maxsim_score(query, document)

    maximum_error = float(np.max(np.abs(actual - expected)))
    rankings_match = all(
        rank_scores(actual[row], np.arange(actual.shape[1]), actual.shape[1])
        == rank_scores(expected[row], np.arange(expected.shape[1]), expected.shape[1])
        for row in range(actual.shape[0])
    )
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6) or not rankings_match:
        raise SystemExit(
            "Compiled MaxSim validation failed: "
            f"max_abs_error={maximum_error}, rankings_match={rankings_match}"
        )
    return {
        "documents": len(documents),
        "queries": len(queries),
        "max_abs_score_error": maximum_error,
        "rankings_match": rankings_match,
    }


def training_sample(tokens: np.ndarray, sample_count: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(tokens.shape[0], size=sample_count, replace=False)
    indices.sort()
    return np.ascontiguousarray(tokens[indices], dtype=np.float32)


def build_faiss_index(tokens: np.ndarray, sample: np.ndarray, nbuckets: int, threads: int):
    try:
        import faiss
    except ModuleNotFoundError as error:
        raise SystemExit("faiss-cpu is required in the controlled WSL environment") from error

    faiss.omp_set_num_threads(threads)
    quantizer = faiss.IndexFlatL2(tokens.shape[1])
    index = faiss.IndexIVFFlat(quantizer, tokens.shape[1], nbuckets, faiss.METRIC_L2)
    start = perf_counter()
    index.train(sample)
    train_seconds = perf_counter() - start
    start = perf_counter()
    index.add(tokens)
    load_seconds = perf_counter() - start
    return index, {
        "train_seconds": train_seconds,
        "load_seconds": load_seconds,
        "total_seconds": train_seconds + load_seconds,
    }


def build_pdx_index(tokens: np.ndarray, sample: np.ndarray, nbuckets: int):
    try:
        from pdxearch.constants import PDXConstants
        from pdxearch.index_factory import IndexPDXBONDIVFFlat
    except ModuleNotFoundError as error:
        raise SystemExit("The pinned PDX package is required in WSL") from error

    if "l2sq" not in PDXConstants.SUPPORTED_METRICS:
        raise SystemExit(f"Expected PDX l2sq metric, got {PDXConstants.SUPPORTED_METRICS}")
    index = IndexPDXBONDIVFFlat(ndim=tokens.shape[1], nbuckets=nbuckets)
    start = perf_counter()
    index.train(sample)
    train_seconds = perf_counter() - start
    start = perf_counter()
    index.add_load(tokens)
    load_seconds = perf_counter() - start
    return index, {
        "train_seconds": train_seconds,
        "load_seconds": load_seconds,
        "total_seconds": train_seconds + load_seconds,
    }


def faiss_hits(index, query: np.ndarray, top_l: int, nprobe: int) -> list[list[tuple[int, float]]]:
    index.nprobe = nprobe
    distances, indices = index.search(np.ascontiguousarray(query, dtype=np.float32), top_l)
    hits = []
    for token_indices, token_distances in zip(indices, distances):
        hits.append(
            [
                (int(index_value), 1.0 - float(distance_value) / 2.0)
                for index_value, distance_value in zip(token_indices, token_distances)
                if int(index_value) >= 0
            ]
        )
    return hits


def pdx_hits(index, query: np.ndarray, top_l: int, nprobe: int) -> list[list[tuple[int, float]]]:
    hits = []
    for query_token in query:
        result = index.search(
            np.ascontiguousarray(query_token, dtype=np.float32),
            top_l,
            nprobe=nprobe,
        )
        hits.append(
            [(int(hit.index), 1.0 - float(hit.distance) / 2.0) for hit in result]
        )
    return hits


def exact_action(inputs: dict[str, Any], k: int) -> Callable[[], BenchmarkObservation]:
    document_indices = np.arange(len(inputs["documents"]), dtype=np.int64)

    def run() -> BenchmarkObservation:
        score_seconds = 0.0
        topk_seconds = 0.0
        rankings = []
        for query in inputs["queries"]:
            query_offsets = np.array([0, len(query)], dtype=np.int64)
            start = perf_counter()
            scores = exact_maxsim_scores(
                inputs["document_values"],
                inputs["document_offsets"],
                query,
                query_offsets,
            )[0]
            score_seconds += perf_counter() - start
            start = perf_counter()
            rankings.append(rank_scores(scores, document_indices, k))
            topk_seconds += perf_counter() - start
        return BenchmarkObservation(
            value={"rankings": rankings},
            stage_seconds={"exact_score": score_seconds, "topk": topk_seconds},
        )

    return run


def rerank_candidates(
    query: np.ndarray,
    documents: list[np.ndarray],
    selected: list[int],
    k: int,
) -> list[int]:
    selected_documents = [documents[index] for index in selected]
    document_values, document_offsets = pack_matrices(selected_documents)
    query_offsets = np.array([0, len(query)], dtype=np.int64)
    scores = exact_maxsim_scores(
        document_values,
        document_offsets,
        query,
        query_offsets,
    )[0]
    return rank_scores(scores, np.asarray(selected, dtype=np.int64), k)


def approximate_action(
    search: Callable[[np.ndarray], list[list[tuple[int, float]]]],
    inputs: dict[str, Any],
    top_c: int,
    policy: str,
    k: int,
) -> Callable[[], BenchmarkObservation]:
    def run() -> BenchmarkObservation:
        stage_seconds = {"search": 0.0, "aggregate": 0.0, "select": 0.0, "exact_rerank_topk": 0.0}
        rankings: list[list[int]] = []
        pools: list[list[int]] = []
        selected_by_query: list[list[int]] = []
        retrieved_hits: list[int] = []

        for query in inputs["queries"]:
            start = perf_counter()
            token_hits = search(query)
            stage_seconds["search"] += perf_counter() - start

            start = perf_counter()
            features: CandidateFeatures = build_candidate_features(
                token_hits,
                inputs["token_to_doc_index"],
                len(inputs["documents"]),
            )
            stage_seconds["aggregate"] += perf_counter() - start

            start = perf_counter()
            selected = select_candidates(features, top_c, policy)
            stage_seconds["select"] += perf_counter() - start

            start = perf_counter()
            ranking = rerank_candidates(query, inputs["documents"], selected, k)
            stage_seconds["exact_rerank_topk"] += perf_counter() - start

            rankings.append(ranking)
            pools.append(features.candidate_doc_indices.astype(int).tolist())
            selected_by_query.append(selected)
            retrieved_hits.append(sum(len(hits) for hits in token_hits))

        return BenchmarkObservation(
            value={
                "rankings": rankings,
                "pools": pools,
                "selected": selected_by_query,
                "retrieved_hits": retrieved_hits,
            },
            stage_seconds=stage_seconds,
        )

    return run


def ranking_ids(rankings: list[list[int]], document_ids: list[str], query_ids: list[str]) -> dict[str, list[str]]:
    return {
        query_id: [document_ids[index] for index in ranking]
        for query_id, ranking in zip(query_ids, rankings)
    }


def exact_recovery(
    reference: list[list[int]],
    candidate: list[list[int]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for k in REPORT_K_VALUES:
        overlaps = [
            topk_overlap([str(value) for value in ref], [str(value) for value in got], k)["ratio"]
            for ref, got in zip(reference, candidate)
        ]
        output[f"mean_recall@{k}"] = float(np.mean(overlaps))
        output[f"all_top{k}_recovered_queries"] = sum(
            set(ref[:k]).issubset(set(got)) for ref, got in zip(reference, candidate)
        )
    return output


def pool_recovery(reference: list[list[int]], pools: list[list[int]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for k in REPORT_K_VALUES:
        recalls = []
        all_recovered = 0
        for ref, pool_values in zip(reference, pools):
            pool = set(pool_values)
            top = set(ref[:k])
            recalls.append(len(top & pool) / len(top))
            all_recovered += int(top.issubset(pool))
        output[f"mean_recall@{k}"] = float(np.mean(recalls))
        output[f"all_top{k}_recovered_queries"] = all_recovered
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not extension_available():
        raise SystemExit("Build the exact kernel with cpp/exact_maxsim/build_wsl.sh first")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    corpus_dir = resolve(args.corpus_dir)
    embeddings_dir = resolve(args.embeddings_dir)
    output_path = resolve(args.output)
    pdx_source = args.pdx_source.resolve()
    pdx_snapshot = git_snapshot(pdx_source) if not args.skip_ivf else None
    if not args.skip_ivf and (
        pdx_snapshot["commit"] != PDX_COMMIT or pdx_snapshot["dirty"]
    ):
        raise SystemExit(
            "Controlled runs require clean PDX commit "
            f"{PDX_COMMIT}; observed {pdx_snapshot}"
        )

    print("Loading and normalizing packed embeddings...")
    inputs = load_inputs(corpus_dir, embeddings_dir, args)
    document_values = inputs["document_values"]
    if args.top_l > document_values.shape[0]:
        raise SystemExit("top-l exceeds the number of indexed document token vectors")
    validation = validate_compiled_kernel(inputs)

    nbuckets = args.nbuckets or 2 * math.ceil(math.sqrt(document_values.shape[0]))
    nbuckets = min(nbuckets, document_values.shape[0])
    sample_count = min(nbuckets * TRAINING_POINTS_PER_BUCKET, document_values.shape[0])
    sample = training_sample(document_values, sample_count)
    print(
        f"documents={len(inputs['documents'])} queries={len(inputs['queries'])} "
        f"doc_tokens={document_values.shape[0]} nbuckets={nbuckets}"
    )

    offline_index_build: dict[str, Any] = {}
    actions: dict[str, Callable[[], BenchmarkObservation]] = {
        "compiled_exact": exact_action(inputs, args.k)
    }
    if not args.skip_ivf:
        print("Building offline FAISS-IVF index...")
        faiss_index, faiss_build = build_faiss_index(
            document_values, sample, nbuckets, args.threads
        )
        print("Building offline PDX-IVF index...")
        pdx_index, pdx_build = build_pdx_index(document_values, sample, nbuckets)
        offline_index_build.update({"faiss_ivf": faiss_build, "pdx_ivf": pdx_build})

        faiss_search = lambda query: faiss_hits(faiss_index, query, args.top_l, args.nprobe)
        pdx_search = lambda query: pdx_hits(pdx_index, query, args.top_l, args.nprobe)
        for top_c in args.c_values:
            budget_label = "full" if top_c == 0 else str(top_c)
            actions[f"faiss_ivf_c{budget_label}"] = approximate_action(
                faiss_search, inputs, top_c, args.selection_policy, args.k
            )
            actions[f"pdx_ivf_c{budget_label}"] = approximate_action(
                pdx_search, inputs, top_c, args.selection_policy, args.k
            )

    plaid_configs: list[PlaidConfig] = list(args.plaid_config)
    if plaid_configs:
        plaid_index_folder = resolve(args.plaid_index_folder)
        print(
            f"Building/loading offline PLAID index with {len(plaid_configs)} "
            "search configurations..."
        )
        plaid_indexes, plaid_build = build_plaid_indexes(
            inputs["documents"],
            inputs["document_ids"],
            plaid_configs,
            index_folder=plaid_index_folder,
            index_name=args.plaid_index_name,
            nbits=args.plaid_nbits,
            kmeans_niters=args.plaid_kmeans_niters,
            rebuild=args.plaid_rebuild_index,
            threads=args.threads,
        )
        offline_index_build["plaid"] = plaid_build
        for config in plaid_configs:
            actions[config.label] = plaid_action(
                plaid_indexes[config.label],
                inputs["queries"],
                inputs["document_ids"],
                args.k,
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

    arm_results = {}
    plaid_results = {}
    full_exact_comparisons = len(inputs["queries"]) * len(inputs["documents"])
    for name, value in values.items():
        if name == "compiled_exact":
            continue
        ids = ranking_ids(value["rankings"], inputs["document_ids"], inputs["query_ids"])
        if name.startswith("plaid_"):
            config = next(config for config in plaid_configs if config.label == name)
            configured_scores = min(config.n_full_scores, len(inputs["documents"]))
            plaid_results[name] = {
                "config": config.as_dict(),
                "timing": timing["arms"][name],
                "exact_recovery": exact_recovery(exact_rankings, value["rankings"]),
                "qrels_metrics": compute_qrels_metrics(
                    ids,
                    inputs["qrels"],
                    k_values=REPORT_K_VALUES,
                    mrr_k=10,
                )["macro"],
                "configured_work": {
                    "full_score_budget_per_query": config.n_full_scores,
                    "upper_bound_full_scores": configured_scores * len(inputs["queries"]),
                    "upper_bound_full_score_ratio": configured_scores
                    / len(inputs["documents"]),
                    "actual_candidate_count_exposed_by_api": False,
                },
                "topk_rankings": ids,
                "topk_scores": value["scores"],
            }
            continue
        reranked_comparisons = actual_rerank_count(value["selected"])
        arm_results[name] = {
            "timing": timing["arms"][name],
            "exact_recovery": exact_recovery(exact_rankings, value["rankings"]),
            "candidate_pool_recovery": pool_recovery(exact_rankings, value["pools"]),
            "qrels_metrics": compute_qrels_metrics(
                ids,
                inputs["qrels"],
                k_values=REPORT_K_VALUES,
                mrr_k=10,
            )["macro"],
            "work": {
                "mean_candidate_pool_size": float(np.mean([len(pool) for pool in value["pools"]])),
                "mean_selected_candidates": float(np.mean([len(items) for items in value["selected"]])),
                "reranked_comparisons": reranked_comparisons,
                "reranked_comparison_ratio": reranked_comparisons / full_exact_comparisons,
                "total_retrieved_token_hits": int(sum(value["retrieved_hits"])),
            },
            "topk_rankings": ids,
        }

    cross_engine_agreement = {}
    if not args.skip_ivf:
        for top_c in args.c_values:
            budget_label = "full" if top_c == 0 else str(top_c)
            faiss_value = values[f"faiss_ivf_c{budget_label}"]
            pdx_value = values[f"pdx_ivf_c{budget_label}"]
            cross_engine_agreement[f"c{budget_label}"] = {
                "candidate_pools_equal": all(
                    set(left) == set(right)
                    for left, right in zip(faiss_value["pools"], pdx_value["pools"])
                ),
                "selected_candidates_equal": faiss_value["selected"] == pdx_value["selected"],
                "topk_rankings_equal": faiss_value["rankings"] == pdx_value["rankings"],
                "retrieved_hit_counts_equal": (
                    faiss_value["retrieved_hits"] == pdx_value["retrieved_hits"]
                ),
            }

    result = {
        "schema_version": (
            "controlled_ivf_plaid_v2" if plaid_configs else "controlled_ivf_pilot_v1"
        ),
        "status": "pilot_not_final_claim",
        "timing_boundary": (
            "resident normalized query token vectors to final ranked document IDs; "
            "IVF includes search, aggregation, selection, exact reranking, and top-k, "
            "while PLAID includes its complete opaque retrieval/full-scoring call"
        ),
        "metadata": collect_runtime_metadata(
            PROJECT_ROOT,
            input_paths=(
                corpus_dir / "qrels.json",
                embeddings_dir / "documents_packed.npz",
                embeddings_dir / "queries_packed.npz",
            ),
            packages=(
                "numpy",
                "faiss-cpu",
                "pylate",
                "fast-plaid",
                "torch",
                "pdxearch",
            ),
        ),
        "pdx_source": pdx_snapshot,
        "dataset": {
            "corpus_dir": str(corpus_dir),
            "embeddings_dir": str(embeddings_dir),
            "selection": (
                "subset pilot"
                if args.max_documents or args.query_start or args.max_queries
                else "full prepared inputs"
            ),
            "query_start": args.query_start,
            "documents": len(inputs["documents"]),
            "queries": len(inputs["queries"]),
            "document_token_vectors": int(document_values.shape[0]),
            "query_token_vectors": int(inputs["query_values"].shape[0]),
            "embedding_dimension": int(document_values.shape[1]),
            "retained_qrels_labels": int(sum(len(values) for values in inputs["qrels"].values())),
        },
        "config": {
            "seed": RANDOM_SEED,
            "k": args.k,
            "top_l": args.top_l,
            "nprobe": args.nprobe,
            "c_values": args.c_values,
            "c_zero_meaning": "rerank every unique document in the candidate pool",
            "selection_policy": args.selection_policy,
            "nbuckets": nbuckets,
            "training_points": sample_count,
            "threads": args.threads,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "metric": "squared L2 on L2-normalized tokens; similarity = 1 - distance/2",
            "skip_ivf": args.skip_ivf,
            "plaid": {
                "enabled": bool(plaid_configs),
                "configs": [config.as_dict() for config in plaid_configs],
                "nbits": args.plaid_nbits,
                "kmeans_niters": args.plaid_kmeans_niters,
                "candidate_count_note": (
                    "n_full_scores is a configured PLAID full-scoring budget; "
                    "the current API does not expose the realized candidate count"
                ),
            },
        },
        "kernel_validation": validation,
        "offline_index_build": offline_index_build,
        "interleaving": {
            "arm_order": timing["arm_order"],
            "measured_execution_order": timing["measured_execution_order"],
        },
        "exact_reference": {
            "timing": timing["arms"]["compiled_exact"],
            "qrels_metrics": exact_qrels,
            "full_exact_query_document_comparisons": full_exact_comparisons,
            "topk_rankings": exact_ranking_ids,
        },
        "cross_engine_agreement": cross_engine_agreement,
        "approximate_arms": arm_results,
        "plaid_arms": plaid_results,
    }
    save_json(output_path, result)

    exact_median = timing["arms"]["compiled_exact"]["end_to_end_summary"]["median"]
    print(f"compiled_exact median={exact_median:.6f}s qrelsR@10={exact_qrels['recall@10']:.4f}")
    for name, arm in arm_results.items():
        median = arm["timing"]["end_to_end_summary"]["median"]
        recovery = arm["exact_recovery"]["mean_recall@10"]
        selected = arm["work"]["mean_selected_candidates"]
        print(f"{name:>18}: median={median:.6f}s exactR@10={recovery:.4f} meanC={selected:.1f}")
    for name, arm in plaid_results.items():
        median = arm["timing"]["end_to_end_summary"]["median"]
        recovery = arm["exact_recovery"]["mean_recall@10"]
        budget = arm["config"]["n_full_scores"]
        print(f"{name:>18}: median={median:.6f}s exactR@10={recovery:.4f} fullC={budget}")
    print(f"Saved pilot result: {output_path}")


if __name__ == "__main__":
    main()
