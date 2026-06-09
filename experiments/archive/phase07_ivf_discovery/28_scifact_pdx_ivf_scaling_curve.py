"""SciFact scaling curve: exact MaxSim vs IVF-on-PDX candidate generation.

Iteration 15 showed that IVF-on-PDX (`IndexPDXBONDIVFFlat`) with a small
`nprobe` gives a large candidate-generation speedup over both exhaustive BOND
and exact NumPy MaxSim, at the full 1000-document SciFact subset.

This script measures how that speedup behaves as the corpus grows, using the
already-exported SciFact embeddings on document-prefix subsets (the same
subsetting used in experiment 24). For each subset it reports:

- exact NumPy MaxSim time,
- IVF index build time (train + load, one-time),
- IVF candidate-generation time per (nprobe, L),
- candidate-pool coverage of the exact top-k,
- best reranked exact recall over the standard top-C policies,
- speedup = exact_seconds / candidate_generation_seconds.

`nprobe = 0` explores all buckets (exact). Run from the WSL PDX environment.

This is a component scaling curve for the current Python/WSL implementation,
not a controlled end-to-end speedup claim. Note that only 1000 documents are
available locally; a true large-scale curve needs a bigger corpus (e.g. CoRECT
/ BEIR), which is a planned next step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from time import perf_counter
from typing import Any

import numpy as np

from utils_colbert import (
    compute_qrels_metrics,
    flatten_document_embeddings,
    l2_normalize,
    load_packed_embeddings,
    maxsim_score,
    maxsim_scores_for_candidate_docs,
    rank_documents,
    save_json,
    topk_overlap,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "scifact_benchmark"
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "scifact_benchmark_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
OUTPUT_PATH = RESULTS_DIR / "scifact_pdx_ivf_scaling_curve.json"

DEFAULT_DOC_LIMITS = [250, 500, 1000]
DEFAULT_NPROBE_VALUES = [0, 8, 16]
DEFAULT_L_VALUES = [50, 100]
DEFAULT_C_VALUES = [20, 50, 100]
POLICIES = ["approx_score", "retrieved_token_count", "union_approx_and_count"]
REPORT_K_VALUES = [1, 3, 5, 10]
TRAINING_POINTS_PER_BUCKET = 50
RANDOM_SEED = 0


def import_pdx_api():
    try:
        from pdxearch.constants import PDXConstants
        from pdxearch.index_factory import IndexPDXBONDIVFFlat

        return IndexPDXBONDIVFFlat, PDXConstants.SUPPORTED_METRICS
    except ModuleNotFoundError as first_exc:
        source_path = PROJECT_ROOT / "external" / "PDX" / "python"
        if source_path.exists():
            sys.path.insert(0, str(source_path))
            try:
                from pdxearch.constants import PDXConstants
                from pdxearch.index_factory import IndexPDXBONDIVFFlat

                return IndexPDXBONDIVFFlat, PDXConstants.SUPPORTED_METRICS
            except ModuleNotFoundError as second_exc:
                raise SystemExit(
                    "Could not import PDX. Build/install PDX in WSL/Linux first. "
                    f"Original error: {second_exc}"
                ) from second_exc
        raise SystemExit(
            "Could not import PDX. Build/install PDX in WSL/Linux first. "
            f"Original error: {first_exc}"
        ) from first_exc


def filter_qrels_to_docs(qrels: dict[str, list[str]], doc_ids: list[str]) -> dict[str, list[str]]:
    doc_id_set = set(doc_ids)
    filtered = {}
    for query_id, relevant_ids in qrels.items():
        kept = [doc_id for doc_id in relevant_ids if doc_id in doc_id_set]
        if kept:
            filtered[query_id] = kept
    return filtered


def compute_exact_rankings(
    queries: list[np.ndarray],
    documents: list[np.ndarray],
    query_ids: list[str],
    doc_ids: list[str],
) -> dict[str, Any]:
    start = perf_counter()
    rankings_by_query = {}
    for query_id, query_matrix in zip(query_ids, queries):
        scores = np.empty(len(documents), dtype=np.float32)
        for doc_index, document_matrix in enumerate(documents):
            scores[doc_index] = maxsim_score(query_matrix, document_matrix, normalize=False)
        ranking = rank_documents(scores)
        rankings_by_query[query_id] = [doc_ids[index] for index in ranking]
    seconds = perf_counter() - start
    return {
        "seconds": seconds,
        "comparisons": len(queries) * len(documents),
        "rankings_by_query": rankings_by_query,
    }


def build_ivf_index(index_cls, flat_tokens: np.ndarray, nbuckets: int):
    rng = np.random.default_rng(RANDOM_SEED)
    training_points = min(nbuckets * TRAINING_POINTS_PER_BUCKET, flat_tokens.shape[0])
    sample_idx = rng.choice(flat_tokens.shape[0], size=training_points, replace=False)
    sample_idx.sort()
    training_sample = np.ascontiguousarray(flat_tokens[sample_idx], dtype=np.float32)

    index = index_cls(ndim=flat_tokens.shape[1], nbuckets=nbuckets)
    train_start = perf_counter()
    index.train(training_sample)
    train_seconds = perf_counter() - train_start
    load_start = perf_counter()
    index.add_load(flat_tokens)
    load_seconds = perf_counter() - load_start
    return index, {
        "nbuckets": nbuckets,
        "training_points": int(training_points),
        "train_seconds": train_seconds,
        "load_seconds": load_seconds,
        "build_seconds": train_seconds + load_seconds,
    }


def retrieve_query_token_hits(index, query_matrix: np.ndarray, top_l: int, nprobe: int):
    query_token_hits = []
    start = perf_counter()
    for query_token in query_matrix:
        hits = index.search(np.ascontiguousarray(query_token, dtype=np.float32), top_l, nprobe=nprobe)
        query_token_hits.append([(int(hit.index), 1.0 - (float(hit.distance) / 2.0)) for hit in hits])
    return query_token_hits, perf_counter() - start


def build_candidate_features(
    query_token_hits: list[list[tuple[int, float]]],
    token_to_doc_index: np.ndarray,
    num_documents: int,
) -> dict[str, Any]:
    query_token_count = len(query_token_hits)
    per_token_doc_best = np.full((query_token_count, num_documents), -np.inf, dtype=np.float32)
    retrieved_token_count = np.zeros(num_documents, dtype=np.int32)
    candidate_doc_indices: set[int] = set()
    for query_token_index, hits in enumerate(query_token_hits):
        for token_index, similarity in hits:
            doc_index = int(token_to_doc_index[int(token_index)])
            candidate_doc_indices.add(doc_index)
            retrieved_token_count[doc_index] += 1
            if similarity > per_token_doc_best[query_token_index, doc_index]:
                per_token_doc_best[query_token_index, doc_index] = similarity
    matched_mask = np.isfinite(per_token_doc_best)
    approx_scores = np.where(matched_mask, per_token_doc_best, 0.0).sum(axis=0)
    matched_query_token_count = matched_mask.sum(axis=0).astype(np.int32)
    return {
        "candidate_doc_indices": candidate_doc_indices,
        "approx_scores": approx_scores.astype(np.float32),
        "retrieved_token_count": retrieved_token_count,
        "matched_query_token_count": matched_query_token_count,
    }


def sort_by_approx(features: dict[str, Any]) -> list[int]:
    return sorted(features["candidate_doc_indices"], key=lambda doc: (-float(features["approx_scores"][doc]), doc))


def sort_by_count(features: dict[str, Any]) -> list[int]:
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (-int(features["retrieved_token_count"][doc]), -float(features["approx_scores"][doc]), doc),
    )


def select_candidates(features: dict[str, Any], top_c: int, policy: str) -> list[int]:
    approx_order = sort_by_approx(features)
    count_order = sort_by_count(features)
    if policy == "approx_score":
        return approx_order[:top_c]
    if policy == "retrieved_token_count":
        return count_order[:top_c]
    if policy == "union_approx_and_count":
        first_take = math.ceil(top_c / 2)
        approx_head = set(approx_order[:first_take])
        count_head = set(count_order[: top_c - first_take])
        selected = approx_head.union(count_head)
        for doc_index in approx_order:
            if len(selected) >= top_c:
                break
            selected.add(doc_index)
        return sorted(
            selected,
            key=lambda doc: (
                -(doc in approx_head and doc in count_head),
                -int(features["matched_query_token_count"][doc]),
                -float(features["approx_scores"][doc]),
                doc,
            ),
        )[:top_c]
    raise ValueError(f"Unknown policy: {policy}")


def pool_coverage(generated: list[dict[str, Any]], exact_rankings: dict[str, list[str]], doc_ids: list[str]):
    top5_hits = 0
    top10_hits = 0
    pool_sizes = []
    recall5 = []
    recall10 = []
    for item in generated:
        pool = {doc_ids[index] for index in item["features"]["candidate_doc_indices"]}
        pool_sizes.append(len(pool))
        reference = exact_rankings[item["query_id"]]
        oracle_pool_ranking = [doc_id for doc_id in reference if doc_id in pool]
        top5_hits += int(set(reference[:5]).issubset(pool))
        top10_hits += int(set(reference[:10]).issubset(pool))
        recall5.append(topk_overlap(reference, oracle_pool_ranking, 5)["ratio"])
        recall10.append(topk_overlap(reference, oracle_pool_ranking, 10)["ratio"])
    n = len(generated)
    return {
        "mean_candidate_pool_size": float(np.mean(pool_sizes)),
        "pool_recall5_all": top5_hits / n,
        "pool_recall10_all": top10_hits / n,
        "pool_mean_recall5": float(np.mean(recall5)),
        "pool_mean_recall10": float(np.mean(recall10)),
    }


def best_rerank_recall(
    generated: list[dict[str, Any]],
    queries: list[np.ndarray],
    documents: list[np.ndarray],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
    c_values: list[int],
) -> dict[str, Any]:
    """Find the best exact recall@5 (C=20) and recall@10 (C=50) over policies."""
    results = []
    for policy in POLICIES:
        for top_c in c_values:
            recall5 = []
            recall10 = []
            rerank_seconds = 0.0
            selected_counts = []
            for item in generated:
                features = item["features"]
                selected = select_candidates(features, top_c, policy)
                selected_counts.append(len(selected))
                start = perf_counter()
                scores = maxsim_scores_for_candidate_docs(
                    queries[item["query_index"]], documents, selected, normalize=False
                )
                rerank_seconds += perf_counter() - start
                ordered = list(scores)
                score_arr = np.array([scores[i] for i in ordered], dtype=np.float32)
                reranked = [doc_ids[ordered[i]] for i in rank_documents(score_arr)]
                reference = exact_rankings[item["query_id"]]
                recall5.append(topk_overlap(reference, reranked, 5)["ratio"])
                recall10.append(topk_overlap(reference, reranked, 10)["ratio"])
            results.append(
                {
                    "policy": policy,
                    "top_c": top_c,
                    "mean_exact_recall5": float(np.mean(recall5)),
                    "mean_exact_recall10": float(np.mean(recall10)),
                    "rerank_seconds": rerank_seconds,
                    "mean_selected_candidate_count": float(np.mean(selected_counts)),
                }
            )
    best_c20_recall5 = max(
        (r for r in results if r["top_c"] == 20),
        key=lambda r: r["mean_exact_recall5"],
        default=None,
    )
    best_c50_recall10 = max(
        (r for r in results if r["top_c"] == 50),
        key=lambda r: r["mean_exact_recall10"],
        default=None,
    )
    return {"best_c20_recall5": best_c20_recall5, "best_c50_recall10": best_c50_recall10, "all": results}


def run_for_doc_limit(
    *,
    index_cls,
    doc_limit: int,
    normalized_documents_all: list[np.ndarray],
    normalized_queries: list[np.ndarray],
    doc_ids_all: list[str],
    query_ids: list[str],
    nprobe_values: list[int],
    l_values: list[int],
    c_values: list[int],
) -> dict[str, Any]:
    documents = normalized_documents_all[:doc_limit]
    doc_ids = doc_ids_all[:doc_limit]
    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(documents)

    exact = compute_exact_rankings(normalized_queries, documents, query_ids, doc_ids)
    exact_rankings = exact["rankings_by_query"]

    np.random.seed(RANDOM_SEED)
    nbuckets = 2 * math.ceil(math.sqrt(flat_tokens.shape[0]))
    index, build_info = build_ivf_index(index_cls, flat_tokens, nbuckets)

    nprobe_runs = []
    for nprobe in nprobe_values:
        l_runs = []
        for top_l in l_values:
            generated = []
            generation_seconds = 0.0
            for query_index, query_matrix in enumerate(normalized_queries):
                hits, seconds = retrieve_query_token_hits(index, query_matrix, top_l, nprobe)
                features = build_candidate_features(hits, token_to_doc_index, len(doc_ids))
                generated.append(
                    {"query_id": query_ids[query_index], "query_index": query_index, "features": features}
                )
                generation_seconds += seconds

            coverage = pool_coverage(generated, exact_rankings, doc_ids)
            rerank = best_rerank_recall(
                generated, normalized_queries, documents, exact_rankings, doc_ids, c_values
            )
            speedup = exact["seconds"] / generation_seconds if generation_seconds else 0.0
            l_runs.append(
                {
                    "nprobe": nprobe,
                    "top_l": top_l,
                    "candidate_generation_seconds": generation_seconds,
                    "speedup_vs_exact": speedup,
                    "pool_coverage": coverage,
                    "best_rerank": rerank,
                }
            )
            best50 = rerank["best_c50_recall10"]
            print(
                f"  nprobe={nprobe if nprobe else 'ALL':>3} L={top_l:>3}: "
                f"gen={generation_seconds:.4f}s speedup={speedup:6.2f}x "
                f"poolR5={coverage['pool_recall5_all']:.3f} poolR10={coverage['pool_recall10_all']:.3f} "
                f"bestC50_rec@10={best50['mean_exact_recall10']:.4f}"
            )
        nprobe_runs.append({"nprobe": nprobe, "l_runs": l_runs})

    return {
        "doc_limit": doc_limit,
        "documents": len(doc_ids),
        "document_token_vectors": int(flat_tokens.shape[0]),
        "query_token_vectors": int(sum(len(q) for q in normalized_queries)),
        "exact": {"seconds": exact["seconds"], "comparisons": exact["comparisons"]},
        "ivf_index": build_info,
        "nprobe_runs": nprobe_runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-limits", nargs="+", type=int, default=DEFAULT_DOC_LIMITS)
    parser.add_argument("--nprobe-values", nargs="+", type=int, default=DEFAULT_NPROBE_VALUES)
    parser.add_argument("--l-values", nargs="+", type=int, default=DEFAULT_L_VALUES)
    parser.add_argument("--c-values", nargs="+", type=int, default=DEFAULT_C_VALUES)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_cls, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")

    with (CORPUS_DIR / "qrels.json").open("r", encoding="utf-8") as file:
        json.load(file)  # presence check only; ranking comparison uses exact MaxSim
    document_pack = load_packed_embeddings(EMBEDDINGS_DIR / "documents_packed.npz")
    query_pack = load_packed_embeddings(EMBEDDINGS_DIR / "queries_packed.npz")
    documents = unpack_embeddings(document_pack["values"], document_pack["offsets"])
    queries = unpack_embeddings(query_pack["values"], query_pack["offsets"])
    doc_ids_all = document_pack["ids"]
    query_ids = query_pack["ids"]
    max_docs = len(doc_ids_all)
    doc_limits = [limit for limit in args.doc_limits if 0 < limit <= max_docs]
    if not doc_limits:
        raise SystemExit(f"No valid doc limits. Available document count: {max_docs}")

    normalized_documents_all = [l2_normalize(document) for document in documents]
    normalized_queries = [l2_normalize(query) for query in queries]

    subsets = []
    for doc_limit in doc_limits:
        print(f"doc_limit={doc_limit}")
        subsets.append(
            run_for_doc_limit(
                index_cls=index_cls,
                doc_limit=doc_limit,
                normalized_documents_all=normalized_documents_all,
                normalized_queries=normalized_queries,
                doc_ids_all=doc_ids_all,
                query_ids=query_ids,
                nprobe_values=args.nprobe_values,
                l_values=args.l_values,
                c_values=args.c_values,
            )
        )

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "candidate_generation": "ivf_bond_l2sq_on_l2_normalized_vectors",
            "rerank": "exact_normalized_inner_product_maxsim",
            "nprobe_zero_meaning": "explore all buckets (exact)",
        },
        "timing_note": (
            "Component scaling curve for the current Python/WSL implementation. "
            "Not a controlled end-to-end speedup claim. Only 1000 documents are "
            "available locally; a true large-scale curve needs a bigger corpus."
        ),
        "global_dataset": {
            "available_documents": max_docs,
            "queries": len(query_ids),
            "total_available_document_token_vectors": int(document_pack["values"].shape[0]),
            "total_query_token_vectors": int(query_pack["values"].shape[0]),
            "embedding_dimension": int(document_pack["values"].shape[1]),
        },
        "doc_limits": doc_limits,
        "nprobe_values": args.nprobe_values,
        "l_values": args.l_values,
        "c_values": args.c_values,
        "policies": POLICIES,
        "subsets": subsets,
    }
    save_json(args.output, output)

    print("\nSciFact IVF-on-PDX scaling curve")
    for subset in subsets:
        print(
            f"docs={subset['documents']} tokens={subset['document_token_vectors']} "
            f"exact={subset['exact']['seconds']:.4f}s ivf_build={subset['ivf_index']['build_seconds']:.4f}s"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
