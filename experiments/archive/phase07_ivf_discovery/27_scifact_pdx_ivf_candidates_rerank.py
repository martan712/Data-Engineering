"""SciFact: IVF-on-PDX (BOND) candidate generation + exact MaxSim reranking.

Motivation
----------
Flat PDX-BOND (`IndexPDXBONDFlat`) is *exhaustive*: every query token still
scans every document token vector, pruning only dimensions. On the 1000-doc
SciFact subset that loses to a dense NumPy MaxSim matmul (see experiments 22-26).

PDX's own author reports that flat PDX "falls short when retrieving more than 10
neighbours or targeting recall below 0.95", and that the fix is an IVF index on
the PDX layout, where each query token only scans a few clusters (`nprobe`)
instead of the whole collection. This experiment tests that route for ColBERT
token-level candidate generation:

- `nprobe = 0`  -> explore ALL buckets (exact, lets BOND pruning shine)
- `nprobe = k`  -> explore only k clusters per query token (approximate, faster)

For each (nprobe, L) we measure candidate-generation time, candidate-pool
coverage of the exact MaxSim top-k, and exact recall after reranking the
top-C selected candidates. Results are compared against the exact NumPy MaxSim
baseline (experiment 21) and the flat-BOND numbers (experiment 22).

This is a component-timing study, not a controlled end-to-end speedup claim.
"""

from __future__ import annotations

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
EXACT_PATH = RESULTS_DIR / "scifact_colbert_maxsim_numpy.json"
FLAT_BOND_PATH = RESULTS_DIR / "scifact_pdx_candidates_rerank.json"
OUTPUT_PATH = RESULTS_DIR / "scifact_pdx_ivf_candidates_rerank.json"

# 0 means "explore all buckets" (exact). Other values probe that many clusters.
NPROBE_VALUES = [0, 4, 8, 16, 32, 64]
L_VALUES = [10, 50, 100]
C_VALUES = [20, 50, 100]
REPORT_K_VALUES = [1, 3, 5, 10]
POLICIES = [
    "approx_score",
    "retrieved_token_count",
    "union_approx_and_count",
]
TRAINING_POINTS_PER_BUCKET = 50
RANDOM_SEED = 0


def import_pdx_api():
    """Import the IVF-BOND PDX index, falling back to the source checkout."""
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


def build_ivf_index(index_cls, flat_tokens: np.ndarray, nbuckets: int) -> tuple[Any, dict[str, float]]:
    """Train and load an IVF-BOND index over the flat token-vector matrix."""
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
    """Search each query token vector against the IVF-BOND index."""
    query_token_hits = []
    start = perf_counter()
    for query_token in query_matrix:
        hits = index.search(
            np.ascontiguousarray(query_token, dtype=np.float32),
            top_l,
            nprobe=nprobe,
        )
        query_token_hits.append(
            [(int(hit.index), 1.0 - (float(hit.distance) / 2.0)) for hit in hits]
        )
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
        "retrieved_token_vectors": sum(len(hits) for hits in query_token_hits),
    }


def sort_by_approx(features: dict[str, Any]) -> list[int]:
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (-float(features["approx_scores"][doc]), doc),
    )


def sort_by_count(features: dict[str, Any]) -> list[int]:
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (
            -int(features["retrieved_token_count"][doc]),
            -float(features["approx_scores"][doc]),
            doc,
        ),
    )


def select_candidates(features: dict[str, Any], top_c: int | None, policy: str) -> list[int]:
    if top_c is None:
        return sort_by_approx(features)
    approx_order = sort_by_approx(features)
    count_order = sort_by_count(features)
    if policy == "approx_score":
        return approx_order[:top_c]
    if policy == "retrieved_token_count":
        return count_order[:top_c]
    if policy == "union_approx_and_count":
        first_take = math.ceil(top_c / 2)
        selected = set(approx_order[:first_take]).union(count_order[: top_c - first_take])
        for doc_index in approx_order:
            if len(selected) >= top_c:
                break
            selected.add(doc_index)
        return sorted(
            selected,
            key=lambda doc: (
                -(
                    doc in set(approx_order[:first_take])
                    and doc in set(count_order[: top_c - first_take])
                ),
                -int(features["matched_query_token_count"][doc]),
                -float(features["approx_scores"][doc]),
                doc,
            ),
        )[:top_c]
    raise ValueError(f"Unknown policy: {policy}")


def rerank_candidates(
    query_matrix: np.ndarray,
    documents: list[np.ndarray],
    selected_doc_indices: list[int],
) -> tuple[list[int], float]:
    start = perf_counter()
    candidate_scores = maxsim_scores_for_candidate_docs(
        query_matrix,
        documents,
        selected_doc_indices,
        normalize=False,
    )
    elapsed = perf_counter() - start
    if not candidate_scores:
        return [], elapsed
    ordered_candidates = list(candidate_scores)
    scores = np.array([candidate_scores[index] for index in ordered_candidates], dtype=np.float32)
    ranking = rank_documents(scores)
    return [ordered_candidates[index] for index in ranking], elapsed


def compare_exact(reference: list[str], reranked: list[str], selected: set[str], pool: set[str]) -> dict:
    exact_top5 = set(reference[:5])
    exact_top10 = set(reference[:10])
    return {
        "exact_recall_at_k": {
            str(k): topk_overlap(reference, reranked, k)["ratio"] for k in REPORT_K_VALUES
        },
        "exact_top1_match": reference[:1] == reranked[:1],
        "exact_top5_all_recovered": exact_top5.issubset(set(reranked[:5])),
        "exact_top10_all_recovered": exact_top10.issubset(set(reranked[:10])),
        "pool_contains_top5": exact_top5.issubset(pool),
        "pool_contains_top10": exact_top10.issubset(pool),
        "selector_loss_at_5": (
            "none"
            if exact_top5.issubset(selected)
            else "pool_coverage_failure"
            if not exact_top5.issubset(pool)
            else "selector_failure"
        ),
        "selector_loss_at_10": (
            "none"
            if exact_top10.issubset(selected)
            else "pool_coverage_failure"
            if not exact_top10.issubset(pool)
            else "selector_failure"
        ),
    }


def summarize_policy(
    *,
    policy: str,
    nprobe: int,
    top_l: int,
    top_c: int | None,
    generated: list[dict[str, Any]],
    normalized_queries: list[np.ndarray],
    normalized_documents: list[np.ndarray],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
    qrels: dict[str, list[str]],
    full_exact_comparisons: int,
) -> dict[str, Any]:
    query_results = []
    rankings_by_query = {}
    total_rerank_seconds = 0.0
    for item in generated:
        query_id = item["query_id"]
        query_index = item["query_index"]
        features = item["features"]
        selected = select_candidates(features, top_c, policy)
        reranked, rerank_seconds = rerank_candidates(
            normalized_queries[query_index],
            normalized_documents,
            selected,
        )
        total_rerank_seconds += rerank_seconds
        selected_doc_ids = {doc_ids[index] for index in selected}
        pool_doc_ids = {doc_ids[index] for index in features["candidate_doc_indices"]}
        reranked_doc_ids = [doc_ids[index] for index in reranked]
        rankings_by_query[query_id] = reranked_doc_ids
        query_results.append(
            {
                "query_id": query_id,
                "candidate_pool_size": len(features["candidate_doc_indices"]),
                "selected_candidate_count": len(selected),
                "comparison_to_exact": compare_exact(
                    exact_rankings[query_id],
                    reranked_doc_ids,
                    selected_doc_ids,
                    pool_doc_ids,
                ),
            }
        )

    reranked_comparisons = sum(result["selected_candidate_count"] for result in query_results)
    return {
        "policy": policy,
        "nprobe": nprobe,
        "top_l": top_l,
        "top_c": top_c,
        "total_rerank_seconds": total_rerank_seconds,
        "reranked_comparisons": reranked_comparisons,
        "reranked_comparison_ratio": reranked_comparisons / full_exact_comparisons,
        "mean_candidate_pool_size": float(np.mean([q["candidate_pool_size"] for q in query_results])),
        "mean_selected_candidate_count": reranked_comparisons / len(query_results),
        "mean_exact_recall_at_k": {
            str(k): float(
                np.mean([q["comparison_to_exact"]["exact_recall_at_k"][str(k)] for q in query_results])
            )
            for k in REPORT_K_VALUES
        },
        "exact_top1_match_count": sum(q["comparison_to_exact"]["exact_top1_match"] for q in query_results),
        "exact_top5_all_recovered_count": sum(
            q["comparison_to_exact"]["exact_top5_all_recovered"] for q in query_results
        ),
        "exact_top10_all_recovered_count": sum(
            q["comparison_to_exact"]["exact_top10_all_recovered"] for q in query_results
        ),
        "selector_loss": {
            "5": {
                state: sum(q["comparison_to_exact"]["selector_loss_at_5"] == state for q in query_results)
                for state in ("pool_coverage_failure", "selector_failure", "none")
            },
            "10": {
                state: sum(q["comparison_to_exact"]["selector_loss_at_10"] == state for q in query_results)
                for state in ("pool_coverage_failure", "selector_failure", "none")
            },
        },
        "qrels_metrics": compute_qrels_metrics(rankings_by_query, qrels, k_values=(1, 3, 5, 10), mrr_k=10),
    }


def pool_coverage(
    generated: list[dict[str, Any]],
    exact_rankings: dict[str, list[str]],
    doc_ids: list[str],
) -> dict[str, Any]:
    top5_hits = 0
    top10_hits = 0
    pool_sizes = []
    for item in generated:
        pool = {doc_ids[index] for index in item["features"]["candidate_doc_indices"]}
        pool_sizes.append(len(pool))
        reference = exact_rankings[item["query_id"]]
        top5_hits += int(set(reference[:5]).issubset(pool))
        top10_hits += int(set(reference[:10]).issubset(pool))
    n = len(generated)
    return {
        "mean_candidate_pool_size": float(np.mean(pool_sizes)),
        "pool_recall5_all": top5_hits / n,
        "pool_recall10_all": top10_hits / n,
        "pool_top5_all_count": top5_hits,
        "pool_top10_all_count": top10_hits,
    }


def main() -> None:
    index_cls, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")
    if not EXACT_PATH.exists():
        raise SystemExit("Run 21_scifact_colbert_maxsim_numpy.py before this script.")

    with EXACT_PATH.open("r", encoding="utf-8") as file:
        exact = json.load(file)
    with (CORPUS_DIR / "qrels.json").open("r", encoding="utf-8") as file:
        qrels = json.load(file)

    document_pack = load_packed_embeddings(EMBEDDINGS_DIR / "documents_packed.npz")
    query_pack = load_packed_embeddings(EMBEDDINGS_DIR / "queries_packed.npz")
    documents = unpack_embeddings(document_pack["values"], document_pack["offsets"])
    queries = unpack_embeddings(query_pack["values"], query_pack["offsets"])
    normalized_documents = [l2_normalize(document) for document in documents]
    normalized_queries = [l2_normalize(query) for query in queries]
    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(normalized_documents)
    doc_ids = document_pack["ids"]
    query_ids = query_pack["ids"]
    exact_rankings = {result["query_id"]: result["ranking_doc_ids"] for result in exact["query_results"]}

    np.random.seed(RANDOM_SEED)
    nbuckets = 2 * math.ceil(math.sqrt(flat_tokens.shape[0]))
    index, build_info = build_ivf_index(index_cls, flat_tokens, nbuckets)

    full_exact_comparisons = len(normalized_queries) * len(normalized_documents)
    exact_reference_seconds = float(exact.get("elapsed_seconds", exact.get("runtime_seconds", 0.0)))

    runs = []
    for nprobe in NPROBE_VALUES:
        nprobe_runs = []
        for top_l in L_VALUES:
            generated = []
            generation_seconds = 0.0
            retrieved_tokens = 0
            for query_index, query_matrix in enumerate(normalized_queries):
                hits, seconds = retrieve_query_token_hits(index, query_matrix, top_l, nprobe)
                features = build_candidate_features(hits, token_to_doc_index, len(doc_ids))
                generated.append(
                    {"query_id": query_ids[query_index], "query_index": query_index, "features": features}
                )
                generation_seconds += seconds
                retrieved_tokens += features["retrieved_token_vectors"]

            coverage = pool_coverage(generated, exact_rankings, doc_ids)
            policy_results = [
                summarize_policy(
                    policy=policy,
                    nprobe=nprobe,
                    top_l=top_l,
                    top_c=top_c,
                    generated=generated,
                    normalized_queries=normalized_queries,
                    normalized_documents=normalized_documents,
                    exact_rankings=exact_rankings,
                    doc_ids=doc_ids,
                    qrels=qrels,
                    full_exact_comparisons=full_exact_comparisons,
                )
                for policy in POLICIES
                for top_c in C_VALUES
            ]
            nprobe_runs.append(
                {
                    "nprobe": nprobe,
                    "top_l": top_l,
                    "candidate_generation_seconds": generation_seconds,
                    "total_retrieved_token_vectors": retrieved_tokens,
                    "pool_coverage": coverage,
                    "policy_results": policy_results,
                }
            )
            print(
                f"nprobe={nprobe if nprobe else 'ALL'} L={top_l}: "
                f"gen={generation_seconds:.4f}s pool={coverage['mean_candidate_pool_size']:.1f} "
                f"pool_rec5={coverage['pool_recall5_all']:.4f} pool_rec10={coverage['pool_recall10_all']:.4f}"
            )
        runs.append({"nprobe": nprobe, "l_runs": nprobe_runs})

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "candidate_generation": "ivf_bond_l2sq_on_l2_normalized_vectors",
            "rerank": "exact_normalized_inner_product_maxsim",
            "nprobe_zero_meaning": "explore all buckets (exact)",
        },
        "timing_note": (
            "Component timing on the current Python/WSL implementation. "
            "Not a controlled end-to-end speedup claim."
        ),
        "dataset": {
            "documents": len(doc_ids),
            "queries": len(normalized_queries),
            "total_document_token_vectors": int(flat_tokens.shape[0]),
            "total_query_token_vectors": int(sum(len(q) for q in normalized_queries)),
            "embedding_dimension": int(flat_tokens.shape[1]),
        },
        "exact_reference": {
            "seconds": exact_reference_seconds,
            "full_exact_document_comparisons": full_exact_comparisons,
        },
        "ivf_index": build_info,
        "nprobe_values": NPROBE_VALUES,
        "l_values": L_VALUES,
        "c_values": C_VALUES,
        "policies": POLICIES,
        "runs": runs,
    }
    save_json(OUTPUT_PATH, output)

    print("\nSciFact IVF-on-PDX candidate generation + exact rerank")
    print(f"documents: {len(doc_ids)}  queries: {len(normalized_queries)}")
    print(f"flat token matrix: {tuple(flat_tokens.shape)}  nbuckets: {nbuckets}")
    print(
        f"index build: train={build_info['train_seconds']:.4f}s "
        f"load={build_info['load_seconds']:.4f}s"
    )
    print(f"exact NumPy MaxSim reference: {exact_reference_seconds:.4f}s")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
