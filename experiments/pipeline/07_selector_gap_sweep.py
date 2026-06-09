"""Selector-gap sweep: re-rank budget C and selection policy vs quality/latency.

Motivation: on both SciFact-full and NFCorpus the candidate POOL contains the
exact top-10 for ~94-100% of queries, yet the reranked top-10 agreement with
exact MaxSim plateaus around 0.84-0.86. The pool is fine; the fixed C=50
selection into the re-rank is what loses documents. This script quantifies how
much budget (C) and which selection policy is needed to close that gap, and at
what latency cost.

Method: candidate generation is FIXED (FAISS-IVF over flat token vectors,
nprobe/L from the headline config) and computed once per query. Then we sweep
(policy x C), re-ranking the selected candidates with exact MaxSim, measuring:
  - exact agreement@10 (top-10 overlap with full exact MaxSim ranking)
  - qrels recall@10
  - re-rank seconds (candidate-gen seconds are constant and reported once)

C=0 means "re-rank the entire candidate pool" (upper bound for the pool).

Runs in the Windows venv (faiss-cpu, no PDX needed).

Example
-------
    .\\.venv\\Scripts\\python.exe experiments/35_selector_gap_sweep.py \
        --corpus-dir artifacts/nfcorpus_benchmark \
        --embeddings-dir artifacts/nfcorpus_benchmark_embeddings \
        --output artifacts/results/nfcorpus_selector_gap_sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PROJECT_ROOT, setup_imports

setup_imports()
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

RANDOM_SEED = 0
TRAINING_POINTS_PER_BUCKET = 50
POLICIES = ["approx_score", "retrieved_token_count", "union_approx_and_count"]
DEFAULT_C_VALUES = [20, 50, 100, 200, 400, 0]  # 0 = entire pool


def compute_exact(queries, documents, query_ids, doc_ids, qrels) -> dict[str, Any]:
    start = perf_counter()
    rankings_by_query = {}
    for query_id, query_matrix in zip(query_ids, queries):
        scores = np.empty(len(documents), dtype=np.float32)
        for doc_index, document_matrix in enumerate(documents):
            scores[doc_index] = maxsim_score(query_matrix, document_matrix, normalize=False)
        rankings_by_query[query_id] = [doc_ids[i] for i in rank_documents(scores)]
    seconds = perf_counter() - start
    return {
        "seconds": seconds,
        "rankings_by_query": rankings_by_query,
        "qrels_metrics": compute_qrels_metrics(
            rankings_by_query, qrels, k_values=(1, 3, 5, 10), mrr_k=10
        )["macro"],
    }


def build_faiss_ivf(flat_tokens: np.ndarray, nbuckets: int):
    import faiss

    faiss.omp_set_num_threads(1)
    dim = flat_tokens.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nbuckets, faiss.METRIC_L2)
    rng = np.random.default_rng(RANDOM_SEED)
    training_points = min(nbuckets * TRAINING_POINTS_PER_BUCKET, flat_tokens.shape[0])
    sample_idx = rng.choice(flat_tokens.shape[0], size=training_points, replace=False)
    sample_idx.sort()
    index.train(np.ascontiguousarray(flat_tokens[sample_idx], dtype=np.float32))
    index.add(flat_tokens)
    return index


def generate_candidates(index, query_matrix, token_to_doc_index, num_documents, top_l, nprobe):
    """One FAISS search per query; returns per-doc features for selection."""
    index.nprobe = nprobe
    query_block = np.ascontiguousarray(query_matrix, dtype=np.float32)
    t0 = perf_counter()
    distances, indices = index.search(query_block, top_l)
    generation_seconds = perf_counter() - t0

    per_token_doc_best = np.full((len(query_matrix), num_documents), -np.inf, dtype=np.float32)
    retrieved_token_count = np.zeros(num_documents, dtype=np.int32)
    candidate_doc_indices: set[int] = set()
    for token_row in range(indices.shape[0]):
        for hit_col in range(indices.shape[1]):
            token_index = int(indices[token_row, hit_col])
            if token_index < 0:
                continue
            similarity = 1.0 - float(distances[token_row, hit_col]) / 2.0
            doc_index = int(token_to_doc_index[token_index])
            candidate_doc_indices.add(doc_index)
            retrieved_token_count[doc_index] += 1
            if similarity > per_token_doc_best[token_row, doc_index]:
                per_token_doc_best[token_row, doc_index] = similarity
    matched_mask = np.isfinite(per_token_doc_best)
    approx_scores = np.where(matched_mask, per_token_doc_best, 0.0).sum(axis=0)
    return {
        "candidate_doc_indices": candidate_doc_indices,
        "approx_scores": approx_scores.astype(np.float32),
        "retrieved_token_count": retrieved_token_count,
        "matched_query_token_count": matched_mask.sum(axis=0).astype(np.int32),
        "generation_seconds": generation_seconds,
    }


def sort_by_approx(features):
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (-float(features["approx_scores"][doc]), doc),
    )


def sort_by_count(features):
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (
            -int(features["retrieved_token_count"][doc]),
            -float(features["approx_scores"][doc]),
            doc,
        ),
    )


def select_candidates(features, top_c, policy):
    pool_size = len(features["candidate_doc_indices"])
    if top_c <= 0 or top_c >= pool_size:
        return list(features["candidate_doc_indices"])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--embeddings-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--top-l", type=int, default=100)
    parser.add_argument("--c-values", nargs="+", type=int, default=DEFAULT_C_VALUES)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    corpus_dir = resolve(args.corpus_dir)
    embeddings_dir = resolve(args.embeddings_dir)
    output_path = resolve(args.output)
    dataset_name = corpus_dir.name.removesuffix("_benchmark") or corpus_dir.name

    with (corpus_dir / "qrels.json").open("r", encoding="utf-8") as file:
        qrels = json.load(file)
    document_pack = load_packed_embeddings(embeddings_dir / "documents_packed.npz")
    query_pack = load_packed_embeddings(embeddings_dir / "queries_packed.npz")
    documents = [l2_normalize(d) for d in unpack_embeddings(document_pack["values"], document_pack["offsets"])]
    queries = [l2_normalize(q) for q in unpack_embeddings(query_pack["values"], query_pack["offsets"])]
    doc_ids = document_pack["ids"]
    query_ids = query_pack["ids"]
    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(documents)

    print(f"dataset={dataset_name} documents={len(doc_ids)} queries={len(queries)} tokens={flat_tokens.shape[0]}")

    print("Exact NumPy MaxSim reference...")
    exact = compute_exact(queries, documents, query_ids, doc_ids, qrels)
    exact_rankings = exact["rankings_by_query"]
    print(f"  exact: {exact['seconds']:.3f}s qrels_recall@10={exact['qrels_metrics']['recall@10']:.4f}")

    nbuckets = 2 * math.ceil(math.sqrt(flat_tokens.shape[0]))
    print(f"Building FAISS-IVF (nbuckets={nbuckets})...")
    index = build_faiss_ivf(flat_tokens, nbuckets)

    # Fixed candidate generation, computed once.
    print(f"Generating candidates (nprobe={args.nprobe}, L={args.top_l})...")
    per_query_features = []
    generation_seconds = 0.0
    pool_sizes = []
    pool_recall10_all = 0
    for query_index, query_matrix in enumerate(queries):
        features = generate_candidates(
            index, query_matrix, token_to_doc_index, len(doc_ids), args.top_l, args.nprobe
        )
        generation_seconds += features["generation_seconds"]
        per_query_features.append(features)
        pool = {doc_ids[i] for i in features["candidate_doc_indices"]}
        pool_sizes.append(len(pool))
        reference = exact_rankings[query_ids[query_index]]
        pool_recall10_all += int(set(reference[:10]).issubset(pool))
    mean_pool_size = float(np.mean(pool_sizes))
    pool_coverage10 = pool_recall10_all / len(queries)
    print(
        f"  gen={generation_seconds:.3f}s mean_pool={mean_pool_size:.1f} "
        f"pool_coverage@10={pool_coverage10:.3f}"
    )

    sweep = []
    for policy in POLICIES:
        for top_c in args.c_values:
            rerank_seconds = 0.0
            agreement10, agreement5 = [], []
            selected_counts = []
            rankings_by_query = {}
            for query_index, features in enumerate(per_query_features):
                selected = select_candidates(features, top_c, policy)
                selected_counts.append(len(selected))
                t0 = perf_counter()
                scores = maxsim_scores_for_candidate_docs(
                    queries[query_index], documents, selected, normalize=False
                )
                rerank_seconds += perf_counter() - t0
                ordered = list(scores)
                score_arr = np.array([scores[i] for i in ordered], dtype=np.float32)
                reranked = [doc_ids[ordered[i]] for i in rank_documents(score_arr)]
                query_id = query_ids[query_index]
                rankings_by_query[query_id] = reranked
                reference = exact_rankings[query_id]
                agreement5.append(topk_overlap(reference, reranked, 5)["ratio"])
                agreement10.append(topk_overlap(reference, reranked, 10)["ratio"])
            qrels_metrics = compute_qrels_metrics(
                rankings_by_query, qrels, k_values=(1, 3, 5, 10), mrr_k=10
            )["macro"]
            row = {
                "policy": policy,
                "top_c": top_c,
                "effective_c_label": "pool" if top_c <= 0 else str(top_c),
                "mean_selected_candidates": float(np.mean(selected_counts)),
                "rerank_seconds": rerank_seconds,
                "total_seconds": generation_seconds + rerank_seconds,
                "agreement@5": float(np.mean(agreement5)),
                "agreement@10": float(np.mean(agreement10)),
                "qrels_metrics": qrels_metrics,
            }
            sweep.append(row)
            print(
                f"  {policy:<26} C={row['effective_c_label']:>4}: "
                f"sel={row['mean_selected_candidates']:6.1f} "
                f"rerank={rerank_seconds:6.3f}s total={row['total_seconds']:6.3f}s "
                f"agree@10={row['agreement@10']:.3f} "
                f"qrelsR@10={qrels_metrics['recall@10']:.4f}"
            )

    output = {
        "dataset": dataset_name,
        "config": {
            "nprobe": args.nprobe,
            "top_l": args.top_l,
            "c_values": args.c_values,
            "policies": POLICIES,
            "engine": "faiss_ivf_l2_on_l2_normalized_vectors",
            "note": "C=0 means re-rank the entire candidate pool.",
        },
        "dataset_stats": {
            "documents": len(doc_ids),
            "queries": len(queries),
            "total_document_token_vectors": int(flat_tokens.shape[0]),
        },
        "exact_reference": {
            "seconds": exact["seconds"],
            "qrels_metrics": exact["qrels_metrics"],
        },
        "candidate_generation": {
            "seconds": generation_seconds,
            "mean_pool_size": mean_pool_size,
            "pool_coverage@10": pool_coverage10,
        },
        "sweep": sweep,
    }
    save_json(output_path, output)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
