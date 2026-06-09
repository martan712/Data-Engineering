"""Exact MaxSim vs IVF-on-PDX candidate generation for any BEIR benchmark.

Generalizes experiments 27/28 to run on an arbitrary prepared+encoded corpus
(via --corpus-dir / --embeddings-dir). Computes the exact NumPy MaxSim baseline
inline, builds an `IndexPDXBONDIVFFlat` index, runs an nprobe x L sweep, and
reports candidate-generation speedup, pool coverage, reranked exact recall, and
qrels metrics (exact baseline vs best IVF config).

Used for the CoRECT/BEIR scale-up. Run from the WSL PDX environment.

Example
-------
    python 31_beir_ivf_benchmark.py \
        --corpus-dir artifacts/scifact_full_benchmark \
        --embeddings-dir artifacts/scifact_full_benchmark_embeddings \
        --output artifacts/results/scifact_full_ivf_benchmark.json
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPROBE_VALUES = [0, 8, 16, 32]
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


def compute_exact(
    queries: list[np.ndarray],
    documents: list[np.ndarray],
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, list[str]],
) -> dict[str, Any]:
    start = perf_counter()
    rankings_by_query = {}
    for query_id, query_matrix in zip(query_ids, queries):
        scores = np.empty(len(documents), dtype=np.float32)
        for doc_index, document_matrix in enumerate(documents):
            scores[doc_index] = maxsim_score(query_matrix, document_matrix, normalize=False)
        rankings_by_query[query_id] = [doc_ids[index] for index in rank_documents(scores)]
    seconds = perf_counter() - start
    return {
        "seconds": seconds,
        "comparisons": len(queries) * len(documents),
        "rankings_by_query": rankings_by_query,
        "qrels_metrics": compute_qrels_metrics(rankings_by_query, qrels, k_values=(1, 3, 5, 10), mrr_k=10)["macro"],
    }


def build_ivf_index(index_cls, flat_tokens: np.ndarray, nbuckets: int):
    rng = np.random.default_rng(RANDOM_SEED)
    training_points = min(nbuckets * TRAINING_POINTS_PER_BUCKET, flat_tokens.shape[0])
    sample_idx = rng.choice(flat_tokens.shape[0], size=training_points, replace=False)
    sample_idx.sort()
    training_sample = np.ascontiguousarray(flat_tokens[sample_idx], dtype=np.float32)
    index = index_cls(ndim=flat_tokens.shape[1], nbuckets=nbuckets)
    t0 = perf_counter()
    index.train(training_sample)
    train_seconds = perf_counter() - t0
    t0 = perf_counter()
    index.add_load(flat_tokens)
    load_seconds = perf_counter() - t0
    return index, {
        "nbuckets": nbuckets,
        "training_points": int(training_points),
        "train_seconds": train_seconds,
        "load_seconds": load_seconds,
        "build_seconds": train_seconds + load_seconds,
    }


def retrieve_query_token_hits(index, query_matrix: np.ndarray, top_l: int, nprobe: int):
    hits_per_token = []
    start = perf_counter()
    for query_token in query_matrix:
        hits = index.search(np.ascontiguousarray(query_token, dtype=np.float32), top_l, nprobe=nprobe)
        hits_per_token.append([(int(h.index), 1.0 - (float(h.distance) / 2.0)) for h in hits])
    return hits_per_token, perf_counter() - start


def build_candidate_features(query_token_hits, token_to_doc_index, num_documents):
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
    return {
        "candidate_doc_indices": candidate_doc_indices,
        "approx_scores": approx_scores.astype(np.float32),
        "retrieved_token_count": retrieved_token_count,
        "matched_query_token_count": matched_mask.sum(axis=0).astype(np.int32),
    }


def sort_by_approx(features):
    return sorted(features["candidate_doc_indices"], key=lambda doc: (-float(features["approx_scores"][doc]), doc))


def sort_by_count(features):
    return sorted(
        features["candidate_doc_indices"],
        key=lambda doc: (-int(features["retrieved_token_count"][doc]), -float(features["approx_scores"][doc]), doc),
    )


def select_candidates(features, top_c, policy):
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


def pool_coverage(generated, exact_rankings, doc_ids):
    top5_hits = top10_hits = 0
    pool_sizes, recall5, recall10 = [], [], []
    for item in generated:
        pool = {doc_ids[index] for index in item["features"]["candidate_doc_indices"]}
        pool_sizes.append(len(pool))
        reference = exact_rankings[item["query_id"]]
        oracle = [doc_id for doc_id in reference if doc_id in pool]
        top5_hits += int(set(reference[:5]).issubset(pool))
        top10_hits += int(set(reference[:10]).issubset(pool))
        recall5.append(topk_overlap(reference, oracle, 5)["ratio"])
        recall10.append(topk_overlap(reference, oracle, 10)["ratio"])
    n = len(generated)
    return {
        "mean_candidate_pool_size": float(np.mean(pool_sizes)),
        "pool_recall5_all": top5_hits / n,
        "pool_recall10_all": top10_hits / n,
        "pool_mean_recall5": float(np.mean(recall5)),
        "pool_mean_recall10": float(np.mean(recall10)),
    }


def rerank_all_policies(generated, queries, documents, exact_rankings, doc_ids, qrels, c_values):
    results = []
    for policy in POLICIES:
        for top_c in c_values:
            recall5, recall10 = [], []
            rerank_seconds = 0.0
            selected_counts = []
            rankings_by_query = {}
            for item in generated:
                selected = select_candidates(item["features"], top_c, policy)
                selected_counts.append(len(selected))
                t0 = perf_counter()
                scores = maxsim_scores_for_candidate_docs(
                    queries[item["query_index"]], documents, selected, normalize=False
                )
                rerank_seconds += perf_counter() - t0
                ordered = list(scores)
                score_arr = np.array([scores[i] for i in ordered], dtype=np.float32)
                reranked = [doc_ids[ordered[i]] for i in rank_documents(score_arr)]
                rankings_by_query[item["query_id"]] = reranked
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
                    "qrels_metrics": compute_qrels_metrics(
                        rankings_by_query, qrels, k_values=(1, 3, 5, 10), mrr_k=10
                    )["macro"],
                }
            )
    best_c20 = max((r for r in results if r["top_c"] == 20), key=lambda r: r["mean_exact_recall5"], default=None)
    best_c50 = max((r for r in results if r["top_c"] == 50), key=lambda r: r["mean_exact_recall10"], default=None)
    return {"best_c20_recall5": best_c20, "best_c50_recall10": best_c50, "all": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--embeddings-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nprobe-values", nargs="+", type=int, default=DEFAULT_NPROBE_VALUES)
    parser.add_argument("--l-values", nargs="+", type=int, default=DEFAULT_L_VALUES)
    parser.add_argument("--c-values", nargs="+", type=int, default=DEFAULT_C_VALUES)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    index_cls, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")

    corpus_dir = resolve(args.corpus_dir)
    embeddings_dir = resolve(args.embeddings_dir)
    output_path = resolve(args.output)

    with (corpus_dir / "qrels.json").open("r", encoding="utf-8") as file:
        qrels = json.load(file)
    document_pack = load_packed_embeddings(embeddings_dir / "documents_packed.npz")
    query_pack = load_packed_embeddings(embeddings_dir / "queries_packed.npz")
    documents = [l2_normalize(d) for d in unpack_embeddings(document_pack["values"], document_pack["offsets"])]
    queries = [l2_normalize(q) for q in unpack_embeddings(query_pack["values"], query_pack["offsets"])]
    doc_ids = document_pack["ids"]
    query_ids = query_pack["ids"]
    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(documents)

    print(f"documents={len(doc_ids)} queries={len(queries)} doc_token_vectors={flat_tokens.shape[0]}")
    print("Computing exact NumPy MaxSim baseline...")
    exact = compute_exact(queries, documents, query_ids, doc_ids, qrels)
    exact_rankings = exact["rankings_by_query"]
    print(f"exact MaxSim: {exact['seconds']:.4f}s  qrels={exact['qrels_metrics']}")

    np.random.seed(RANDOM_SEED)
    nbuckets = 2 * math.ceil(math.sqrt(flat_tokens.shape[0]))
    print(f"Building IVF index (nbuckets={nbuckets})...")
    index, build_info = build_ivf_index(index_cls, flat_tokens, nbuckets)
    print(f"IVF build: train={build_info['train_seconds']:.4f}s load={build_info['load_seconds']:.4f}s")

    runs = []
    for nprobe in args.nprobe_values:
        for top_l in args.l_values:
            generated = []
            generation_seconds = 0.0
            for query_index, query_matrix in enumerate(queries):
                hits, seconds = retrieve_query_token_hits(index, query_matrix, top_l, nprobe)
                features = build_candidate_features(hits, token_to_doc_index, len(doc_ids))
                generated.append(
                    {"query_id": query_ids[query_index], "query_index": query_index, "features": features}
                )
                generation_seconds += seconds
            coverage = pool_coverage(generated, exact_rankings, doc_ids)
            rerank = rerank_all_policies(
                generated, queries, documents, exact_rankings, doc_ids, qrels, args.c_values
            )
            speedup = exact["seconds"] / generation_seconds if generation_seconds else 0.0
            best50 = rerank["best_c50_recall10"]
            runs.append(
                {
                    "nprobe": nprobe,
                    "top_l": top_l,
                    "candidate_generation_seconds": generation_seconds,
                    "speedup_vs_exact": speedup,
                    "pool_coverage": coverage,
                    "best_rerank": rerank,
                }
            )
            print(
                f"  nprobe={nprobe if nprobe else 'ALL':>3} L={top_l:>3}: "
                f"gen={generation_seconds:.4f}s speedup={speedup:6.2f}x "
                f"poolR5={coverage['pool_recall5_all']:.3f} poolR10={coverage['pool_recall10_all']:.3f} "
                f"bestC50_rec@10={best50['mean_exact_recall10']:.4f} "
                f"qrels_rec@10={best50['qrels_metrics']['recall@10']:.4f}"
            )

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "candidate_generation": "ivf_bond_l2sq_on_l2_normalized_vectors",
            "rerank": "exact_normalized_inner_product_maxsim",
            "nprobe_zero_meaning": "explore all buckets (exact)",
        },
        "timing_note": "Component timing on the current Python/WSL implementation. Not a controlled end-to-end speedup claim.",
        "corpus_dir": str(corpus_dir),
        "embeddings_dir": str(embeddings_dir),
        "dataset": {
            "documents": len(doc_ids),
            "queries": len(queries),
            "total_document_token_vectors": int(flat_tokens.shape[0]),
            "total_query_token_vectors": int(sum(len(q) for q in queries)),
            "embedding_dimension": int(flat_tokens.shape[1]),
        },
        "exact_reference": {
            "seconds": exact["seconds"],
            "full_exact_document_comparisons": exact["comparisons"],
            "qrels_metrics": exact["qrels_metrics"],
        },
        "ivf_index": build_info,
        "nprobe_values": args.nprobe_values,
        "l_values": args.l_values,
        "c_values": args.c_values,
        "policies": POLICIES,
        "runs": runs,
    }
    save_json(output_path, output)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
