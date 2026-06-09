"""End-to-end baselines for the two-stage IVF-on-PDX + exact-rerank pipeline.

Runs three reference systems on the SAME prepared+encoded BEIR corpus so the
PDX-IVF numbers (measured in `31_beir_ivf_benchmark.py`) can be positioned
against established alternatives instead of only against single-threaded NumPy:

  1. Exact NumPy MaxSim          - quality ceiling + slow reference.
  2. PLAID (PyLate / fast-plaid) - the established ColBERT ANN engine (SOTA).
  3. FAISS-IVF + exact rerank    - same two-stage recipe as PDX, different
                                   engine; isolates "IVF idea" vs "PDX scan".

The ColBERT token embeddings are already L2-normalized, so l2/IP/cosine
orderings coincide; PLAID consumes the raw per-document arrays while the exact
and FAISS paths use the same normalized vectors as the PDX pipeline.

Runs in the Windows venv (no PDX needed): faiss-cpu + pylate are installed
there. The PDX-IVF row is read from the saved PDX benchmark JSON for the table.

Example
-------
    .\\.venv\\Scripts\\python.exe experiments/32_baselines_full_scifact.py \
        --corpus-dir artifacts/scifact_full_benchmark \
        --embeddings-dir artifacts/scifact_full_benchmark_embeddings \
        --pdx-result artifacts/results/scifact_full_ivf_benchmark.json \
        --output artifacts/results/scifact_full_baselines.json
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
REPORT_K_VALUES = (1, 3, 5, 10)
TRAINING_POINTS_PER_BUCKET = 50


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------
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
            rankings_by_query, qrels, k_values=REPORT_K_VALUES, mrr_k=10
        )["macro"],
    }


def exact_agreement(rankings_by_query, exact_rankings, k=10) -> float:
    ratios = []
    for query_id, reference in exact_rankings.items():
        candidate = rankings_by_query.get(query_id, [])
        ratios.append(topk_overlap(reference, candidate, k)["ratio"])
    return float(np.mean(ratios)) if ratios else 0.0


# ---------------------------------------------------------------------------
# FAISS-IVF + exact rerank (same two-stage recipe as PDX-IVF)
# ---------------------------------------------------------------------------
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
    training_sample = np.ascontiguousarray(flat_tokens[sample_idx], dtype=np.float32)
    t0 = perf_counter()
    index.train(training_sample)
    train_seconds = perf_counter() - t0
    t0 = perf_counter()
    index.add(flat_tokens)
    load_seconds = perf_counter() - t0
    return index, {
        "nbuckets": nbuckets,
        "training_points": int(training_points),
        "train_seconds": train_seconds,
        "load_seconds": load_seconds,
        "build_seconds": train_seconds + load_seconds,
    }


def faiss_ivf_pipeline(
    index,
    queries,
    documents,
    query_ids,
    doc_ids,
    token_to_doc_index,
    exact_rankings,
    qrels,
    *,
    nprobe: int,
    top_l: int,
    top_c: int,
) -> dict[str, Any]:
    index.nprobe = nprobe
    num_documents = len(doc_ids)
    rankings_by_query = {}
    pool_recall10_all = 0
    pool_sizes = []
    generation_seconds = 0.0
    rerank_seconds = 0.0

    for query_index, query_matrix in enumerate(queries):
        query_block = np.ascontiguousarray(query_matrix, dtype=np.float32)
        t0 = perf_counter()
        distances, indices = index.search(query_block, top_l)
        generation_seconds += perf_counter() - t0

        per_token_doc_best = np.full((len(query_matrix), num_documents), -np.inf, dtype=np.float32)
        candidate_doc_indices: set[int] = set()
        for token_row in range(indices.shape[0]):
            for hit_col in range(indices.shape[1]):
                token_index = int(indices[token_row, hit_col])
                if token_index < 0:
                    continue
                similarity = 1.0 - float(distances[token_row, hit_col]) / 2.0
                doc_index = int(token_to_doc_index[token_index])
                candidate_doc_indices.add(doc_index)
                if similarity > per_token_doc_best[token_row, doc_index]:
                    per_token_doc_best[token_row, doc_index] = similarity
        approx_scores = np.where(np.isfinite(per_token_doc_best), per_token_doc_best, 0.0).sum(axis=0)

        selected = sorted(
            candidate_doc_indices,
            key=lambda doc: (-float(approx_scores[doc]), doc),
        )[:top_c]

        t0 = perf_counter()
        scores = maxsim_scores_for_candidate_docs(
            queries[query_index], documents, selected, normalize=False
        )
        rerank_seconds += perf_counter() - t0
        ordered = list(scores)
        score_arr = np.array([scores[i] for i in ordered], dtype=np.float32)
        reranked = [doc_ids[ordered[i]] for i in rank_documents(score_arr)]
        rankings_by_query[query_ids[query_index]] = reranked

        pool = {doc_ids[i] for i in candidate_doc_indices}
        pool_sizes.append(len(pool))
        reference = exact_rankings[query_ids[query_index]]
        pool_recall10_all += int(set(reference[:10]).issubset(pool))

    return {
        "nprobe": nprobe,
        "top_l": top_l,
        "top_c": top_c,
        "candidate_generation_seconds": generation_seconds,
        "rerank_seconds": rerank_seconds,
        "total_seconds": generation_seconds + rerank_seconds,
        "mean_candidate_pool_size": float(np.mean(pool_sizes)),
        "pool_recall10_all": pool_recall10_all / len(queries),
        "exact_agreement_recall@10": exact_agreement(rankings_by_query, exact_rankings, 10),
        "qrels_metrics": compute_qrels_metrics(
            rankings_by_query, qrels, k_values=REPORT_K_VALUES, mrr_k=10
        )["macro"],
    }


# ---------------------------------------------------------------------------
# PLAID (PyLate / fast-plaid)
# ---------------------------------------------------------------------------
def run_plaid(
    raw_documents,
    raw_queries,
    query_ids,
    doc_ids,
    exact_rankings,
    qrels,
    *,
    index_folder: Path,
    k: int,
    n_ivf_probe: int,
    n_full_scores: int,
    nbits: int,
    index_name: str = "baseline",
) -> dict[str, Any]:
    from pylate import indexes, retrieve

    index = indexes.PLAID(
        index_folder=str(index_folder),
        index_name=index_name,
        override=True,
        nbits=nbits,
        kmeans_niters=4,
        n_ivf_probe=n_ivf_probe,
        n_full_scores=n_full_scores,
        show_progress=False,
        device="cpu",
        use_triton=False,
    )
    t0 = perf_counter()
    index.add_documents(documents_ids=doc_ids, documents_embeddings=raw_documents)
    build_seconds = perf_counter() - t0

    retriever = retrieve.ColBERT(index=index)
    t0 = perf_counter()
    results = retriever.retrieve(queries_embeddings=raw_queries, k=k, device="cpu")
    retrieve_seconds = perf_counter() - t0

    rankings_by_query = {}
    for query_id, hits in zip(query_ids, results):
        rankings_by_query[query_id] = [str(hit["id"]) for hit in hits]

    return {
        "nbits": nbits,
        "n_ivf_probe": n_ivf_probe,
        "n_full_scores": n_full_scores,
        "k": k,
        "build_seconds": build_seconds,
        "retrieve_seconds": retrieve_seconds,
        "exact_agreement_recall@10": exact_agreement(rankings_by_query, exact_rankings, 10),
        "qrels_metrics": compute_qrels_metrics(
            rankings_by_query, qrels, k_values=REPORT_K_VALUES, mrr_k=10
        )["macro"],
    }


def load_pdx_best(pdx_result_path: Path) -> dict[str, Any] | None:
    if not pdx_result_path.exists():
        return None
    data = json.loads(pdx_result_path.read_text(encoding="utf-8"))
    best = None
    for run in data.get("runs", []):
        if run.get("nprobe", 0) == 0:
            continue
        best_c50 = run.get("best_rerank", {}).get("best_c50_recall10") or {}
        qrels_recall10 = best_c50.get("qrels_metrics", {}).get("recall@10", 0.0)
        gen = run.get("candidate_generation_seconds", float("inf"))
        rerank = best_c50.get("rerank_seconds", 0.0)
        candidate = {
            "nprobe": run["nprobe"],
            "top_l": run["top_l"],
            "candidate_generation_seconds": gen,
            "rerank_seconds": rerank,
            "total_seconds": gen + rerank,
            "pool_recall10_all": run.get("pool_coverage", {}).get("pool_recall10_all"),
            "qrels_recall@10": qrels_recall10,
        }
        if best is None or (
            qrels_recall10 > best["qrels_recall@10"]
            or (qrels_recall10 == best["qrels_recall@10"] and gen < best["candidate_generation_seconds"])
        ):
            best = candidate
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--embeddings-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pdx-result", type=Path, default=None)
    parser.add_argument("--faiss-nprobe", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--faiss-l", type=int, default=100)
    parser.add_argument("--faiss-c", type=int, default=50)
    parser.add_argument("--plaid-nbits", type=int, default=4)
    parser.add_argument("--plaid-nprobe", type=int, default=8)
    parser.add_argument("--plaid-full-scores", type=int, default=8192)
    parser.add_argument("--plaid-k", type=int, default=100)
    parser.add_argument("--skip-plaid", action="store_true")
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

    raw_documents = unpack_embeddings(document_pack["values"], document_pack["offsets"])
    raw_queries = unpack_embeddings(query_pack["values"], query_pack["offsets"])
    documents = [l2_normalize(d) for d in raw_documents]
    queries = [l2_normalize(q) for q in raw_queries]
    doc_ids = document_pack["ids"]
    query_ids = query_pack["ids"]
    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(documents)

    print(
        f"documents={len(doc_ids)} queries={len(queries)} "
        f"doc_token_vectors={flat_tokens.shape[0]} dim={flat_tokens.shape[1]}"
    )

    print("\n[1/3] Exact NumPy MaxSim reference...")
    exact = compute_exact(queries, documents, query_ids, doc_ids, qrels)
    exact_rankings = exact["rankings_by_query"]
    print(f"  exact: {exact['seconds']:.3f}s  qrels_recall@10={exact['qrels_metrics']['recall@10']:.4f}")

    print("\n[2/3] FAISS-IVF + exact rerank...")
    nbuckets = 2 * math.ceil(math.sqrt(flat_tokens.shape[0]))
    faiss_index, faiss_build = build_faiss_ivf(flat_tokens, nbuckets)
    print(
        f"  build: nbuckets={nbuckets} train={faiss_build['train_seconds']:.3f}s "
        f"load={faiss_build['load_seconds']:.3f}s"
    )
    faiss_runs = []
    for nprobe in args.faiss_nprobe:
        run = faiss_ivf_pipeline(
            faiss_index, queries, documents, query_ids, doc_ids, token_to_doc_index,
            exact_rankings, qrels, nprobe=nprobe, top_l=args.faiss_l, top_c=args.faiss_c,
        )
        faiss_runs.append(run)
        print(
            f"  nprobe={nprobe:>2} L={args.faiss_l} C={args.faiss_c}: "
            f"gen={run['candidate_generation_seconds']:.3f}s "
            f"rerank={run['rerank_seconds']:.3f}s "
            f"total={run['total_seconds']:.3f}s "
            f"speedup={exact['seconds'] / run['total_seconds']:6.2f}x "
            f"poolR10={run['pool_recall10_all']:.3f} "
            f"agree@10={run['exact_agreement_recall@10']:.3f} "
            f"qrels_recall@10={run['qrels_metrics']['recall@10']:.4f}"
        )

    plaid_result = None
    if not args.skip_plaid:
        print("\n[3/3] PLAID (PyLate / fast-plaid)...")
        plaid_folder = resolve(Path("artifacts") / "baseline_plaid_index")
        plaid_result = run_plaid(
            raw_documents, raw_queries, query_ids, doc_ids, exact_rankings, qrels,
            index_folder=plaid_folder, k=args.plaid_k, n_ivf_probe=args.plaid_nprobe,
            n_full_scores=args.plaid_full_scores, nbits=args.plaid_nbits,
            index_name=f"{dataset_name}_baseline",
        )
        print(
            f"  build={plaid_result['build_seconds']:.3f}s "
            f"retrieve={plaid_result['retrieve_seconds']:.3f}s "
            f"speedup={exact['seconds'] / plaid_result['retrieve_seconds']:6.2f}x "
            f"agree@10={plaid_result['exact_agreement_recall@10']:.3f} "
            f"qrels_recall@10={plaid_result['qrels_metrics']['recall@10']:.4f}"
        )

    pdx_best = load_pdx_best(resolve(args.pdx_result)) if args.pdx_result else None

    output = {
        "note": (
            "End-to-end baseline comparison on a single machine (Windows venv). "
            "Exact/FAISS/PLAID measured here; PDX-IVF row read from the saved WSL "
            "benchmark (different machine/env) for reference, so absolute PDX times "
            "are not directly comparable to the FAISS/PLAID times measured here."
        ),
        "embeddings_normalized": True,
        "dataset": {
            "documents": len(doc_ids),
            "queries": len(queries),
            "total_document_token_vectors": int(flat_tokens.shape[0]),
            "embedding_dimension": int(flat_tokens.shape[1]),
        },
        "exact_reference": {
            "seconds": exact["seconds"],
            "qrels_metrics": exact["qrels_metrics"],
        },
        "faiss_ivf": {"build": faiss_build, "runs": faiss_runs},
        "plaid": plaid_result,
        "pdx_ivf_best_from_saved_result": pdx_best,
    }
    save_json(output_path, output)
    print(f"\nSaved: {output_path}")

    # Compact comparison table (quality is the directly comparable axis).
    print(f"\n=== Quality / latency summary ({dataset_name}) ===")
    print(f"{'method':<34}{'time(50q)':>12}{'qrels R@10':>12}{'agree@10':>10}")
    print(f"{'exact NumPy MaxSim':<34}{exact['seconds']:>11.3f}s{exact['qrels_metrics']['recall@10']:>12.4f}{1.0:>10.3f}")
    for run in faiss_runs:
        label = f"FAISS-IVF (np={run['nprobe']},C={run['top_c']})"
        print(
            f"{label:<34}{run['total_seconds']:>11.3f}s"
            f"{run['qrels_metrics']['recall@10']:>12.4f}{run['exact_agreement_recall@10']:>10.3f}"
        )
    if plaid_result:
        print(
            f"{'PLAID (np=' + str(args.plaid_nprobe) + ',nbits=' + str(args.plaid_nbits) + ')':<34}"
            f"{plaid_result['retrieve_seconds']:>11.3f}s"
            f"{plaid_result['qrels_metrics']['recall@10']:>12.4f}{plaid_result['exact_agreement_recall@10']:>10.3f}"
        )
    if pdx_best:
        print(
            f"{'PDX-IVF (np=' + str(pdx_best['nprobe']) + ',C=50) [WSL]':<34}"
            f"{pdx_best['total_seconds']:>11.3f}s{pdx_best['qrels_recall@10']:>12.4f}{'-':>10}"
        )


if __name__ == "__main__":
    main()
