"""Instrumentation for a MaxSim-aware multi-vector BOND (the research kernel).

This is a *correct, measurable* NumPy reference of document-level branch-and-bound
for ColBERT MaxSim, used to decide whether a real (C++/PDX) kernel is worth
building and how it should be designed. It does NOT try to be fast; it counts the
work a real kernel would do.

Idea (multi-vector generalization of the 2002 BOND paper)
---------------------------------------------------------
For query token matrix Q = [q_1..q_m] and a document with token matrix D, the
MaxSim score is S = sum_i max_j <q_i, d_j>. We want the top-k *documents* by S.

Scan embedding dimensions in a fixed order. After a prefix P of dimensions, the
partial dot pp(i,j) = sum_{c in P} q_i[c] d_j[c] is known, and for unit-norm
ColBERT vectors the remainder is bounded by Cauchy-Schwarz:

    |<q_i,d_j> - pp(i,j)|  <=  sqrt(1 - ||q_i[P]||^2) * sqrt(1 - ||d_j[P]||^2)

So per document we get a running upper bound U and lower bound L on S. With a
shared vertical scan across all query tokens, maintain the k-th largest L as a
threshold tau; any document with U < tau cannot reach the top-k and is pruned
(branch-and-bound). U only decreases and tau only increases as dimensions are
scanned, so the procedure is monotonic and *provably never prunes a true top-k
document* (true S in [L, U], and tau <= k-th largest true S).

What we measure
---------------
- work_ratio: (doc-token-dimension products actually scanned) / (all of them).
  Counted ONCE per (doc token, dimension) regardless of m query tokens -- this is
  the shared-scan advantage. 1 / work_ratio is the ceiling speedup vs a naive
  multi-vector full scan.
- live-document fraction vs dimensions scanned (the pruning curve).
- dimensions needed to prune 50 / 90 / 99% of documents.
- correctness: number of true top-k documents wrongly pruned (must be 0).

Runs in the Windows venv (pure NumPy, no PDX/FAISS).

Example
-------
    .\\.venv\\Scripts\\python.exe experiments/33_maxsim_bond_instrumentation.py \
        --embeddings-dir artifacts/scifact_benchmark_embeddings \
        --k 10 --dim-order query_energy \
        --output artifacts/results/maxsim_bond_instrumentation_scifact1k.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import PROJECT_ROOT, setup_imports

setup_imports()
from utils_colbert import (
    flatten_document_embeddings,
    l2_normalize,
    load_packed_embeddings,
    save_json,
    unpack_embeddings,
)

DEFAULT_CHECKPOINTS = [8, 16, 24, 32, 48, 64, 96, 128]


def pca_rotation(flat_tokens: np.ndarray) -> np.ndarray:
    """Uncentered-PCA orthonormal rotation that concentrates token energy.

    Eigenvectors of the second-moment matrix T^T T, sorted by descending
    eigenvalue. Because the transform is orthonormal, inner products and norms
    (hence MaxSim and unit-norm) are preserved exactly; only the per-dimension
    energy distribution changes, which is what the Cauchy-Schwarz bound exploits.
    """
    moment = flat_tokens.T.astype(np.float64) @ flat_tokens.astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(moment)
    order = np.argsort(-eigvals)
    return eigvecs[:, order].astype(np.float32)


def dimension_order(flat_tokens: np.ndarray, queries: list[np.ndarray], mode: str) -> np.ndarray:
    """Return a global permutation of dimensions to scan in."""
    dim = flat_tokens.shape[1]
    if mode == "natural":
        return np.arange(dim)
    if mode == "query_energy":
        # High-energy query dimensions first -> ||q[P]|| grows fast -> tighter bound.
        energy = np.zeros(dim, dtype=np.float64)
        for q in queries:
            energy += (q.astype(np.float64) ** 2).sum(axis=0)
        return np.argsort(-energy).astype(np.int64)
    if mode == "doc_var":
        # Highest-variance document dimensions first.
        var = flat_tokens.astype(np.float64).var(axis=0)
        return np.argsort(-var).astype(np.int64)
    raise ValueError(f"Unknown dim-order mode: {mode}")


def exact_maxsim_scores(query: np.ndarray, flat_tokens: np.ndarray, doc_starts: np.ndarray) -> np.ndarray:
    """Exact per-document MaxSim score for one query (full dot + segment max)."""
    dots = flat_tokens @ query.T
    return np.maximum.reduceat(dots, doc_starts, axis=0).sum(axis=1)


def run_bond_for_query(
    query: np.ndarray,
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    num_documents: int,
    checkpoints: list[int],
    k: int,
    threshold_mode: str,
    seed_fraction: float,
) -> dict[str, Any]:
    """Simulate document-level MaxSim branch-and-bound for one query.

    `query` and `flat_tokens` must already be column-reordered to scan order.
    `doc_starts` are the per-document start offsets into `flat_tokens` (contiguous).

    threshold_mode controls the pruning threshold tau:
      - "bound" : tau = k-th largest running *lower* bound (self-contained BOND).
      - "oracle": tau = true k-th best MaxSim score, fixed from the start. Upper
                  bound on dimension-pruning potential given a perfect seed.
      - "seed"  : tau = k-th best exact score among the top `seed_fraction` of
                  documents ranked by the partial score at the FIRST checkpoint
                  (a realistic cheap seed; never exceeds the true k-th, so exact).
    """
    num_tokens, dim = flat_tokens.shape
    m = query.shape[0]
    exact_scores = exact_maxsim_scores(query, flat_tokens, doc_starts)
    exact_order = np.argsort(-exact_scores)
    exact_topk = set(exact_order[:k].tolist())
    oracle_threshold = float(exact_scores[exact_order[k - 1]]) if num_documents >= k else -np.inf

    partial_dot = np.zeros((num_tokens, m), dtype=np.float32)
    token_sqnorm = np.zeros(num_tokens, dtype=np.float64)
    query_sqnorm = np.zeros(m, dtype=np.float64)

    pruned = np.zeros(num_documents, dtype=bool)
    prune_dim = np.full(num_documents, dim, dtype=np.int32)

    live_curve = []
    seed_threshold = -np.inf
    prev = 0
    for cp_index, checkpoint in enumerate(checkpoints):
        cols = slice(prev, checkpoint)
        partial_dot += flat_tokens[:, cols] @ query[:, cols].T
        token_sqnorm += (flat_tokens[:, cols].astype(np.float64) ** 2).sum(axis=1)
        query_sqnorm += (query[:, cols].astype(np.float64) ** 2).sum(axis=1)
        prev = checkpoint

        remain_doc = np.sqrt(np.clip(1.0 - token_sqnorm, 0.0, None)).astype(np.float32)
        remain_query = np.sqrt(np.clip(1.0 - query_sqnorm, 0.0, None)).astype(np.float32)
        slack = remain_doc[:, None] * remain_query[None, :]
        upper_tokens = partial_dot + slack
        lower_tokens = partial_dot - slack

        max_upper = np.maximum.reduceat(upper_tokens, doc_starts, axis=0)
        max_lower = np.maximum.reduceat(lower_tokens, doc_starts, axis=0)
        upper_doc = max_upper.sum(axis=1)
        lower_doc = max_lower.sum(axis=1)

        if cp_index == 0 and threshold_mode == "seed":
            # Cheap seed: exact-score the top-fraction by the first-checkpoint
            # partial document score, then take their k-th best exact score.
            seed_count = max(k, int(np.ceil(num_documents * seed_fraction)))
            partial_doc_score = np.maximum.reduceat(partial_dot, doc_starts, axis=0).sum(axis=1)
            seed_docs = np.argsort(-partial_doc_score)[:seed_count]
            seed_scores = np.sort(exact_scores[seed_docs])[::-1]
            seed_threshold = float(seed_scores[k - 1]) if len(seed_scores) >= k else -np.inf

        if threshold_mode == "oracle":
            threshold = oracle_threshold
        elif threshold_mode == "seed":
            running = np.partition(lower_doc[~pruned], -k)[-k] if (~pruned).sum() > k else -np.inf
            threshold = max(seed_threshold, running)
        else:
            live_lb = lower_doc[~pruned]
            threshold = np.partition(live_lb, -k)[-k] if live_lb.size > k else -np.inf

        live = ~pruned
        if live.sum() > k and np.isfinite(threshold):
            newly_pruned = live & (upper_doc < threshold)
            prune_dim[newly_pruned] = checkpoint
            pruned |= newly_pruned
        live_curve.append({"dim": checkpoint, "live": int((~pruned).sum())})

    wrongly_pruned = sum(1 for doc in exact_topk if prune_dim[doc] < dim)
    return {
        "prune_dim": prune_dim,
        "live_curve": live_curve,
        "wrongly_pruned_topk": wrongly_pruned,
        "exact_topk": exact_topk,
    }


def summarize(per_query, num_documents, tokens_per_doc, total_tokens, dim, checkpoints, k):
    work_ratios = []
    live_fraction_by_dim = {c: [] for c in checkpoints}
    dims_to_prune = {"50%": [], "90%": [], "99%": []}
    total_wrong = 0

    for result in per_query:
        prune_dim = result["prune_dim"]
        work = float(np.sum(tokens_per_doc * prune_dim))
        work_ratios.append(work / (total_tokens * dim))
        total_wrong += result["wrongly_pruned_topk"]
        for point in result["live_curve"]:
            live_fraction_by_dim[point["dim"]].append(point["live"] / num_documents)
        for label, frac in (("50%", 0.5), ("90%", 0.9), ("99%", 0.99)):
            target_live = num_documents * (1.0 - frac)
            reached = next(
                (p["dim"] for p in result["live_curve"] if p["live"] <= target_live),
                dim,
            )
            dims_to_prune[label].append(reached)

    return {
        "k": k,
        "mean_work_ratio": float(np.mean(work_ratios)),
        "ceiling_speedup_vs_naive_full_scan": float(1.0 / np.mean(work_ratios)),
        "mean_live_fraction_by_dim": {
            str(c): float(np.mean(live_fraction_by_dim[c])) for c in checkpoints
        },
        "mean_dims_to_prune": {label: float(np.mean(v)) for label, v in dims_to_prune.items()},
        "true_topk_wrongly_pruned_total": int(total_wrong),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-documents", type=int, default=0, help="0 = all documents")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=DEFAULT_CHECKPOINTS)
    parser.add_argument(
        "--dim-order",
        choices=["query_energy", "natural", "doc_var"],
        default="query_energy",
    )
    parser.add_argument(
        "--threshold-modes",
        nargs="+",
        choices=["bound", "seed", "oracle"],
        default=["bound", "seed", "oracle"],
    )
    parser.add_argument("--seed-fraction", type=float, default=0.02)
    parser.add_argument("--rotation", choices=["none", "pca"], default="none")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    embeddings_dir = resolve(args.embeddings_dir)
    output_path = resolve(args.output)

    document_pack = load_packed_embeddings(embeddings_dir / "documents_packed.npz")
    query_pack = load_packed_embeddings(embeddings_dir / "queries_packed.npz")
    documents = [l2_normalize(d) for d in unpack_embeddings(document_pack["values"], document_pack["offsets"])]
    queries = [l2_normalize(q) for q in unpack_embeddings(query_pack["values"], query_pack["offsets"])]
    if args.max_documents and args.max_documents < len(documents):
        documents = documents[: args.max_documents]

    flat_tokens, token_to_doc_index, _ = flatten_document_embeddings(documents)
    num_documents = len(documents)
    dim = flat_tokens.shape[1]
    checkpoints = [c for c in args.checkpoints if c < dim] + [dim]

    if args.rotation == "pca":
        rotation = pca_rotation(flat_tokens)
        flat_tokens = np.ascontiguousarray(flat_tokens @ rotation)
        queries = [np.ascontiguousarray(q @ rotation) for q in queries]

    order = dimension_order(flat_tokens, queries, args.dim_order)
    flat_tokens_ordered = np.ascontiguousarray(flat_tokens[:, order])
    queries_ordered = [np.ascontiguousarray(q[:, order]) for q in queries]

    doc_starts = np.zeros(num_documents, dtype=np.int64)
    counts = np.bincount(token_to_doc_index, minlength=num_documents)
    doc_starts[1:] = np.cumsum(counts)[:-1]
    tokens_per_doc = counts.astype(np.float64)

    print(
        f"documents={num_documents} doc_token_vectors={flat_tokens.shape[0]} dim={dim} "
        f"k={args.k} rotation={args.rotation} dim_order={args.dim_order} checkpoints={checkpoints}"
    )

    summaries = {}
    for mode in args.threshold_modes:
        start = perf_counter()
        per_query = []
        for query in queries_ordered:
            per_query.append(
                run_bond_for_query(
                    query, flat_tokens_ordered, doc_starts, num_documents,
                    checkpoints, args.k, mode, args.seed_fraction,
                )
            )
        elapsed = perf_counter() - start
        summary = summarize(
            per_query, num_documents, tokens_per_doc, flat_tokens.shape[0], dim, checkpoints, args.k
        )
        summary["simulation_seconds"] = elapsed
        summaries[mode] = summary
        print(
            f"  mode={mode:<7} work_ratio={summary['mean_work_ratio']:.4f} "
            f"ceiling_speedup={summary['ceiling_speedup_vs_naive_full_scan']:.2f}x "
            f"wrong_topk={summary['true_topk_wrongly_pruned_total']}"
        )

    output = {
        "embeddings_dir": str(embeddings_dir),
        "rotation": args.rotation,
        "dim_order": args.dim_order,
        "seed_fraction": args.seed_fraction,
        "checkpoints": checkpoints,
        "dataset": {
            "documents": num_documents,
            "queries": len(queries),
            "total_document_token_vectors": int(flat_tokens.shape[0]),
            "embedding_dimension": dim,
        },
        "summaries_by_threshold_mode": summaries,
        "bound": "cauchy_schwarz_remainder_on_unit_norm_vectors",
        "note": (
            "work_ratio counts (doc-token, dimension) pairs scanned once across all "
            "query tokens (shared vertical scan). Provably exact: no true top-k "
            "document can be pruned. 'oracle' = upper bound on dimension-pruning "
            "potential given a perfect threshold; 'seed' = realistic cheap seed."
        ),
    }
    save_json(output_path, output)

    print("\n=== MaxSim multi-vector BOND pruning (exhaustive, k=%d) ===" % args.k)
    print(f"{'mode':<9}{'work_ratio':>12}{'ceiling_x':>12}{'wrong_topk':>12}")
    for mode in args.threshold_modes:
        s = summaries[mode]
        print(
            f"{mode:<9}{s['mean_work_ratio']:>12.4f}"
            f"{s['ceiling_speedup_vs_naive_full_scan']:>11.2f}x{s['true_topk_wrongly_pruned_total']:>12}"
        )
    print("\nlive-document fraction by dimensions scanned:")
    header = "dim  " + "".join(f"{m:>10}" for m in args.threshold_modes)
    print(header)
    for c in checkpoints:
        row = f"{c:>3}  " + "".join(
            f"{summaries[m]['mean_live_fraction_by_dim'][str(c)]:>10.3f}" for m in args.threshold_modes
        )
        print(row)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
