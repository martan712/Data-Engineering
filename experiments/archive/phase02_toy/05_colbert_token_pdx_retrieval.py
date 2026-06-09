"""Toy ColBERT token-vector retrieval with PDX-BOND.

PDX's current Python API exposes squared L2 search (`l2sq`), not inner product.
For this first experiment, document and query token vectors are L2-normalized,
then PDX-BOND retrieves nearest token vectors. For unit vectors:

    squared_l2(q, x) = 2 - 2 * dot(q, x)

So nearest-by-L2 is equivalent to largest cosine/inner-product similarity.
Retrieved token hits are mapped back to document ids and aggregated with a
ColBERT-like per-query-token max followed by a sum over query tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from time import perf_counter

import numpy as np

from utils_colbert import (
    l2_normalize,
    load_packed_embeddings,
    maxsim_score,
    rank_documents,
    recall_at_k,
    unpack_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "pylate_embeddings"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
OUTPUT_PATH = RESULTS_DIR / "colbert_token_pdx_retrieval.json"
L_VALUES = [5, 10, 20, 50]
REPORT_K_VALUES = [1, 3, 5]


def import_pdx_api():
    """Import the installed PDX API, falling back to the source checkout."""
    try:
        from pdxearch.constants import PDXConstants
        from pdxearch.index_factory import IndexPDXBONDFlat
        return IndexPDXBONDFlat, PDXConstants.SUPPORTED_METRICS
    except ModuleNotFoundError as first_exc:
        source_path = PROJECT_ROOT / "external" / "PDX" / "python"
        if source_path.exists():
            sys.path.insert(0, str(source_path))
            try:
                from pdxearch.constants import PDXConstants
                from pdxearch.index_factory import IndexPDXBONDFlat
                return IndexPDXBONDFlat, PDXConstants.SUPPORTED_METRICS
            except ModuleNotFoundError as second_exc:
                raise SystemExit(
                    "Could not import PDX. Build/install PDX in WSL/Linux first. "
                    f"Original error: {second_exc}"
                ) from second_exc
        raise SystemExit(
            "Could not import PDX. Build/install PDX in WSL/Linux first. "
            f"Original error: {first_exc}"
        ) from first_exc


def exact_normalized_reference(
    queries: list[np.ndarray],
    documents: list[np.ndarray],
) -> tuple[list[dict], float]:
    """Compute exact normalized-IP MaxSim reference rankings."""
    start = perf_counter()
    results = []
    for query_index, query_matrix in enumerate(queries):
        scores = np.array(
            [
                maxsim_score(query_matrix, document_matrix, normalize=True)
                for document_matrix in documents
            ],
            dtype=np.float32,
        )
        results.append(
            {
                "query_index": query_index,
                "scores": scores.tolist(),
                "ranking": rank_documents(scores),
            }
        )
    return results, perf_counter() - start


def flatten_documents(
    documents: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten document token matrices and keep token-to-document metadata."""
    flat_vectors = []
    token_to_doc_index = []
    token_to_token_position = []

    for doc_index, document_matrix in enumerate(documents):
        flat_vectors.append(document_matrix)
        token_count = len(document_matrix)
        token_to_doc_index.extend([doc_index] * token_count)
        token_to_token_position.extend(range(token_count))

    return (
        np.ascontiguousarray(np.vstack(flat_vectors), dtype=np.float32),
        np.array(token_to_doc_index, dtype=np.int64),
        np.array(token_to_token_position, dtype=np.int64),
    )


def aggregate_token_hits(
    *,
    index,
    query_matrix: np.ndarray,
    token_to_doc_index: np.ndarray,
    num_documents: int,
    top_l: int,
) -> tuple[np.ndarray, int, float]:
    """Retrieve token hits from PDX and aggregate them into document scores.

    Missing document contributions for a query token are set to 0.0. This keeps
    the approximation simple, but it is a limitation: exact MaxSim has one
    contribution from every document for every query token.
    """
    per_token_doc_max = np.full(
        (len(query_matrix), num_documents),
        -np.inf,
        dtype=np.float32,
    )
    retrieved_tokens = 0
    start = perf_counter()

    for query_token_index, query_token in enumerate(query_matrix):
        hits = index.search(np.ascontiguousarray(query_token, dtype=np.float32), top_l)
        retrieved_tokens += len(hits)

        for hit in hits:
            token_index = int(hit.index)
            doc_index = int(token_to_doc_index[token_index])

            # PDX returns squared L2. On normalized vectors, convert to cosine.
            similarity = 1.0 - (float(hit.distance) / 2.0)

            if similarity > per_token_doc_max[query_token_index, doc_index]:
                per_token_doc_max[query_token_index, doc_index] = similarity

    elapsed = perf_counter() - start
    missing_mask = ~np.isfinite(per_token_doc_max)
    filled = np.where(missing_mask, 0.0, per_token_doc_max)
    scores = filled.sum(axis=0)
    return scores.astype(np.float32), retrieved_tokens, elapsed


def compare_rankings(
    reference_ranking: list[int],
    candidate_ranking: list[int],
) -> dict:
    """Summarize ranking agreement against the exact reference ranking."""
    k_values = [k for k in REPORT_K_VALUES if k <= len(reference_ranking)]
    prefix_match_count = 0
    for ref_doc, cand_doc in zip(reference_ranking, candidate_ranking):
        if ref_doc != cand_doc:
            break
        prefix_match_count += 1

    return {
        "exact_full_ranking_match": reference_ranking == candidate_ranking,
        "top1_match": reference_ranking[:1] == candidate_ranking[:1],
        "prefix_match_count": prefix_match_count,
        "recall_at_k": {
            str(k): recall_at_k(reference_ranking, candidate_ranking, k)
            for k in k_values
        },
    }


def main() -> None:
    IndexPDXBONDFlat, supported_metrics = import_pdx_api()
    if "l2sq" not in supported_metrics:
        raise SystemExit(f"Expected l2sq support in PDX API, got {supported_metrics}")

    document_pack = load_packed_embeddings(EMBEDDINGS_DIR / "toy_document_embeddings.npz")
    query_pack = load_packed_embeddings(EMBEDDINGS_DIR / "toy_query_embeddings.npz")
    documents = unpack_embeddings(document_pack["values"], document_pack["offsets"])
    queries = unpack_embeddings(query_pack["values"], query_pack["offsets"])

    normalized_documents = [l2_normalize(document) for document in documents]
    normalized_queries = [l2_normalize(query) for query in queries]
    flat_tokens, token_to_doc_index, token_to_token_position = flatten_documents(
        normalized_documents
    )

    exact_results, exact_time = exact_normalized_reference(
        normalized_queries,
        normalized_documents,
    )

    index = IndexPDXBONDFlat(ndim=flat_tokens.shape[1])
    start = perf_counter()
    index.add_load(flat_tokens)
    index_time = perf_counter() - start

    runs = []
    for top_l in L_VALUES:
        query_runs = []
        total_retrieved = 0
        total_query_time = 0.0

        for exact_result, query_matrix in zip(exact_results, normalized_queries):
            scores, retrieved_tokens, query_time = aggregate_token_hits(
                index=index,
                query_matrix=query_matrix,
                token_to_doc_index=token_to_doc_index,
                num_documents=len(normalized_documents),
                top_l=min(top_l, len(flat_tokens)),
            )
            candidate_ranking = rank_documents(scores)
            comparison = compare_rankings(
                exact_result["ranking"],
                candidate_ranking,
            )
            total_retrieved += retrieved_tokens
            total_query_time += query_time
            query_runs.append(
                {
                    "query_index": exact_result["query_index"],
                    "scores": scores.tolist(),
                    "ranking": candidate_ranking,
                    "retrieved_token_vectors": retrieved_tokens,
                    "elapsed_seconds": query_time,
                    "comparison": comparison,
                }
            )

        mean_recall = {
            str(k): float(np.mean([
                query_run["comparison"]["recall_at_k"][str(k)]
                for query_run in query_runs
                if str(k) in query_run["comparison"]["recall_at_k"]
            ]))
            for k in REPORT_K_VALUES
            if k <= len(normalized_documents)
        }
        runs.append(
            {
                "top_l": top_l,
                "total_retrieved_token_vectors": total_retrieved,
                "elapsed_seconds": total_query_time,
                "mean_recall_at_k": mean_recall,
                "full_ranking_matches": sum(
                    1
                    for query_run in query_runs
                    if query_run["comparison"]["exact_full_ranking_match"]
                ),
                "queries": query_runs,
            }
        )

    output = {
        "metric": {
            "pdx_supported_metrics": supported_metrics,
            "used": "l2sq_on_l2_normalized_vectors",
            "similarity_interpretation": "cosine_similarity = 1 - squared_l2 / 2",
            "reason": "Current PDX Python API exposes l2sq, not inner product.",
        },
        "documents": {
            "ids": document_pack["ids"],
            "values_shape": list(document_pack["values"].shape),
            "offsets": document_pack["offsets"].tolist(),
        },
        "queries": {
            "ids": query_pack["ids"],
            "values_shape": list(query_pack["values"].shape),
            "offsets": query_pack["offsets"].tolist(),
        },
        "token_index": {
            "flat_values_shape": list(flat_tokens.shape),
            "token_to_doc_id": [
                document_pack["ids"][int(doc_index)]
                for doc_index in token_to_doc_index
            ],
            "token_to_doc_index": token_to_doc_index.tolist(),
            "token_to_token_position": token_to_token_position.tolist(),
            "index_build_seconds": index_time,
        },
        "exact_reference": {
            "metric": "normalized_inner_product_maxsim",
            "elapsed_seconds": exact_time,
            "queries": exact_results,
        },
        "runs": runs,
        "limitation": (
            "This is token-level candidate retrieval. Missing document hits for "
            "a query token contribute 0.0, so it is an approximation of exact "
            "MaxSim unless top-L retrieves the best token for each relevant "
            "document/query-token pair."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("ColBERT token-vector PDX retrieval")
    print(f"PDX supported metrics: {supported_metrics}")
    print("Metric used: l2sq on L2-normalized token vectors")
    print(f"Flat token matrix: {flat_tokens.shape}")
    print(f"Index build time: {index_time:.6f}s")
    print(f"Exact normalized MaxSim time: {exact_time:.6f}s")
    for run in runs:
        print(f"\nL={run['top_l']}")
        print(f"  retrieved token vectors: {run['total_retrieved_token_vectors']}")
        print(f"  PDX query time: {run['elapsed_seconds']:.6f}s")
        print(f"  mean recall@k: {run['mean_recall_at_k']}")
        print(
            "  full ranking matches: "
            f"{run['full_ranking_matches']}/{len(run['queries'])}"
        )
        first_query = run["queries"][0]
        print(f"  first query ranking: {first_query['ranking']}")
        print(
            "  first query exact ranking: "
            f"{exact_results[0]['ranking']}"
        )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
