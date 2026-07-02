"""Exact MaxSim oracle for top-k document retrieval.

Single responsibility: compute exact ColBERT MaxSim scores for all documents and
return the top-k document ids and scores.  Used as ground-truth for recall
measurement and correctness gating in every experiment.

Ported artifact: exact MaxSim oracle from
  research/colbert/02b_corect_bruteforce.py (exact_maxsim_scores function and
  top-k selection logic).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §1.1 (objective:
  score(q,d) = sum_i max_j <q_i, d_j>), §2.5 (exact finalization proof), §4.3
  (set-equality top-k semantics; ties broken arbitrarily as in np.argpartition).
"""

from __future__ import annotations

import numpy as np


def exact_maxsim_scores(
    query: np.ndarray,
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
) -> np.ndarray:
    """Compute exact per-document MaxSim scores for one query.

    Parameters
    ----------
    query      : float32 [m, D]    — query token embeddings
    flat_tokens: float32 [T, D]    — all document tokens concatenated
    doc_starts : int64  [num_docs] — start offset of each document in flat_tokens

    Returns
    -------
    scores : float32 [num_docs] — score(q, d) = sum_i max_j <q_i, d_j>
    """
    num_docs = len(doc_starts)
    T = flat_tokens.shape[0]
    scores = np.zeros(num_docs, dtype=np.float32)
    if T == 0 or num_docs == 0:
        return scores
    S = query @ flat_tokens.T                          # [m, T] — one BLAS call
    ends = np.empty(num_docs, dtype=np.int64)
    ends[:-1] = doc_starts[1:]
    ends[-1]  = T
    all_nonempty = bool((ends > doc_starts).all())
    if all_nonempty:
        # reduceat: zero Python loop — fast path used by all real datasets.
        idx = np.asarray(doc_starts, dtype=np.intp)
        doc_maxes = np.maximum.reduceat(S, idx, axis=1)   # [m, num_docs]
        scores[:] = doc_maxes.sum(axis=0)
    else:
        # Fallback for edge cases with empty documents (test-only in practice).
        for d in range(num_docs):
            if doc_starts[d] < ends[d]:
                scores[d] = float(S[:, doc_starts[d]:ends[d]].max(axis=1).sum())
    return scores.astype(np.float32)


def exact_maxsim_topk(
    query: np.ndarray,
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-k document ids and their exact MaxSim scores.

    Parameters
    ----------
    query, flat_tokens, doc_starts : see exact_maxsim_scores
    k : int — number of results to return

    Returns
    -------
    ids    : int64  [k] — document indices in descending score order
    scores : float32 [k]
    """
    scores = exact_maxsim_scores(query, flat_tokens, doc_starts)
    num_docs = len(scores)
    actual_k = min(k, num_docs)

    if actual_k == num_docs:
        # return all docs sorted descending
        ids = np.argsort(scores)[::-1].astype(np.int64)
        return ids, scores[ids].astype(np.float32)

    # np.argpartition gives the actual top-k (set semantics, Stage 1 §4.3)
    top_idx = np.argpartition(scores, -actual_k)[-actual_k:]
    top_scores = scores[top_idx]
    sort_order = np.argsort(top_scores)[::-1]
    ids = top_idx[sort_order].astype(np.int64)
    return ids, scores[ids].astype(np.float32)
