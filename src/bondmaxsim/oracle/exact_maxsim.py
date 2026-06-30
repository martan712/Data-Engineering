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
    raise NotImplementedError


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
    raise NotImplementedError
