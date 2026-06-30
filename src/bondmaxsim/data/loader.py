"""Dataset and embedding loading utilities.

Single responsibility: load BEIR corpora, return packed token-embedding arrays
and doc-offset vectors in the layout expected by the rest of the package.

Ported artifact: load_packed_embeddings / unpack_embeddings / flatten_document_embeddings
  from research/colbert/02b_corect_bruteforce.py and
  research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §1.3 (doc_offsets
  partition), §4.1 (unit-norm must be verified here before any downstream use).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_packed_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load a .npz archive produced by the embedding export step.

    Returns a dict with at least 'values' (float32, shape [total_tokens, D])
    and 'offsets' (int64, shape [num_docs]).
    """
    raise NotImplementedError


def unpack_embeddings(
    values: np.ndarray, offsets: np.ndarray
) -> list[np.ndarray]:
    """Split flat token array back into per-document matrices.

    Parameters
    ----------
    values:  float32 array of shape [total_tokens, D]
    offsets: int64 array of shape [num_docs] — start index of each document

    Returns
    -------
    List of float32 arrays, each of shape [n_d, D].
    """
    raise NotImplementedError


def flatten_document_embeddings(
    documents: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a list of per-document token matrices into a single array.

    Returns
    -------
    flat_tokens : float32 [total_tokens, D]  — all tokens concatenated
    token_to_doc : int64 [total_tokens]       — doc index for each token
    doc_starts   : int64 [num_docs]           — start offset of each doc in flat_tokens
    """
    raise NotImplementedError
