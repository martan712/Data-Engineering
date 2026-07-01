"""Token packing and dimension-major layout conversion.

Single responsibility: convert between token-major and dimension-major storage
layouts used by the per-document oracle kernel (M3 in bond_maxsim_methodology.md).

Ported artifact: layout logic from
  research/preliminaries/09_maxsim_pruning/maxsim_kernels.cpp (docs[offset*D + z*n_d + j]
  storage scheme described in Stage 1 §5.2 and M3).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.2 (per-document
  dim-major layout: docs[offset*D + z*n_d + j]), §5.3 (wide-block layout for the
  faithful PDX-BOND uses a different packing).
"""

from __future__ import annotations

import numpy as np

# FETCH schedule from archive/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py.
# Controls how many columns are fetched per pruning round (uint32 array).
DEFAULT_FETCH: np.ndarray = np.array(
    [4, 8, 8, 12, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64,
     128, 128, 128, 128, 256, 256, 512, 1024, 2048, 4096],
    dtype=np.uint32,
)


def pack_dim_major(doc_tokens: np.ndarray) -> np.ndarray:
    """Repack a [n_d, D] token matrix into dimension-major order [D, n_d].

    The resulting array has contiguous columns (one dimension across all tokens).
    This is the per-document layout used by cpp/per_document_oracle/.
    """
    return np.ascontiguousarray(doc_tokens.T)


def unpack_dim_major(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_dim_major: [D, n_d] -> [n_d, D]."""
    return np.ascontiguousarray(packed.T)


def pack_corpus(docs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a list of per-document token matrices into a flat dim-major buffer.

    Mirrors ``build_layout`` from
    archive/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py.

    For each document ``d`` with shape [n_d, D] the tokens are transposed to
    [D, n_d] (dim-major) and ravelled before concatenation so that, within a
    document, all values of dimension ``z`` are contiguous.

    Parameters
    ----------
    docs : list of float32 arrays each with shape [n_d, D]

    Returns
    -------
    flat : float32 [total_tokens * D] — concatenated dim-major buffers
    offs : uint64 [len(docs) + 1]     — cumulative TOKEN counts (not byte offsets)
    """
    offs = np.zeros(len(docs) + 1, dtype=np.uint64)
    chunks: list[np.ndarray] = []
    for k, doc in enumerate(docs):
        chunks.append(np.ascontiguousarray(doc.T).ravel())  # dim-major (D, n_d)
        offs[k + 1] = offs[k] + doc.shape[0]
    flat = np.ascontiguousarray(
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32),
        dtype=np.float32,
    )
    return flat, offs


def build_qcum(query: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Build the cumulative squared-norm prefix table for a query.

    Mirrors ``qcum`` from
    archive/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py.

    ``qcum[i, z]`` = sum of squared query-token values over the first ``z``
    dimensions in ``order`` for query token ``i``.  The leading zero column
    allows a two-pointer Cauchy-Schwarz bound computation without a branch.

    Parameters
    ----------
    query : float32 [m, D] — query token embeddings
    order : int/uint array [D] — dimension scan order (permutation of 0..D-1)

    Returns
    -------
    c : float32 [m, D+1]  — c[:, 0] = 0; c[:, 1:] = cumsum(query[:, order]**2, axis=1)
    """
    m, D = query.shape
    c = np.zeros((m, D + 1), dtype=np.float32)
    c[:, 1:] = np.cumsum(query[:, order] ** 2, axis=1)
    return np.ascontiguousarray(c, dtype=np.float32)
