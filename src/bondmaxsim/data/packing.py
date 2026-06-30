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


def pack_dim_major(doc_tokens: np.ndarray) -> np.ndarray:
    """Repack a [n_d, D] token matrix into dimension-major order [D, n_d].

    The resulting array has contiguous columns (one dimension across all tokens).
    This is the per-document layout used by cpp/per_document_oracle/.
    """
    raise NotImplementedError


def unpack_dim_major(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_dim_major: [D, n_d] -> [n_d, D]."""
    raise NotImplementedError
