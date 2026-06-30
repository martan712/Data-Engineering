"""PDX-IVF + exact MaxSim rerank baseline.

Single responsibility: wrap PDX-sigmod's IVF search (IndexPDXBONDFlat or
IndexPDXIVF variants from extern/PDX-sigmod/) for candidate generation, then
apply exact MaxSim reranking over the returned candidate set.

Ported artifact: PDX-IVF prototype from the Mikel branch pipeline.
Substrate: extern/PDX-sigmod/ (pinned commit fdc62f2) — IndexPDXBONDFlat,
  bench_bond harness.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.1 (PDX-sigmod
  ships BOND as PDXearch variant for single-vector L2; this wrapper uses it for
  candidate generation, not for MaxSim dimension pruning), §5.4 (candidate-
  generation arm must be kept separate from dimension-pruning arm).
"""

from __future__ import annotations

import numpy as np

from bondmaxsim.schema import ResultRecord


class PDXIVFBaseline:
    """PDX-IVF candidate retrieval followed by exact MaxSim rerank.

    Parameters
    ----------
    flat_tokens : float32 [T, D] — all document tokens
    doc_starts  : int64 [N]      — start offset of each document
    """

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def build(self) -> None:
        """Build the PDX-IVF index via the PDX-sigmod C++ library."""
        raise NotImplementedError

    def search(
        self,
        queries: list[np.ndarray],
        k: int = 10,
        candidate_budget: int = 1000,
    ) -> list[ResultRecord]:
        """Retrieve candidates with PDX-IVF, rerank with exact MaxSim."""
        raise NotImplementedError
