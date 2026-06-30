"""FAISS-IVF + exact MaxSim rerank baseline.

Single responsibility: build a FAISS-IVF index over document token embeddings,
retrieve nprobe-based candidates per query token, aggregate by document, and
rerank with exact MaxSim over the candidate set.

Ported artifact: FAISS-IVF + exact rerank pipeline from
  research/colbert/03_beir_comparison.py and the Mikel branch pipeline utilities.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.4 (candidate-
  generation alternative; IVF speedups must NEVER be read as BOND speedups;
  this baseline uses single-vector distance bounds over tokens, not the Section 2
  document bound), §8 item 9 (mechanism vs candidate-generation must be kept
  separate in reporting).
"""

from __future__ import annotations

import numpy as np

from bondmaxsim.schema import ResultRecord


class FaissIVFBaseline:
    """FAISS-IVF nearest-token retrieval followed by exact MaxSim rerank.

    Parameters
    ----------
    flat_tokens : float32 [T, D] — all document tokens
    doc_starts  : int64 [N]      — start offset of each document in flat_tokens
    n_lists     : int            — number of IVF centroids
    """

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
        n_lists: int = 100,
    ) -> None:
        raise NotImplementedError

    def build(self) -> None:
        """Train and populate the FAISS-IVF index."""
        raise NotImplementedError

    def search(
        self,
        queries: list[np.ndarray],
        k: int = 10,
        candidate_budget: int = 1000,
        nprobe: int = 10,
    ) -> list[ResultRecord]:
        """Retrieve candidates with IVF, rerank with exact MaxSim, return ResultRecords."""
        raise NotImplementedError
