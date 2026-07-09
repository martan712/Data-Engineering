"""PDX-IVF + exact MaxSim rerank baseline (the Mikel-branch flat PDX-IVF).

Single responsibility: wrap PDX-sigmod's IndexPDXBONDIVFFlat (token-level IVF
whose buckets are scanned with the PDX BOND kernel) for candidate generation,
then apply exact MaxSim reranking over the returned candidate set.

Ported artifact: PDX-IVF prototype from the Mikel branch pipeline
  (experiments/pipeline/03_beir_ivf_benchmark.py on branch `Mikel`; phases
  07/27-28 "IVF discovery").  Faithful to its approx_score candidate policy:
  per query token, retrieve top-L tokens (L2 distance on unit-norm vectors,
  similarity = 1 - d/2), aggregate per document the sum over query tokens of
  the best retrieved similarity, keep the top ``candidate_budget`` documents.
Substrate: extern/PDX-sigmod/ (pinned commit fdc62f2) python package
  `pdxearch` — IndexPDXBONDIVFFlat trains its coarse IVF with faiss k-means
  internally (pdxearch.index_core.IVF wraps faiss.IndexIVFFlat), so this arm
  and the faiss_ivf arm share the quantizer; the difference under test is the
  in-bucket scan kernel (PDX BOND vs faiss flat).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.1, §5.4
  (candidate-generation arm must be kept separate from the dimension-pruning
  arm; an IVF speedup must never be read as a BOND-mechanism speedup).
"""

from __future__ import annotations

import numpy as np

from bondmaxsim.baselines.faiss_ivf import gather_candidate_tokens
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores


class PDXIVFBaseline:
    """PDX-IVF (BOND bucket scan) candidate retrieval + exact MaxSim rerank.

    Mirrors FaissIVFBaseline's interface so the Stage 4 drivers can time both
    identically.  The pdxearch search API is single-vector, so candidate
    generation loops over the m query tokens (as the Mikel pipeline did).

    Parameters
    ----------
    flat_tokens : float32 [T, D] — all document tokens (unit-norm)
    doc_starts  : int64 [N]      — start offset of each document
    n_buckets   : int | None     — IVF partitions; None = same 4*sqrt(T)
                  power-of-two heuristic as FaissIVFBaseline (shared quantizer
                  scale keeps the two arms comparable)
    nprobe      : int            — buckets probed per query token
    train_points_per_bucket : int — k-means training subsample per bucket
    seed        : int            — training subsample seed
    """

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
        n_buckets: int | None = None,
        nprobe: int = 32,
        train_points_per_bucket: int = 64,
        seed: int = 42,
    ) -> None:
        self.flat_tokens = np.ascontiguousarray(flat_tokens, dtype=np.float32)
        self.doc_starts = np.asarray(doc_starts, dtype=np.int64)
        self.num_docs = len(doc_starts)
        T = self.flat_tokens.shape[0]
        if n_buckets is None:
            n_buckets = int(2 ** np.clip(np.round(np.log2(4.0 * np.sqrt(T))), 8, 13))
        self.n_buckets = n_buckets
        self.nprobe = nprobe
        self.train_points_per_bucket = train_points_per_bucket
        self.seed = seed
        self.index = None
        self._token_to_doc = (
            np.searchsorted(self.doc_starts, np.arange(T, dtype=np.int64), side="right") - 1
        )

    def build(self) -> None:
        """Train the coarse IVF (faiss k-means on a subsample, as the Mikel
        pipeline did) and materialize the PDX layout.  In-memory only: the
        pdxearch persist/restore path needs its own file format and the build
        is minutes, not hours, at this corpus scale."""
        from pdxearch.index_factory import IndexPDXBONDIVFFlat

        rng = np.random.default_rng(self.seed)
        T = self.flat_tokens.shape[0]
        n_train = min(self.n_buckets * self.train_points_per_bucket, T)
        sample = np.sort(rng.choice(T, size=n_train, replace=False))
        index = IndexPDXBONDIVFFlat(ndim=self.flat_tokens.shape[1],
                                    nbuckets=self.n_buckets)
        index.train(np.ascontiguousarray(self.flat_tokens[sample]))
        index.add_load(self.flat_tokens)
        self.index = index

    def candidates(
        self,
        query: np.ndarray,
        candidate_budget: int,
        k_token: int | None = None,
    ) -> np.ndarray:
        """Top ``candidate_budget`` doc ids by the approx_score policy.

        k_token defaults to max(256, candidate_budget // 4) — same fill
        heuristic as FaissIVFBaseline.
        """
        if self.index is None:
            raise RuntimeError("PDXIVFBaseline.build() must be called before search.")
        if k_token is None:
            k_token = max(256, candidate_budget // 4)

        approx = np.zeros(self.num_docs, dtype=np.float32)
        seen = np.zeros(self.num_docs, dtype=bool)
        Q = np.ascontiguousarray(query, dtype=np.float32)
        for token in Q:
            hits = self.index.search(token, k_token, nprobe=self.nprobe)
            tok_ids = np.fromiter((h.index for h in hits), dtype=np.int64,
                                  count=len(hits))
            # L2 on unit vectors -> inner-product similarity (Mikel: 1 - d/2).
            sims = 1.0 - np.fromiter((h.distance for h in hits), dtype=np.float32,
                                     count=len(hits)) / 2.0
            docs = self._token_to_doc[tok_ids]
            # Hits arrive distance-ascending, so the first occurrence of a doc
            # is its best similarity for this query token.
            u_docs, u_first = np.unique(docs, return_index=True)
            approx[u_docs] += sims[u_first]
            seen[u_docs] = True

        cand = np.flatnonzero(seen)
        if len(cand) > candidate_budget:
            top = np.argpartition(approx[cand], -candidate_budget)[-candidate_budget:]
            cand = cand[top]
        return np.sort(cand)

    def rerank(
        self, query: np.ndarray, cand: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact MaxSim over the candidate docs; returns (doc_ids, scores) desc."""
        cand_tokens, cand_starts = gather_candidate_tokens(
            self.flat_tokens, self.doc_starts, cand
        )
        cscores = exact_maxsim_scores(query, cand_tokens, cand_starts)
        actual_k = min(k, len(cand))
        top = np.argpartition(cscores, -actual_k)[-actual_k:]
        order = np.argsort(cscores[top])[::-1]
        top = top[order]
        return cand[top].astype(np.int64), cscores[top]

    def topk(
        self,
        query: np.ndarray,
        k: int = 10,
        candidate_budget: int = 1000,
        k_token: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Full pipeline for one query: PDX-IVF candidates + exact MaxSim rerank.

        Returns (doc_ids desc, exact scores, n_candidates_actual).
        """
        cand = self.candidates(query, candidate_budget, k_token=k_token)
        ids, scores = self.rerank(query, cand, k)
        return ids, scores, len(cand)
