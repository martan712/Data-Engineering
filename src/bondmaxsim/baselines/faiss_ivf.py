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

from pathlib import Path

import numpy as np

from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores


def gather_candidate_tokens(
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    doc_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather the token rows of the given documents into a compact sub-corpus.

    Vectorized (no per-document Python loop) because this gather is on the
    timed rerank path of every candidate-generation arm.

    Returns
    -------
    cand_tokens : float32 [T_cand, D] — tokens of the candidate docs, in
                  doc_ids order
    cand_starts : int64 [len(doc_ids)] — start offset of each candidate doc
                  within cand_tokens
    """
    T = flat_tokens.shape[0]
    num_docs = len(doc_starts)
    ends = np.empty(num_docs, dtype=np.int64)
    ends[:-1] = doc_starts[1:]
    ends[-1] = T

    starts = doc_starts[doc_ids]
    lens = ends[doc_ids] - starts
    total = int(lens.sum())
    cand_starts = np.zeros(len(doc_ids), dtype=np.int64)
    np.cumsum(lens[:-1], out=cand_starts[1:])
    # Row index into flat_tokens for every output row.
    idx = np.arange(total, dtype=np.int64) - np.repeat(cand_starts, lens) + np.repeat(starts, lens)
    return flat_tokens[idx], cand_starts


class FaissIVFBaseline:
    """FAISS-IVF nearest-token retrieval followed by exact MaxSim rerank.

    Candidate generation: every query token searches the token-level IVF index
    (inner product, unit-norm tokens) with the configured nprobe; hits are
    aggregated per document with the standard first-stage approximation
    sum_i max_over_retrieved_j <q_i, d_j> (missing pairs contribute 0), and
    the top ``candidate_budget`` documents by that approximate score become
    the candidate set.  The budget is held FIXED (Stage 1 §5.4 / Convention
    7): the rerank always sees at most ``candidate_budget`` documents, so an
    IVF speedup cannot be mistaken for a scoring-kernel speedup.

    Parameters
    ----------
    flat_tokens : float32 [T, D] — all document tokens
    doc_starts  : int64 [N]      — start offset of each document in flat_tokens
    n_lists     : int | None     — number of IVF centroids; None = the usual
                  4*sqrt(T) heuristic rounded to a power of two in [256, 8192]
    nprobe      : int            — IVF lists probed per query token
    kmeans_niters : int          — k-means iterations (candidate generation does
                  not need converged centroids; 10 ≈ faiss-default quality here)
    train_points_per_centroid : int — training subsample size per centroid
    seed        : int            — k-means training seed
    """

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
        n_lists: int | None = None,
        nprobe: int = 32,
        kmeans_niters: int = 10,
        train_points_per_centroid: int = 64,
        seed: int = 42,
    ) -> None:
        self.flat_tokens = np.ascontiguousarray(flat_tokens, dtype=np.float32)
        self.doc_starts = np.asarray(doc_starts, dtype=np.int64)
        self.num_docs = len(doc_starts)
        T = self.flat_tokens.shape[0]
        if n_lists is None:
            n_lists = int(2 ** np.clip(np.round(np.log2(4.0 * np.sqrt(T))), 8, 13))
        self.n_lists = n_lists
        self.nprobe = nprobe
        self.kmeans_niters = kmeans_niters
        self.train_points_per_centroid = train_points_per_centroid
        self.seed = seed
        self.index = None
        # Doc id of every token row (searchsorted over the offsets partition).
        self._token_to_doc = (
            np.searchsorted(self.doc_starts, np.arange(T, dtype=np.int64), side="right") - 1
        )

    def build(self, cache_path: "str | None" = None) -> None:
        """Train and populate the FAISS-IVF index (inner-product metric).

        cache_path: optional .faiss file; if it exists the index is loaded
        from disk instead of retrained (k-means over ~1M tokens costs minutes),
        and it is written after a fresh build otherwise.
        """
        import faiss

        if cache_path is not None:
            p = Path(cache_path)
            if p.exists():
                self.index = faiss.read_index(str(p))
                self.index.nprobe = self.nprobe
                return

        D = self.flat_tokens.shape[1]
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, self.n_lists, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = self.seed
        index.cp.niter = self.kmeans_niters
        index.cp.max_points_per_centroid = self.train_points_per_centroid
        index.train(self.flat_tokens)
        index.add(self.flat_tokens)
        index.nprobe = self.nprobe
        self.index = index

        if cache_path is not None:
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(p))

    def candidates(
        self,
        query: np.ndarray,
        candidate_budget: int,
        k_token: int | None = None,
    ) -> np.ndarray:
        """Return up to ``candidate_budget`` candidate doc ids for one query.

        k_token defaults to max(256, candidate_budget // 4) retrieved tokens
        per query token — in practice each retrieved token maps to a distinct
        promising document often enough that the union over the m query tokens
        fills the budget (the actual candidate count is returned by topk and
        recorded by the drivers).
        """
        if self.index is None:
            raise RuntimeError("FaissIVFBaseline.build() must be called before search.")
        if k_token is None:
            k_token = max(256, candidate_budget // 4)
        Q = np.ascontiguousarray(query, dtype=np.float32)
        scores, token_ids = self.index.search(Q, k_token)

        approx = np.zeros(self.num_docs, dtype=np.float32)
        seen = np.zeros(self.num_docs, dtype=bool)
        for i in range(Q.shape[0]):
            hit = token_ids[i] >= 0
            docs_i = self._token_to_doc[token_ids[i][hit]]
            # FAISS rows are score-descending, so the FIRST occurrence of a doc
            # in the row is that doc's max hit for this query token.
            u_docs, u_first = np.unique(docs_i, return_index=True)
            approx[u_docs] += scores[i][hit][u_first]
            seen[u_docs] = True

        cand = np.flatnonzero(seen)
        if len(cand) > candidate_budget:
            top = np.argpartition(approx[cand], -candidate_budget)[-candidate_budget:]
            cand = cand[top]
        # Ascending id order: locality for the rerank gather, deterministic output.
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
        """Full pipeline for one query: IVF candidates + exact MaxSim rerank.

        Returns (doc_ids desc, exact scores, n_candidates_actual).
        """
        cand = self.candidates(query, candidate_budget, k_token=k_token)
        ids, scores = self.rerank(query, cand, k)
        return ids, scores, len(cand)
