"""PLAID (ColBERT candidate generation) baseline wrapper.

Single responsibility: wrap PyLate / PLAID retrieval as a baseline so it can be
compared on the same result schema as bondmaxsim methods.

Ported artifact: PLAID wrapper from
  research/colbert/02_corect_quick.py and research/colbert/03_beir_comparison.py.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §5.4 (PLAID is the
  canonical candidate-generation baseline for ColBERT; it uses single-vector
  centroid search, not MaxSim dimension pruning, and must be kept as a separate
  arm), docs/project_b_analysis_and_research_plan.md Stage 4 (PLAID with tuned
  settings is a required baseline for the integration comparison).

Backend: PyLate >= 1.6 FastPlaid (Rust).  The two search-time knobs are
  n_ivf_probe   — centroid lists probed per query token (recall knob), and
  n_full_scores — number of candidate documents fully re-ranked; this is the
                  candidate-budget analog, so the Stage 4 fixed-candidate-set
                  control maps to sweeping n_full_scores (Convention 7: a
                  PLAID speedup from a smaller candidate set must never be
                  read as a scoring-kernel speedup).
Index scale note (R8 "PLAID retuned at appropriate scale"): FastPlaid picks
  the number of centroids from corpus size (16 * sqrt(T) heuristic), so the
  index is already scaled to our small BEIR corpora rather than tuned for
  web-scale defaults; kmeans_niters/nbits are exposed for explicit retuning.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.compat.plaid import PlaidCapabilities, inspect_plaid_capabilities
from bondmaxsim.experiments.candidate_work import (
    CandidateWorkObservation,
    CountObservation,
)

_DEFAULT_INDEX_ROOT: Path = REPO_ROOT / "data" / "plaid_indexes"


class PLAIDBaseline:
    """PLAID retrieval baseline via PyLate's FastPlaid backend.

    Documents are indexed once per dataset from precomputed ColBERT token
    embeddings (no model in the loop); search-time parameters can be changed
    per query batch without rebuilding.

    Parameters
    ----------
    dataset       : str — BEIR corpus name; names the index directory
    index_root    : Path — parent directory for PLAID indexes
                    (default data/plaid_indexes/)
    nbits         : int — product-quantization bits (index-time)
    kmeans_niters : int — k-means iterations (index-time)
    n_ivf_probe   : int — centroid lists probed per query token (search-time)
    n_full_scores : int — documents fully re-ranked = candidate budget
                    (search-time)
    num_threads   : int | None — fast-plaid CPU worker processes (None = default)
    seed          : int — k-means seed
    """

    def __init__(
        self,
        dataset: str,
        index_root: Path | str | None = None,
        nbits: int = 4,
        kmeans_niters: int = 4,
        n_ivf_probe: int = 8,
        n_full_scores: int = 8192,
        num_threads: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.index_root = Path(index_root) if index_root is not None else _DEFAULT_INDEX_ROOT
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.n_ivf_probe = n_ivf_probe
        self.n_full_scores = n_full_scores
        self.num_threads = num_threads
        self.seed = seed
        self._plaid = None
        self._plaid_to_doc: dict | None = None
        self._capabilities: PlaidCapabilities | None = None

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _make_index(self, override: bool):
        from pylate import indexes

        self._capabilities = inspect_plaid_capabilities(indexes.PLAID)
        return indexes.PLAID(
            index_folder=str(self.index_root),
            index_name=self.dataset,
            override=override,
            use_fast=True,
            nbits=self.nbits,
            kmeans_niters=self.kmeans_niters,
            n_ivf_probe=self.n_ivf_probe,
            n_full_scores=self.n_full_scores,
            num_threads=self.num_threads,
            show_progress=False,
            device="cpu",
            seed=self.seed,
        )

    def exists(self) -> bool:
        return (self.index_root / self.dataset / "fast_plaid_index").exists()

    def build(self, doc_embeddings: list[np.ndarray]) -> None:
        """Build (or rebuild) the PLAID index from per-document token matrices."""
        self._plaid = self._make_index(override=True)
        ids = [str(i) for i in range(len(doc_embeddings))]
        self._plaid.add_documents(documents_ids=ids, documents_embeddings=doc_embeddings)
        self._load_id_map()

    def load(self) -> None:
        """Attach to an existing on-disk index (build() must have run before)."""
        if not self.exists():
            raise FileNotFoundError(
                f"No PLAID index for {self.dataset!r} under {self.index_root}; "
                "call build() first."
            )
        self._plaid = self._make_index(override=False)
        self._load_id_map()

    def _load_id_map(self) -> None:
        p = self.index_root / self.dataset / "plaid_ids_to_documents_ids.pkl"
        with p.open("rb") as fh:
            self._plaid_to_doc = pickle.load(fh)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def set_search_params(
        self,
        n_ivf_probe: int | None = None,
        n_full_scores: int | None = None,
    ) -> None:
        """Change search knobs through PyLate's public constructor interface.

        PyLate 1.6.0 has no public mutator. Reattaching the already-built index
        applies the active constructor settings and happens outside timed search.
        """
        if self._plaid is None:
            raise RuntimeError("Index not loaded; call build() or load() first.")
        if n_ivf_probe is not None:
            if n_ivf_probe <= 0:
                raise ValueError("n_ivf_probe must be positive")
            self.n_ivf_probe = n_ivf_probe
        if n_full_scores is not None:
            if n_full_scores <= 0:
                raise ValueError("n_full_scores must be positive")
            self.n_full_scores = n_full_scores
        self._plaid = self._make_index(override=False)

    def search(
        self,
        queries: list[np.ndarray],
        k: int = 10,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch PLAID retrieval; returns per-query (doc_ids int64 desc, scores).

        The whole batch goes through fast-plaid in one call (its natural
        execution mode); per-query latency is batch time / n_queries.
        """
        results, _ = self.search_with_work(queries, k=k)
        return results

    def search_with_work(
        self,
        queries: list[np.ndarray],
        k: int = 10,
        comparison_scope: str = "system_cap",
    ) -> tuple[
        list[tuple[np.ndarray, np.ndarray]],
        list[CandidateWorkObservation],
    ]:
        """Batch search with honest per-query work observability metadata."""
        if self._plaid is None or self._capabilities is None:
            raise RuntimeError("Index not loaded; call build() or load() first.")
        self._capabilities.require_system_cap(comparison_scope)
        results = self._plaid(
            [np.ascontiguousarray(q, dtype=np.float32) for q in queries], k=k
        )
        out = []
        for query_results in results:
            ids = np.array([int(r["id"]) for r in query_results], dtype=np.int64)
            scores = np.array([r["score"] for r in query_results], dtype=np.float32)
            out.append((ids, scores))
        unavailable = CountObservation.unavailable(self._capabilities.count_source)
        no_partition = CountObservation.unavailable(
            "PLAID does not expose document-partition probe counts"
        )
        work = [
            CandidateWorkObservation(
                configured_candidate_cap=CountObservation.unavailable(
                    "PLAID is configured by a full-score cap, not a candidate cap"
                ),
                configured_full_score_cap=CountObservation.exact(
                    self.n_full_scores, "PLAID n_full_scores public constructor setting"
                ),
                unique_candidates_generated=unavailable,
                documents_admitted_to_scoring=unavailable,
                documents_fully_scored=unavailable,
                documents_probed=no_partition,
                partitions_probed=no_partition,
                token_hits_inspected=CountObservation.unavailable(
                    "PyLate PLAID does not expose per-query token-hit counts"
                ),
            )
            for _ in queries
        ]
        return out, work

    def compatibility_record(self) -> dict[str, str | bool]:
        if self._capabilities is None:
            raise RuntimeError("Index not loaded; call build() or load() first.")
        return self._capabilities.to_dict()
