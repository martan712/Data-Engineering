"""IVF-partitioned fused BOND MaxSim scan — the combined Stage 4 system arm.

Single responsibility: partition the corpus into document clusters at build
time, pack EACH partition in the fused kernel's panel layout (build-time,
corpus-side — no per-query repacking), and answer queries by scanning only the
top-``nprobe`` partitions by centroid score with the fused BOND kernel, the
pruning threshold seeded by the running global top-k between partitions.

Rationale (R8 discussion, 2026-07-08): token-level IVF candidate pipelines
(faiss_ivf / pdx_ivf / PLAID) pay a per-query candidate-generation cost that
exceeds the entire fused exhaustive scan at small corpus scale, and their
exact rerank cannot use the panel layout without a per-query repack.
Partition-granularity probing keeps every scanned FLOP inside the fused
kernel (the RQ2 3.2x-over-BLAS shape), adds the IVF saving (unprobed
partitions are never touched) AND the mechanism saving (BOND checkpoints
prune within probed partitions), for one tiny m x P GEMM of probe overhead.

This is an APPROXIMATE method: a true top-k document in an unprobed partition
is lost (single-centroid summaries are a crude filter for multi-vector
semantics), so results must be reported as a latency-recall frontier and
never mixed with the exact-safe arms (Convention 4; nprobe = n_partitions
degenerates to the exact exhaustive scan).

Threshold safety WITHIN the probed set: the running k-th best score over
partitions scanned so far is a valid lower bound on the final k-th best over
all probed partitions (adding documents only raises the k-th best), so
seeding each subsequent kernel call with it preserves exactness relative to
the probed subset (Stage 1 §4.4; same argument as candidate_seed_threshold).
"""

from __future__ import annotations

import numpy as np

from bondmaxsim.data.packing import build_qcum, pack_corpus_panels
from bondmaxsim.kernels._ctypes_util import PackedCorpusPanels
from bondmaxsim.kernels.fused_panel import (
    run_fused_panel_bond_validated,
    run_fused_panel_brute_validated,
)

# Safety margin against cross-implementation fp32 noise at near-tie
# thresholds (same rationale as testbed.thresholds._TAU_SEED_EPS).
_TAU_EPS = 1e-3


class _Partition:
    __slots__ = ("doc_ids", "panel", "corpus", "n_docs", "radius")

    def __init__(self, doc_ids: np.ndarray, panel: tuple, n_docs: int,
                 radius: float, dimension: int) -> None:
        self.doc_ids = doc_ids
        self.corpus = PackedCorpusPanels(*panel[:4], dimension)
        self.panel = (
            self.corpus.data,
            self.corpus.group_offsets,
            self.corpus.doc_offsets,
            self.corpus.group_doc_starts,
            panel[4],
        )
        self.n_docs = n_docs
        self.radius = radius


class PartitionedFusedScan:
    """Document-cluster IVF over panel-packed partitions + fused BOND scan.

    Parameters
    ----------
    flat_tokens  : float32 [T, D] — all document tokens (unit-norm)
    doc_starts   : int64 [N]      — start offset of each document
    n_partitions : int | None     — document clusters; None = the power of two
                   nearest sqrt(N), clamped to [16, 512]
    kmeans_niters: int            — k-means iterations (spherical, on
                   L2-normalized document mean-token vectors)
    seed         : int            — k-means seed
    """

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
        n_partitions: int | None = None,
        kmeans_niters: int = 15,
        seed: int = 42,
    ) -> None:
        self.flat_tokens = np.ascontiguousarray(flat_tokens, dtype=np.float32)
        self.doc_starts = np.asarray(doc_starts, dtype=np.int64)
        self.num_docs = len(doc_starts)
        self.D = self.flat_tokens.shape[1]
        if n_partitions is None:
            n_partitions = int(2 ** np.clip(
                np.round(np.log2(np.sqrt(max(self.num_docs, 1)))), 4, 9))
        self.n_partitions = n_partitions
        self.kmeans_niters = kmeans_niters
        self.seed = seed
        self.centroids: np.ndarray | None = None   # [P_eff, D], unit-norm
        self.radii: np.ndarray | None = None       # [P_eff] max token-to-centroid dist
        self.partitions: list[_Partition] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _doc_means(self) -> np.ndarray:
        T = self.flat_tokens.shape[0]
        ends = np.append(self.doc_starts[1:], T)
        sums = np.add.reduceat(self.flat_tokens, self.doc_starts, axis=0)
        lens = (ends - self.doc_starts).astype(np.float32)[:, None]
        means = sums / np.maximum(lens, 1.0)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        return (means / np.maximum(norms, 1e-12)).astype(np.float32)

    def build(self) -> dict:
        """Cluster documents, pack each partition; returns build stats."""
        import faiss

        means = self._doc_means()
        km = faiss.Kmeans(self.D, self.n_partitions, niter=self.kmeans_niters,
                          seed=self.seed, spherical=True, verbose=False,
                          min_points_per_centroid=1)
        km.train(means)
        _, assign = km.index.search(means, 1)
        assign = assign.ravel()

        T = self.flat_tokens.shape[0]
        ends = np.append(self.doc_starts[1:], T)
        centroids = []
        self.partitions = []
        for p in range(self.n_partitions):
            doc_ids = np.flatnonzero(assign == p)
            if len(doc_ids) == 0:
                continue
            # Gather the partition's tokens into a contiguous sub-corpus.
            lens = ends[doc_ids] - self.doc_starts[doc_ids]
            local_starts = np.zeros(len(doc_ids), dtype=np.int64)
            np.cumsum(lens[:-1], out=local_starts[1:])
            idx = (np.arange(int(lens.sum()), dtype=np.int64)
                   - np.repeat(local_starts, lens)
                   + np.repeat(self.doc_starts[doc_ids], lens))
            part_tokens = self.flat_tokens[idx]
            panel = pack_corpus_panels(part_tokens, local_starts)
            # Recompute the centroid from the actual members (unit-norm), and
            # the TOKEN radius to it: R_p = max_t ||t - c_p||.  With unit-norm
            # queries this certifies the exact-safe partition bound
            #   score(q, d in p) <= sum_i <q_i, c_p> + m * R_p
            # (Cauchy-Schwarz per query token) — the only cheap certificate a
            # skipped partition can offer.  Stored for the e03 accounting probe.
            c = means[doc_ids].mean(axis=0)
            c = (c / max(float(np.linalg.norm(c)), 1e-12)).astype(np.float32)
            radius = float(np.sqrt(
                np.max(np.sum((part_tokens - c) ** 2, axis=1))))
            centroids.append(c)
            self.partitions.append(
                _Partition(doc_ids, panel, len(doc_ids), radius, self.D)
            )
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.radii = np.asarray([p.radius for p in self.partitions],
                                dtype=np.float32)
        sizes = np.array([p.n_docs for p in self.partitions])
        return {
            "n_partitions_requested": self.n_partitions,
            "n_partitions_nonempty": len(self.partitions),
            "docs_per_partition_mean": float(sizes.mean()),
            "docs_per_partition_max": int(sizes.max()),
            "token_radius_mean": float(self.radii.mean()),
            "token_radius_min": float(self.radii.min()),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def probe_order(self, query: np.ndarray) -> np.ndarray:
        """Partitions ranked by sum_i <q_i, c_p> (one m x P GEMM, descending)."""
        scores = (query @ self.centroids.T).sum(axis=0)
        return np.argsort(scores)[::-1]

    def search(
        self,
        lib,
        query: np.ndarray,
        k: int = 10,
        nprobe: int = 8,
        checkpoints: np.ndarray | tuple = (112,),
        bound: str = "tight",
        n_threads: int = 0,
        scanner: str = "bond",
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Scan the top-nprobe partitions with the fused kernel.

        scanner="bond"  : fused BOND (checkpoints + rising seeded tau) —
                          measures IVF-skipping AND dimension-pruning together.
        scanner="brute" : fused dense — the pure "IVF saving with our kernel"
                          ablation (checkpoints/bound/tau unused).
        Both are exact WITHIN the probed set; the approximation is identical
        (unprobed partitions), so the two scanners isolate what BOND adds on
        top of partition skipping.

        Returns (doc_ids desc, scores, stats) with stats containing the
        probed-doc fraction and kernel pruning counters (relative to probed
        docs only — accounting for the approximate arm stays separate).
        """
        if scanner not in ("bond", "brute"):
            raise ValueError(f"Unknown scanner: {scanner!r}")
        Q = np.ascontiguousarray(query, dtype=np.float32)
        order = np.arange(self.D, dtype=np.uint32)     # natural (e08 winner)
        Qcum = build_qcum(Q, order) if scanner == "bond" else None
        cps = np.asarray(checkpoints, dtype=np.uint32)

        probe = self.probe_order(Q)[:min(nprobe, len(self.partitions))]
        all_ids: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        tau = float("-inf")
        docs_probed = 0
        docs_pruned = 0
        for p in probe:
            part = self.partitions[p]
            if scanner == "bond":
                ids, scores, stats = run_fused_panel_bond_validated(
                    lib, part.corpus, Q, order, Qcum,
                    shrink=1.0, tau_seed=tau, K=k, n_threads=n_threads,
                    level="doc", checkpoints=cps, bound=bound)
                docs_pruned += int(stats[1])
            else:
                ids, scores = run_fused_panel_brute_validated(
                    lib, part.corpus, Q, k, n_threads=n_threads)
            valid = (ids < part.n_docs) & np.isfinite(scores)
            all_ids.append(part.doc_ids[ids[valid].astype(np.int64)])
            all_scores.append(scores[valid])
            docs_probed += part.n_docs
            # Rising seed: k-th best over everything scanned so far.
            if scanner == "bond":
                merged = np.concatenate(all_scores)
                if len(merged) >= k:
                    tau = float(np.partition(merged, -k)[-k]) - _TAU_EPS

        ids = np.concatenate(all_ids)
        scores = np.concatenate(all_scores)
        actual_k = min(k, len(ids))
        top = np.argpartition(scores, -actual_k)[-actual_k:]
        srt = np.argsort(scores[top])[::-1]
        top = top[srt]
        return ids[top], scores[top], {
            "docs_probed_pct": 100.0 * docs_probed / self.num_docs,
            "docs_pruned_pct_of_probed": (100.0 * docs_pruned / docs_probed
                                          if docs_probed else 0.0),
            "n_probed_partitions": int(len(probe)),
        }
