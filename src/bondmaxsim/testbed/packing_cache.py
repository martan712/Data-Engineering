"""Lazy corpus-packing and dimension-order dispatch cache for the testbed Runner.

Single responsibility: own the per-doc dim-major (oracle) and wide dim-major
(wide-block) packings of both the original and pca-rotated corpus, building
each lazily on first use, and translate (query, dimension_order) into the
packed kernel inputs each kernel family expects.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bondmaxsim.data.packing import pack_corpus, pack_corpus_panels, pack_corpus_wide
from bondmaxsim.ordering.orders import pca_order, bond_order, natural_order


class PackingCache:
    """Corpus packings + order dispatch shared by the oracle and wide-block modes.

    Parameters
    ----------
    flat_tokens : float32 [total_tokens, D] — all document tokens, row-major
    doc_starts  : int64  [num_docs]          — start token offset of each document
    """

    def __init__(self, flat_tokens: np.ndarray, doc_starts: np.ndarray) -> None:
        # Original token-major corpus (also used directly by the exact oracle).
        self.flat_tokens = np.ascontiguousarray(flat_tokens, dtype=np.float32)
        self.doc_starts  = np.asarray(doc_starts, dtype=np.int64)

        T, D = self.flat_tokens.shape
        self.T = T
        self.D = D
        self.num_docs = len(self.doc_starts)

        # Reconstruct per-doc list for dim-major packing.
        docs: list[np.ndarray] = []
        for d in range(self.num_docs):
            start = int(self.doc_starts[d])
            end   = int(self.doc_starts[d + 1]) if d + 1 < self.num_docs else T
            docs.append(self.flat_tokens[start:end])
        self._docs = docs  # keep for pca rotated packing

        # Per-doc dim-major packed corpus for the oracle kernel.
        self._flat, self._offs = pack_corpus(docs)

        # Corpus token mean for BOND order.
        self._mu = self.flat_tokens.mean(axis=0).astype(np.float32)

        # PCA rotation and rotated corpus — built lazily.
        self._R: Optional[np.ndarray] = None
        self._flat_rot: Optional[np.ndarray] = None
        self._offs_rot: Optional[np.ndarray] = None
        self._flat_tokens_rot: Optional[np.ndarray] = None  # token-major rotated corpus

        # Wide-block kernel packings — built lazily.
        self._wide: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self._wide_rot: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

        # Fused-panel kernel packing (Stage 3b) — built lazily.
        self._panel: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    def get_pca_rotation(self) -> np.ndarray:
        """Build a deterministic orthogonal rotation (QR of a seeded random matrix)."""
        if self._R is None:
            g = np.random.default_rng(123).standard_normal((self.D, self.D))
            Q_r, _ = np.linalg.qr(g)
            self._R = Q_r.astype(np.float32)
        return self._R

    def get_flat_tokens_rot(self) -> np.ndarray:
        """Return the token-major rotated corpus [T, D] (built lazily).

        Orthogonal rotation preserves inner products and unit norm (Stage 1
        §4.5), so this is exact-safe at shrink=1; used both to build the
        wide-block pca packing and as the seed-policy checkpoint corpus.
        """
        if self._flat_tokens_rot is None:
            R = self.get_pca_rotation()
            self._flat_tokens_rot = (self.flat_tokens @ R).astype(np.float32)
        return self._flat_tokens_rot

    def _get_flat_rot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (flat_rot, offs) for the rotated per-doc packing (built lazily)."""
        if self._flat_rot is None:
            R = self.get_pca_rotation()
            rotated_docs = [(d @ R).astype(np.float32) for d in self._docs]
            self._flat_rot, self._offs_rot = pack_corpus(rotated_docs)
        return self._flat_rot, self._offs_rot

    def _get_wide_packing(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Wide packing of the ORIGINAL (un-rotated) corpus (natural/bond orders)."""
        if self._wide is None:
            self._wide = pack_corpus_wide(self.flat_tokens, self.doc_starts)
        return self._wide

    def _get_wide_packing_rot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Wide packing of the pca-ROTATED corpus."""
        if self._wide_rot is None:
            flat_rot = self.get_flat_tokens_rot()
            self._wide_rot = pack_corpus_wide(flat_rot, self.doc_starts)
        return self._wide_rot

    def _get_panel_packing(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Panel-major packing of the ORIGINAL corpus for the Stage 3b fused
        kernel (pack_corpus_panels: 16-token panels, duplicate-last-token doc
        padding).  Natural dimension order only — the fused brute kernel
        scans all dimensions, so order is irrelevant."""
        if self._panel is None:
            self._panel = pack_corpus_panels(self.flat_tokens, self.doc_starts)
        return self._panel

    # ------------------------------------------------------------------
    # Order dispatch
    # ------------------------------------------------------------------

    def dispatch_order(
        self,
        query: np.ndarray,
        dimension_order: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (flat_eff, offs_eff, Q_eff, order_u32) for the given order
        (per-document-oracle kernel packing)."""
        if dimension_order == "natural":
            order = natural_order(query).astype(np.uint32)
            return self._flat, self._offs, query, order

        elif dimension_order == "bond":
            order = bond_order(query, self._mu).astype(np.uint32)
            return self._flat, self._offs, query, order

        elif dimension_order == "pca":
            R = self.get_pca_rotation()
            flat_rot, offs_rot = self._get_flat_rot()
            q_rot, order = pca_order(query, R)
            order = order.astype(np.uint32)
            return flat_rot, offs_rot, q_rot, order

        else:
            raise ValueError(
                f"Unknown dimension_order: {dimension_order!r}. "
                "Expected one of 'natural', 'bond', 'pca'."
            )

    def dispatch_order_wide(
        self,
        query: np.ndarray,
        dimension_order: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (group_data, group_offsets, doc_offsets, group_doc_starts,
        Q_eff, order_u32) for the given order (wide-block kernel packing),
        mirroring dispatch_order()."""
        if dimension_order == "natural":
            order = natural_order(query).astype(np.uint32)
            group_data, group_offsets, doc_offsets, group_doc_starts = self._get_wide_packing()
            return group_data, group_offsets, doc_offsets, group_doc_starts, query, order

        elif dimension_order == "bond":
            order = bond_order(query, self._mu).astype(np.uint32)
            group_data, group_offsets, doc_offsets, group_doc_starts = self._get_wide_packing()
            return group_data, group_offsets, doc_offsets, group_doc_starts, query, order

        elif dimension_order == "pca":
            R = self.get_pca_rotation()
            group_data, group_offsets, doc_offsets, group_doc_starts = self._get_wide_packing_rot()
            q_rot, order = pca_order(query, R)
            order = order.astype(np.uint32)
            return group_data, group_offsets, doc_offsets, group_doc_starts, q_rot, order

        else:
            raise ValueError(
                f"Unknown dimension_order: {dimension_order!r}. "
                "Expected one of 'natural', 'bond', 'pca'."
            )
