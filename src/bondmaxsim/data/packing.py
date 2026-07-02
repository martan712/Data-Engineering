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


def pack_corpus_wide(
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    target_group_tokens: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack a token-major corpus into wide dim-major vectorgroups (Stage 1 §5.3).

    Partitions the corpus into groups of consecutive WHOLE documents,
    targeting ~``target_group_tokens`` tokens per group (a single document
    larger than the target gets its own group -- it is never split across
    groups).  Within a group of ``G`` tokens, storage is dim-major ACROSS ALL
    documents in the group: ``group_data[z*G + t]`` where ``t`` is the
    token's LOCAL index within the group (0..G-1, documents concatenated in
    id order) and ``z`` is the physical dimension.  Groups are concatenated
    into one flat float32 buffer.

    Mirrors the per-document ``pack_corpus`` transpose-and-concatenate
    pattern, generalized from "one document's tokens" to "one group's
    tokens" (see cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.cpp
    header comment for the exact ABI this feeds).

    Parameters
    ----------
    flat_tokens : float32 [T, D] — all document tokens, token-major
    doc_starts  : int64 [n_docs] — start token offset of each document
                  (bondmaxsim.data.loader convention: length n_docs, the
                  start of doc d+1 is doc_starts[d+1] or T for the last doc;
                  NOT n_docs+1)
    target_group_tokens : int — target token count per wide vectorgroup

    Returns
    -------
    group_data       : float32 [T * D] — concatenated dim-major group
                        buffers; group g's data starts at
                        ``group_data[group_offsets[g] * D]`` and spans
                        ``D * (group_offsets[g+1] - group_offsets[g])``
                        floats, laid out ``[z*G + t]`` within that span.
    group_offsets    : uint64 [n_groups + 1] — cumulative GLOBAL token
                        offset of each group's first token
                        (group_offsets[0] = 0, group_offsets[-1] = T);
                        mirrors doc_offsets' cumulative-count convention.
    doc_offsets      : uint64 [n_docs + 1] — cumulative GLOBAL token counts
                        per document (doc_starts extended with a trailing T,
                        the per-document oracle's doc_offsets convention).
    group_doc_starts : uint64 [n_groups + 1] — first GLOBAL document id of
                        each group; group g owns documents
                        [group_doc_starts[g], group_doc_starts[g+1]).
    """
    flat_tokens = np.ascontiguousarray(flat_tokens, dtype=np.float32)
    T, D = flat_tokens.shape
    n_docs = len(doc_starts)

    doc_offsets = np.empty(n_docs + 1, dtype=np.uint64)
    if n_docs > 0:
        doc_offsets[:n_docs] = np.asarray(doc_starts, dtype=np.uint64)
    doc_offsets[n_docs] = T

    group_doc_starts: list[int] = [0]
    group_token_starts: list[int] = [0]
    cur_tokens = 0
    for d in range(n_docs):
        nd = int(doc_offsets[d + 1] - doc_offsets[d])
        if cur_tokens > 0 and cur_tokens + nd > target_group_tokens:
            group_doc_starts.append(d)
            group_token_starts.append(int(doc_offsets[d]))
            cur_tokens = 0
        cur_tokens += nd
    group_doc_starts.append(n_docs)
    group_token_starts.append(T)

    n_groups = len(group_doc_starts) - 1
    group_offsets = np.array(group_token_starts, dtype=np.uint64)

    chunks: list[np.ndarray] = []
    for g in range(n_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        chunk = flat_tokens[start:end]                          # [G, D]
        chunks.append(np.ascontiguousarray(chunk.T).ravel())    # dim-major (D, G)
    group_data = np.ascontiguousarray(
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32),
        dtype=np.float32,
    )

    return (
        group_data,
        group_offsets,
        doc_offsets,
        np.array(group_doc_starts, dtype=np.uint64),
    )


PANEL_TOKENS = 16  # panel width of the fused kernel: one AVX-512 register of fp32


def pack_corpus_panels(
    flat_tokens: np.ndarray,
    doc_starts: np.ndarray,
    target_group_tokens: int = 4096,
    panel_tokens: int = PANEL_TOKENS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack a token-major corpus into panel-major vectorgroups (Stage 3b §5.2).

    Layout for the fused panel MaxSim kernel
    (docs/stage3b_fused_panel_maxsim_kernel.md).  Every document's token count
    is first padded up to a multiple of ``panel_tokens`` by DUPLICATING ITS
    LAST TOKEN (§5.3): ``max_j`` over a multiset is invariant under
    duplicating an element, so all MaxSim scores are bit-identical to the
    unpadded corpus.  Zero-padding would be wrong — ``<q_i, 0> = 0`` clamps
    ``max_j`` at >= 0 and corrupts scores whose true max similarity is
    negative.

    The padded corpus is partitioned into groups of consecutive WHOLE
    documents (~``target_group_tokens`` padded tokens per group; an oversized
    document gets its own group — same rule as ``pack_corpus_wide``).  Within
    a group, tokens are stored in consecutive panels of ``panel_tokens``
    tokens; within a panel, storage is dim-major::

        panel_data[(group_offsets[g] + p*panel_tokens)*D + z*panel_tokens + j]

    for panel index ``p`` within group ``g``, dimension ``z`` in [0, D) and
    lane ``j`` in [0, panel_tokens).  Each dimension slice of a panel is one
    contiguous ``panel_tokens``-float run (64 bytes at panel_tokens=16) — the
    BLAS packed-B micro-panel format, packed once at index-build time
    (Stage 3b §3.1).  Documents never straddle panels or groups.

    A zero-length document contributes no panels; the kernel never offers it
    to the top-k (mirrors the exact oracle's empty-doc fallback semantics).

    Parameters
    ----------
    flat_tokens : float32 [T, D] — all document tokens, token-major
    doc_starts  : int64 [n_docs] — start token offset of each document
                  (loader convention, length n_docs; see pack_corpus_wide)
    target_group_tokens : int — target PADDED token count per group
    panel_tokens : int — tokens per panel (16 = one AVX-512 fp32 register)

    Returns
    -------
    panel_data       : float32 [T_pad * D] — concatenated panel-major group
                       buffers (T_pad = padded token count, multiple of
                       panel_tokens)
    group_offsets    : uint64 [n_groups + 1] — cumulative PADDED token offset
                       of each group's first token (multiples of panel_tokens)
    doc_offsets      : uint64 [n_docs + 1] — cumulative PADDED token counts
                       per document (multiples of panel_tokens); this is what
                       the fused kernel scans
    group_doc_starts : uint64 [n_groups + 1] — first GLOBAL document id of
                       each group; group g owns documents
                       [group_doc_starts[g], group_doc_starts[g+1])
    doc_offsets_unpadded : uint64 [n_docs + 1] — original (unpadded)
                       cumulative token counts, kept for id mapping and stats
    """
    flat_tokens = np.ascontiguousarray(flat_tokens, dtype=np.float32)
    T, D = flat_tokens.shape
    n_docs = len(doc_starts)
    PT = int(panel_tokens)

    doc_offsets_unpadded = np.empty(n_docs + 1, dtype=np.uint64)
    if n_docs > 0:
        doc_offsets_unpadded[:n_docs] = np.asarray(doc_starts, dtype=np.uint64)
    doc_offsets_unpadded[n_docs] = T

    lens = np.diff(doc_offsets_unpadded.astype(np.int64))
    padded_lens = ((lens + PT - 1) // PT) * PT          # empty docs stay 0

    doc_offsets = np.zeros(n_docs + 1, dtype=np.uint64)
    doc_offsets[1:] = np.cumsum(padded_lens).astype(np.uint64)

    # Gather index: for each doc, its real tokens followed by copies of its
    # last token up to the padded length.
    gather = np.empty(int(doc_offsets[-1]), dtype=np.int64)
    for d in range(n_docs):
        s, e = int(doc_offsets_unpadded[d]), int(doc_offsets_unpadded[d + 1])
        ps, pe = int(doc_offsets[d]), int(doc_offsets[d + 1])
        if e > s:
            gather[ps:ps + (e - s)] = np.arange(s, e)
            gather[ps + (e - s):pe] = e - 1

    # Group split on PADDED lengths, whole documents (pack_corpus_wide rule).
    group_doc_starts_l: list[int] = [0]
    group_token_starts: list[int] = [0]
    cur_tokens = 0
    for d in range(n_docs):
        nd = int(padded_lens[d])
        if cur_tokens > 0 and cur_tokens + nd > target_group_tokens:
            group_doc_starts_l.append(d)
            group_token_starts.append(int(doc_offsets[d]))
            cur_tokens = 0
        cur_tokens += nd
    group_doc_starts_l.append(n_docs)
    group_token_starts.append(int(doc_offsets[-1]))

    n_groups = len(group_doc_starts_l) - 1
    group_offsets = np.array(group_token_starts, dtype=np.uint64)

    chunks: list[np.ndarray] = []
    for g in range(n_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        chunk = flat_tokens[gather[start:end]]              # [Gp, D], Gp % PT == 0
        Gp = chunk.shape[0]
        # [Gp/PT, PT, D] -> [Gp/PT, D, PT]: dim-major within each panel.
        chunks.append(
            np.ascontiguousarray(
                chunk.reshape(Gp // PT, PT, D).transpose(0, 2, 1)
            ).ravel()
        )
    panel_data = np.ascontiguousarray(
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32),
        dtype=np.float32,
    )

    return (
        panel_data,
        group_offsets,
        doc_offsets,
        np.array(group_doc_starts_l, dtype=np.uint64),
        doc_offsets_unpadded,
    )


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
