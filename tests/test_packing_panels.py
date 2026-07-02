"""Stage 3b K1 tests: panel-major packing (pack_corpus_panels).

Verifies the layout contract of docs/stage3b_fused_panel_maxsim_kernel.md
§5.2–5.3: every document 16-aligned, panels dim-major, duplicate-last-token
padding (max-invariant, never zeros), and MaxSim score equality between the
padded corpus and the unpadded exact oracle.
"""

from __future__ import annotations

import numpy as np

from bondmaxsim.data.packing import PANEL_TOKENS, pack_corpus_panels
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores


def _make_corpus(seed: int, n_docs: int, D: int, min_len: int = 1, max_len: int = 40):
    rng = np.random.default_rng(seed)
    docs, starts, cur = [], [], 0
    for _ in range(n_docs):
        n = int(rng.integers(min_len, max_len + 1))
        starts.append(cur)
        v = rng.standard_normal((n, D)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        docs.append(v)
        cur += n
    flat = np.concatenate(docs, axis=0).astype(np.float32)
    return flat, np.array(starts, dtype=np.int64)


def _unpack_group_tokens(panel_data, group_offsets, g, D, PT=PANEL_TOKENS):
    """Inverse of the panel-major layout: recover the group's [Gp, D] tokens."""
    start, end = int(group_offsets[g]), int(group_offsets[g + 1])
    Gp = end - start
    buf = panel_data[start * D:end * D]
    return buf.reshape(Gp // PT, D, PT).transpose(0, 2, 1).reshape(Gp, D)


def test_alignment_and_offsets():
    flat, doc_starts = _make_corpus(seed=3, n_docs=57, D=32)
    panel_data, group_offsets, doc_offsets, group_doc_starts, doc_unpadded = \
        pack_corpus_panels(flat, doc_starts, target_group_tokens=512)

    # Every padded doc length and group boundary is a multiple of PANEL_TOKENS.
    assert (np.diff(doc_offsets.astype(np.int64)) % PANEL_TOKENS == 0).all()
    assert (group_offsets.astype(np.int64) % PANEL_TOKENS == 0).all()
    # Groups own consecutive whole documents covering everything exactly once.
    assert group_doc_starts[0] == 0 and group_doc_starts[-1] == len(doc_starts)
    assert (np.diff(group_doc_starts.astype(np.int64)) > 0).all()
    # Group token boundaries coincide with document boundaries.
    assert all(
        int(group_offsets[g]) == int(doc_offsets[int(group_doc_starts[g])])
        for g in range(len(group_offsets) - 1)
    )
    # Unpadded offsets are the original corpus offsets.
    assert doc_unpadded[-1] == flat.shape[0]
    assert panel_data.shape[0] == int(doc_offsets[-1]) * flat.shape[1]


def test_padding_duplicates_last_token_not_zeros():
    flat, doc_starts = _make_corpus(seed=11, n_docs=23, D=16)
    D = flat.shape[1]
    panel_data, group_offsets, doc_offsets, group_doc_starts, doc_unpadded = \
        pack_corpus_panels(flat, doc_starts, target_group_tokens=256)

    for g in range(len(group_offsets) - 1):
        tokens = _unpack_group_tokens(panel_data, group_offsets, g, D)
        for d in range(int(group_doc_starts[g]), int(group_doc_starts[g + 1])):
            ls = int(doc_offsets[d] - group_offsets[g])
            le = int(doc_offsets[d + 1] - group_offsets[g])
            s, e = int(doc_unpadded[d]), int(doc_unpadded[d + 1])
            n = e - s
            # Real tokens reproduced exactly, in order.
            np.testing.assert_array_equal(tokens[ls:ls + n], flat[s:e])
            # Padding lanes are copies of the LAST real token (never zeros).
            for j in range(ls + n, le):
                np.testing.assert_array_equal(tokens[j], flat[e - 1])


def test_padded_maxsim_scores_bit_identical():
    """max_j over a multiset is invariant under duplicating an element, so a
    MaxSim score computed on the padded corpus must equal the unpadded oracle
    exactly (no float tolerance: identical values, identical maxima)."""
    flat, doc_starts = _make_corpus(seed=29, n_docs=41, D=24)
    D = flat.shape[1]
    panel_data, group_offsets, doc_offsets, group_doc_starts, _ = \
        pack_corpus_panels(flat, doc_starts, target_group_tokens=384)

    rng = np.random.default_rng(5)
    q = rng.standard_normal((7, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    ref = exact_maxsim_scores(q, flat, doc_starts)

    got = np.zeros(len(doc_starts), dtype=np.float32)
    for g in range(len(group_offsets) - 1):
        tokens = _unpack_group_tokens(panel_data, group_offsets, g, D)
        S = q @ tokens.T
        for d in range(int(group_doc_starts[g]), int(group_doc_starts[g + 1])):
            ls = int(doc_offsets[d] - group_offsets[g])
            le = int(doc_offsets[d + 1] - group_offsets[g])
            got[d] = S[:, ls:le].max(axis=1).sum()

    np.testing.assert_array_equal(got, ref)


def test_oversized_document_gets_own_group():
    rng = np.random.default_rng(41)
    big = rng.standard_normal((900, 8)).astype(np.float32)
    small = rng.standard_normal((5, 8)).astype(np.float32)
    flat = np.concatenate([small, big, small], axis=0)
    doc_starts = np.array([0, 5, 905], dtype=np.int64)
    _, group_offsets, doc_offsets, group_doc_starts, _ = \
        pack_corpus_panels(flat, doc_starts, target_group_tokens=128)
    # Three groups: [doc0], [doc1 (oversized)], [doc2].
    assert len(group_doc_starts) == 4
    assert list(group_doc_starts) == [0, 1, 2, 3]
    # The oversized doc is intact (padded to a multiple of 16) in its group.
    assert int(doc_offsets[2] - doc_offsets[1]) == 912  # ceil(900/16)*16
