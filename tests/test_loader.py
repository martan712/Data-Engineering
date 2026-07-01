"""Tests for bondmaxsim.data.loader.

Unit tests for flatten_document_embeddings / unpack_embeddings round-trip on
small synthetic ragged corpora (no disk I/O required).  A separate real-data
smoke test loads the scifact dataset but skips cleanly when the .npz blob is
absent so CI without the generated files still passes.

Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §1.3 (doc_offsets
  partition), §4.1 (unit-norm precondition on all loaded embeddings).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bondmaxsim.data.loader import (
    flatten_document_embeddings,
    load_dataset,
    load_packed_embeddings,
    unpack_embeddings,
)
from bondmaxsim.config import REPO_ROOT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = REPO_ROOT / "data" / "embeddings"


def _make_ragged_docs(
    n_docs: int = 6, D: int = 16, seed: int = 42
) -> list[np.ndarray]:
    """Return a list of unit-norm float32 [n_d, D] matrices with variable length."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_docs):
        n_d = int(rng.integers(2, 9))
        v = rng.standard_normal((n_d, D)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        docs.append(v)
    return docs


# ---------------------------------------------------------------------------
# flatten_document_embeddings
# ---------------------------------------------------------------------------


def test_flatten_returns_correct_shapes():
    docs = _make_ragged_docs(n_docs=5, D=16)
    flat, token_to_doc, doc_starts = flatten_document_embeddings(docs)
    total = sum(d.shape[0] for d in docs)
    assert flat.shape == (total, 16)
    assert token_to_doc.shape == (total,)
    assert doc_starts.shape == (5,)


def test_flatten_doc_starts_values():
    docs = _make_ragged_docs(n_docs=4, D=8)
    _, _, doc_starts = flatten_document_embeddings(docs)
    # First doc always starts at 0.
    assert doc_starts[0] == 0
    # Each subsequent start equals the cumulative sum up to that doc.
    expected = 0
    for i, d in enumerate(docs):
        assert doc_starts[i] == expected
        expected += d.shape[0]


def test_flatten_token_to_doc_is_correct():
    docs = _make_ragged_docs(n_docs=4, D=8)
    _, token_to_doc, doc_starts = flatten_document_embeddings(docs)
    for i, d in enumerate(docs):
        start = int(doc_starts[i])
        end = start + d.shape[0]
        assert np.all(token_to_doc[start:end] == i), (
            f"doc {i}: token_to_doc[{start}:{end}] should all be {i}"
        )


def test_flatten_dtypes():
    docs = _make_ragged_docs()
    flat, token_to_doc, doc_starts = flatten_document_embeddings(docs)
    assert flat.dtype == np.float32
    assert token_to_doc.dtype == np.int64
    assert doc_starts.dtype == np.int64


def test_flatten_single_doc():
    doc = np.ones((3, 4), dtype=np.float32)
    flat, token_to_doc, doc_starts = flatten_document_embeddings([doc])
    assert flat.shape == (3, 4)
    assert list(doc_starts) == [0]
    assert list(token_to_doc) == [0, 0, 0]


# ---------------------------------------------------------------------------
# unpack_embeddings round-trip
# ---------------------------------------------------------------------------


def test_unpack_round_trip_exact_values():
    """flatten → unpack must reproduce exact token values for every doc."""
    docs = _make_ragged_docs(n_docs=6, D=16)
    flat, _, doc_starts = flatten_document_embeddings(docs)
    recovered = unpack_embeddings(flat, doc_starts)

    assert len(recovered) == len(docs)
    for i, (orig, rec) in enumerate(zip(docs, recovered)):
        assert orig.shape == rec.shape, f"doc {i}: shape mismatch"
        np.testing.assert_array_equal(orig, rec, err_msg=f"doc {i}: values differ")


def test_unpack_single_doc():
    v = np.arange(6, dtype=np.float32).reshape(2, 3)
    offsets = np.array([0], dtype=np.int64)
    recovered = unpack_embeddings(v, offsets)
    assert len(recovered) == 1
    np.testing.assert_array_equal(recovered[0], v)


def test_unpack_doc_starts_are_monotonically_increasing():
    docs = _make_ragged_docs(n_docs=8, D=4)
    _, _, doc_starts = flatten_document_embeddings(docs)
    assert np.all(np.diff(doc_starts) > 0), "doc_starts must be strictly increasing"


def test_unpack_lengths_match_original():
    docs = _make_ragged_docs(n_docs=5, D=8)
    flat, _, doc_starts = flatten_document_embeddings(docs)
    recovered = unpack_embeddings(flat, doc_starts)
    for i, (orig, rec) in enumerate(zip(docs, recovered)):
        assert orig.shape[0] == rec.shape[0], f"doc {i}: length mismatch"


# ---------------------------------------------------------------------------
# load_packed_embeddings
# ---------------------------------------------------------------------------


def test_load_packed_embeddings_requires_values_and_offsets(tmp_path):
    """load_packed_embeddings must return 'values' and 'offsets' keys."""
    rng = np.random.default_rng(0)
    doc_values = rng.standard_normal((20, 8)).astype(np.float32)
    doc_starts = np.array([0, 5, 12], dtype=np.int64)
    npz_path = tmp_path / "test.npz"
    np.savez(npz_path, doc_values=doc_values, doc_starts=doc_starts)

    result = load_packed_embeddings(npz_path)
    assert "values" in result
    assert "offsets" in result
    assert result["values"].shape == (20, 8)
    assert result["offsets"].shape == (3,)


# ---------------------------------------------------------------------------
# Real-data smoke test (skips if .npz not present)
# ---------------------------------------------------------------------------

_SCIFACT_NPZ = _DATA_DIR / "scifact.npz"


@pytest.mark.skipif(
    not _SCIFACT_NPZ.exists(),
    reason=f"scifact.npz not present at {_SCIFACT_NPZ}; run `python -m bondmaxsim.data.port_embeddings`",
)
def test_load_scifact_shapes_and_norm():
    """Real-data smoke: load SciFact and assert layout + unit-norm invariants."""
    flat_tokens, doc_starts, queries = load_dataset("scifact", verify_norm=True)

    # dtype
    assert flat_tokens.dtype == np.float32, "flat_tokens must be float32"
    assert doc_starts.dtype == np.int64, "doc_starts must be int64"

    # shape
    T, D = flat_tokens.shape
    assert D == 128, f"Expected D=128, got {D}"
    assert flat_tokens.ndim == 2

    # doc_starts monotonically increasing from 0
    assert doc_starts[0] == 0, "First doc_start must be 0"
    assert np.all(np.diff(doc_starts) > 0), "doc_starts must be strictly increasing"

    # All docs fit within flat_tokens
    assert int(doc_starts[-1]) < T, "Last doc_start must be < total token count"

    # queries non-empty
    assert len(queries) > 0, "queries list must be non-empty"

    # Each query is float32 [m, 128]
    for i, q in enumerate(queries):
        assert q.dtype == np.float32, f"query {i} dtype"
        assert q.ndim == 2, f"query {i} must be 2-D"
        assert q.shape[1] == 128, f"query {i} must have D=128"
