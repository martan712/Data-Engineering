"""Dataset and embedding loading utilities.

Single responsibility: load BEIR corpora, return packed token-embedding arrays
and doc-offset vectors in the layout expected by the rest of the package.

Ported artifact: load_packed_embeddings / unpack_embeddings / flatten_document_embeddings
  from research/colbert/02b_corect_bruteforce.py and
  research/preliminaries/09_maxsim_pruning/maxsim_pruning_bench.py.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §1.3 (doc_offsets
  partition), §4.1 (unit-norm must be verified here before any downstream use).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.oracle.normalization import assert_unit_norm

# Default directory for .npz embedding blobs.
_DEFAULT_DATA_DIR: Path = REPO_ROOT / "data" / "embeddings"

# Number of doc tokens to sample for the verify_norm check in load_dataset.
_NORM_SAMPLE: int = 2000


def load_packed_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load a .npz archive produced by the embedding export step.

    Returns a dict with at least 'values' (float32, shape [total_tokens, D])
    and 'offsets' (int64, shape [num_docs]).

    The returned dict also contains all other arrays present in the archive
    under their original keys (e.g. 'query_values', 'query_starts').

    Parameters
    ----------
    path : Path to a .npz file written by bondmaxsim.data.port_embeddings.

    Returns
    -------
    dict with keys including:
      'values'  : float32 [total_tokens, D]   — document token embeddings
      'offsets' : int64   [num_docs]           — start offset of each document
    """
    path = Path(path)
    data = dict(np.load(path))

    # Map from the canonical .npz keys to the legacy 'values'/'offsets' names
    # expected by unpack_embeddings and other callers.
    if "values" not in data and "doc_values" in data:
        data["values"] = data["doc_values"]
    if "offsets" not in data and "doc_starts" in data:
        data["offsets"] = data["doc_starts"]

    return data


def unpack_embeddings(
    values: np.ndarray, offsets: np.ndarray
) -> list[np.ndarray]:
    """Split flat token array back into per-document matrices.

    Parameters
    ----------
    values:  float32 array of shape [total_tokens, D]
    offsets: int64 array of shape [num_docs] — start index of each document

    Returns
    -------
    List of float32 arrays, each of shape [n_d, D].
    """
    T = len(values)
    docs: list[np.ndarray] = []
    num_docs = len(offsets)
    for i in range(num_docs):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < num_docs else T
        docs.append(values[start:end])
    return docs


def flatten_document_embeddings(
    documents: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a list of per-document token matrices into a single array.

    Returns
    -------
    flat_tokens : float32 [total_tokens, D]  — all tokens concatenated
    token_to_doc : int64 [total_tokens]       — doc index for each token
    doc_starts   : int64 [num_docs]           — start offset of each doc in flat_tokens
    """
    if not documents:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    D = documents[0].shape[1]
    flat_chunks: list[np.ndarray] = []
    starts: list[int] = []
    token_to_doc_chunks: list[np.ndarray] = []
    offset = 0

    for i, doc in enumerate(documents):
        n_d = doc.shape[0]
        starts.append(offset)
        flat_chunks.append(np.asarray(doc, dtype=np.float32))
        token_to_doc_chunks.append(np.full(n_d, i, dtype=np.int64))
        offset += n_d

    flat_tokens = np.concatenate(flat_chunks, axis=0)
    token_to_doc = np.concatenate(token_to_doc_chunks, axis=0)
    doc_starts = np.array(starts, dtype=np.int64)

    return flat_tokens, token_to_doc, doc_starts


def load_dataset(
    name: str,
    data_dir: Path | None = None,
    verify_norm: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Load a ported dataset by name and return token-major arrays.

    Parameters
    ----------
    name       : dataset name, e.g. 'scifact', 'nfcorpus', 'arguana', 'scidocs'.
    data_dir   : directory containing <name>.npz files.  Defaults to
                 <repo_root>/data/embeddings (REPO_ROOT from bondmaxsim.config).
    verify_norm: if True, assert unit-norm on a sample of doc tokens and on all
                 query tokens (raises AssertionError if violated).

    Returns
    -------
    flat_tokens : float32 [T, 128]         — all document tokens, token-major
    doc_starts  : int64  [num_docs]        — start offset of each document
    queries     : list of float32 [m, 128] — per-query token matrices

    Raises
    ------
    FileNotFoundError if the .npz is missing — run
        `python -m bondmaxsim.data.port_embeddings`
    to generate it.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    npz_path = Path(data_dir) / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {npz_path}\n"
            "Generate it by running:\n"
            "    python -m bondmaxsim.data.port_embeddings"
        )

    data = np.load(npz_path)
    flat_tokens: np.ndarray = data["doc_values"].astype(np.float32)
    doc_starts: np.ndarray = data["doc_starts"].astype(np.int64)
    query_values: np.ndarray = data["query_values"].astype(np.float32)
    query_starts: np.ndarray = data["query_starts"].astype(np.int64)

    # Reconstruct queries list from flat query array + starts.
    Tq = len(query_values)
    num_queries = len(query_starts)
    queries: list[np.ndarray] = []
    for i in range(num_queries):
        start = int(query_starts[i])
        end = int(query_starts[i + 1]) if i + 1 < num_queries else Tq
        queries.append(query_values[start:end])

    if verify_norm:
        # Sample doc tokens for norm check (avoid loading all T tokens at once).
        T = len(flat_tokens)
        sample_n = min(_NORM_SAMPLE, T)
        sample_idx = np.linspace(0, T - 1, sample_n, dtype=int)
        assert_unit_norm(flat_tokens[sample_idx])

        # Check all query tokens (typically small).
        assert_unit_norm(query_values)

    return flat_tokens, doc_starts, queries
