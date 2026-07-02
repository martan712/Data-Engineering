"""ctypes bindings to the per-document-oracle C++ kernel.

Single responsibility: locate and load the shared library produced by
cpp/per_document_oracle/, expose its entry points with correct ctypes
signatures, and raise a clear error if the library has not been built yet.

Ported artifact: C++ kernel ABI from
  archive/preliminaries/09_maxsim_pruning/maxsim_kernels.cpp (accounting mode,
  exact-safe oracle) and
  archive/preliminaries/10_maxsim_pruning_opt/maxsim_kernels.cpp (throughput mode).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §6 (accounting mode
  counts only live-set FMAs; throughput mode uses dense warmup then positional
  survivor scan; the two must be called separately), §8 item 3.

ABI (both knn variants share the same signature):
  uint64 maxsim_knn_{accounting,throughput}(
      const float* docs, const uint64_t* doc_offsets, size_t n_docs,
      const float* query, size_t m, size_t D,
      const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
      const float* Qcum, float shrink, size_t K,
      uint32_t* topk_id, float* topk_score, uint64_t* stats)
  uint64 maxsim_full(
      const float* docs, const uint64_t* doc_offsets, size_t n_docs,
      const float* query, size_t m, size_t D, size_t K,
      uint32_t* topk_id, float* topk_score)
"""

from __future__ import annotations

import ctypes

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.packing import DEFAULT_FETCH
from bondmaxsim.kernels._ctypes_util import f32p, fp, load_library, lp, u32p, u64p, up, csz

_PER_DOC_LIB_PATH = REPO_ROOT / "cpp" / "per_document_oracle" / "per_document_oracle.so"


# ---------------------------------------------------------------------------
# Library loader
# ---------------------------------------------------------------------------

def load_per_document_oracle() -> ctypes.CDLL:
    """Load the per-document-oracle shared library (accounting + throughput modes).

    Sets argtypes and restype for:
      - maxsim_knn_accounting  (exp-09 live-set-only counter)
      - maxsim_knn_throughput  (exp-10 dense-warmup + positional survivor scan)
      - maxsim_full            (brute-force baseline)

    Raises
    ------
    FileNotFoundError
        If the library has not been built.  Run::

            make -C cpp/per_document_oracle

        or ``./setup.sh`` from the repo root.
    """
    lib = load_library(_PER_DOC_LIB_PATH, "make -C cpp/per_document_oracle")

    # Shared knn signature (accounting and throughput use the same ABI):
    #   uint64 fn(f32*, u64*, size_t, f32*, size_t, size_t,
    #             u32*, u32*, size_t, f32*, float, size_t,
    #             u32*, f32*, u64*)
    _knn_argtypes = [
        f32p, u64p, csz,           # docs, doc_offsets, n_docs
        f32p, csz,  csz,           # query, m, D
        u32p, u32p, csz,           # order, fetch_schedule, n_fetch
        f32p, ctypes.c_float, csz,  # Qcum, shrink, K
        u32p, f32p, u64p,          # topk_id, topk_score, stats
    ]
    for fname in ("maxsim_knn_accounting", "maxsim_knn_throughput"):
        fn = getattr(lib, fname)
        fn.argtypes = _knn_argtypes
        fn.restype  = ctypes.c_uint64

    # maxsim_full: no order/fetch/Qcum/shrink/stats
    #   uint64 fn(f32*, u64*, size_t, f32*, size_t, size_t, size_t, u32*, f32*)
    lib.maxsim_full.argtypes = [
        f32p, u64p, csz,    # docs, doc_offsets, n_docs
        f32p, csz,  csz,    # query, m, D
        csz,                # K
        u32p, f32p,         # topk_id, topk_score
    ]
    lib.maxsim_full.restype = ctypes.c_uint64

    return lib


# ---------------------------------------------------------------------------
# Typed Python wrappers
# ---------------------------------------------------------------------------

def run_accounting(
    lib: ctypes.CDLL,
    flat: np.ndarray,
    offs: np.ndarray,
    Q: np.ndarray,
    order: np.ndarray,
    Qcum: np.ndarray,
    shrink: float,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call maxsim_knn_accounting and return (ids[K], scores[K], stats[3]).

    Parameters
    ----------
    lib   : CDLL from load_per_document_oracle()
    flat  : float32 [total_tokens * D] — per-doc dim-major packed corpus
    offs  : uint64  [n_docs + 1]       — cumulative token offsets
    Q     : float32 [m, D]             — query token matrix (contiguous)
    order : int/uint [D]               — dimension scan order
    Qcum  : float32 [m, D+1]          — cumulative squared norms along order
    shrink: float                      — recall knob (1.0 = exact-safe)
    K     : int                        — number of results to return

    Returns
    -------
    ids    : uint32 [K]
    scores : float32 [K]
    stats  : uint64 [3] — [cells_scanned, docs_pruned, tokens_pruned]
    """
    Q    = np.ascontiguousarray(Q,    dtype=np.float32)
    Qcum = np.ascontiguousarray(Qcum, dtype=np.float32)
    order_u32 = np.ascontiguousarray(order, dtype=np.uint32)
    fetch     = np.ascontiguousarray(DEFAULT_FETCH, dtype=np.uint32)

    m      = Q.shape[0]
    n_docs = len(offs) - 1

    ids    = np.empty(K, dtype=np.uint32)
    scores = np.empty(K, dtype=np.float32)
    stats  = np.zeros(3,  dtype=np.uint64)

    lib.maxsim_knn_accounting(
        fp(flat), lp(offs), csz(n_docs),
        fp(Q), csz(m), csz(Q.shape[1]),
        up(order_u32), up(fetch), csz(len(fetch)),
        fp(Qcum), ctypes.c_float(shrink), csz(K),
        up(ids), fp(scores), lp(stats),
    )
    return ids, scores, stats


def run_throughput(
    lib: ctypes.CDLL,
    flat: np.ndarray,
    offs: np.ndarray,
    Q: np.ndarray,
    order: np.ndarray,
    Qcum: np.ndarray,
    shrink: float,
    K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call maxsim_knn_throughput and return (ids[K], scores[K], stats[3]).

    Same parameters as run_accounting; uses the dense-warmup throughput kernel.
    stats[0] (cells_scanned) is NOT the true algorithmic work in this mode
    (warm dense phase inflates the count) — see Stage 1 §6.
    """
    Q    = np.ascontiguousarray(Q,    dtype=np.float32)
    Qcum = np.ascontiguousarray(Qcum, dtype=np.float32)
    order_u32 = np.ascontiguousarray(order, dtype=np.uint32)
    fetch     = np.ascontiguousarray(DEFAULT_FETCH, dtype=np.uint32)

    m      = Q.shape[0]
    n_docs = len(offs) - 1

    ids    = np.empty(K, dtype=np.uint32)
    scores = np.empty(K, dtype=np.float32)
    stats  = np.zeros(3,  dtype=np.uint64)

    lib.maxsim_knn_throughput(
        fp(flat), lp(offs), csz(n_docs),
        fp(Q), csz(m), csz(Q.shape[1]),
        up(order_u32), up(fetch), csz(len(fetch)),
        fp(Qcum), ctypes.c_float(shrink), csz(K),
        up(ids), fp(scores), lp(stats),
    )
    return ids, scores, stats


def run_full(
    lib: ctypes.CDLL,
    flat: np.ndarray,
    offs: np.ndarray,
    Q: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Call maxsim_full (brute-force baseline) and return (ids[K], scores[K]).

    Parameters
    ----------
    lib  : CDLL from load_per_document_oracle()
    flat : float32 [total_tokens * D] — per-doc dim-major packed corpus
    offs : uint64  [n_docs + 1]
    Q    : float32 [m, D]
    K    : int
    """
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    m      = Q.shape[0]
    n_docs = len(offs) - 1

    ids    = np.empty(K, dtype=np.uint32)
    scores = np.empty(K, dtype=np.float32)

    lib.maxsim_full(
        fp(flat), lp(offs), csz(n_docs),
        fp(Q), csz(m), csz(Q.shape[1]),
        csz(K),
        up(ids), fp(scores),
    )
    return ids, scores
