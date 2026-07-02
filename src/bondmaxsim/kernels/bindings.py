"""ctypes bindings to the compiled C++ kernels.

Single responsibility: locate and load the shared libraries produced by
cpp/per_document_oracle/ (and eventually cpp/wide_block_maxsim_bond/), expose
their entry points with correct ctypes signatures, and raise a clear error if
the library has not been built yet.

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

Wide-block kernel ABI (both variants share the same signature; see
cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.cpp header comment for the
full layout/algorithm description, Stage 1 §5.3):
  uint64 wide_block_maxsim_{accounting,throughput}(
      const float* group_data, const uint64_t* group_offsets, size_t n_groups,
      const uint64_t* doc_offsets, size_t n_docs,
      const uint64_t* group_doc_starts,
      const float* query, size_t m, size_t D,
      const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
      const float* Qcum, float shrink, float tau_seed, size_t K,
      uint32_t* topk_id, float* topk_score, uint64_t* stats,
      uint64_t* block_doc_live, uint64_t* block_token_live)
  group_data/group_offsets/doc_offsets/group_doc_starts come from
  bondmaxsim.data.packing.pack_corpus_wide.  tau_seed seeds the document
  pruning threshold (tau = max(tau_seed, topk.threshold())); -inf recovers
  the self_bound policy.  block_doc_live/block_token_live (len n_fetch) are
  NULLABLE per-fetch-boundary live-doc/live-token accumulators (pass None
  from Python to skip collecting them).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.packing import DEFAULT_FETCH

_PER_DOC_LIB_PATH = REPO_ROOT / "cpp" / "per_document_oracle" / "per_document_oracle.so"
_WIDE_BLOCK_LIB_PATH = REPO_ROOT / "cpp" / "wide_block_maxsim_bond" / "wide_block_maxsim_bond.so"

# ---------------------------------------------------------------------------
# ctypes type aliases
# ---------------------------------------------------------------------------
_f32p = ctypes.POINTER(ctypes.c_float)
_u32p = ctypes.POINTER(ctypes.c_uint32)
_u64p = ctypes.POINTER(ctypes.c_uint64)
_csz  = ctypes.c_size_t


def _fp(a: np.ndarray) -> ctypes.POINTER:
    return a.ctypes.data_as(_f32p)


def _up(a: np.ndarray) -> ctypes.POINTER:
    return a.ctypes.data_as(_u32p)


def _lp(a: np.ndarray) -> ctypes.POINTER:
    return a.ctypes.data_as(_u64p)


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
    if not _PER_DOC_LIB_PATH.exists():
        raise FileNotFoundError(
            f"Kernel library not found: {_PER_DOC_LIB_PATH}\n"
            "Build it with:\n"
            "    make -C cpp/per_document_oracle\n"
            "or run ./setup.sh from the repo root."
        )

    lib = ctypes.CDLL(str(_PER_DOC_LIB_PATH))

    # Shared knn signature (accounting and throughput use the same ABI):
    #   uint64 fn(f32*, u64*, size_t, f32*, size_t, size_t,
    #             u32*, u32*, size_t, f32*, float, size_t,
    #             u32*, f32*, u64*)
    _knn_argtypes = [
        _f32p, _u64p, _csz,       # docs, doc_offsets, n_docs
        _f32p, _csz,  _csz,       # query, m, D
        _u32p, _u32p, _csz,       # order, fetch_schedule, n_fetch
        _f32p, ctypes.c_float, _csz,  # Qcum, shrink, K
        _u32p, _f32p, _u64p,      # topk_id, topk_score, stats
    ]
    for fname in ("maxsim_knn_accounting", "maxsim_knn_throughput"):
        fn = getattr(lib, fname)
        fn.argtypes = _knn_argtypes
        fn.restype  = ctypes.c_uint64

    # maxsim_full: no order/fetch/Qcum/shrink/stats
    #   uint64 fn(f32*, u64*, size_t, f32*, size_t, size_t, size_t, u32*, f32*)
    lib.maxsim_full.argtypes = [
        _f32p, _u64p, _csz,    # docs, doc_offsets, n_docs
        _f32p, _csz,  _csz,    # query, m, D
        _csz,                  # K
        _u32p, _f32p,          # topk_id, topk_score
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
        _fp(flat), _lp(offs), _csz(n_docs),
        _fp(Q), _csz(m), _csz(Q.shape[1]),
        _up(order_u32), _up(fetch), _csz(len(fetch)),
        _fp(Qcum), ctypes.c_float(shrink), _csz(K),
        _up(ids), _fp(scores), _lp(stats),
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
        _fp(flat), _lp(offs), _csz(n_docs),
        _fp(Q), _csz(m), _csz(Q.shape[1]),
        _up(order_u32), _up(fetch), _csz(len(fetch)),
        _fp(Qcum), ctypes.c_float(shrink), _csz(K),
        _up(ids), _fp(scores), _lp(stats),
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
        _fp(flat), _lp(offs), _csz(n_docs),
        _fp(Q), _csz(m), _csz(Q.shape[1]),
        _csz(K),
        _up(ids), _fp(scores),
    )
    return ids, scores


# ---------------------------------------------------------------------------
# Wide-block MaxSim BOND kernel (Stage 2 deliverable)
# ---------------------------------------------------------------------------

def load_wide_block_kernel() -> ctypes.CDLL:
    """Load the wide-block MaxSim BOND shared library (Stage 2 deliverable).

    Sets argtypes and restype for:
      - wide_block_maxsim_accounting  (survivor-only scan; true algorithmic work)
      - wide_block_maxsim_throughput  (dense-warmup + positional survivor scan)

    Raises
    ------
    FileNotFoundError
        If the library has not been built.  Run::

            make -C cpp/wide_block_maxsim_bond

        or ``./setup.sh`` from the repo root.
    """
    if not _WIDE_BLOCK_LIB_PATH.exists():
        raise FileNotFoundError(
            f"Kernel library not found: {_WIDE_BLOCK_LIB_PATH}\n"
            "Build it with:\n"
            "    make -C cpp/wide_block_maxsim_bond\n"
            "or run ./setup.sh from the repo root."
        )

    lib = ctypes.CDLL(str(_WIDE_BLOCK_LIB_PATH))

    # Shared wide-block signature (accounting and throughput use the same ABI):
    #   uint64 fn(f32*, u64*, size_t, u64*, size_t, u64*,
    #             f32*, size_t, size_t, u32*, u32*, size_t,
    #             f32*, float, float, size_t,
    #             u32*, f32*, u64*, u64*, u64*)
    _wide_argtypes = [
        _f32p, _u64p, _csz,            # group_data, group_offsets, n_groups
        _u64p, _csz,                   # doc_offsets, n_docs
        _u64p,                         # group_doc_starts
        _f32p, _csz, _csz,             # query, m, D
        _u32p, _u32p, _csz,            # order, fetch_schedule, n_fetch
        _f32p, ctypes.c_float, ctypes.c_float, _csz,  # Qcum, shrink, tau_seed, K
        _u32p, _f32p, _u64p,           # topk_id, topk_score, stats
        _u64p, _u64p,                  # block_doc_live, block_token_live (nullable)
    ]
    for fname in ("wide_block_maxsim_accounting", "wide_block_maxsim_throughput"):
        fn = getattr(lib, fname)
        fn.argtypes = _wide_argtypes
        fn.restype  = ctypes.c_uint64

    return lib


def _run_wide_block(
    fn,
    group_data: np.ndarray,
    group_offsets: np.ndarray,
    doc_offsets: np.ndarray,
    group_doc_starts: np.ndarray,
    Q: np.ndarray,
    order: np.ndarray,
    Qcum: np.ndarray,
    shrink: float,
    tau_seed: float,
    K: int,
    collect_block_stats: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Shared calling convention for wide_block_maxsim_{accounting,throughput}."""
    group_data       = np.ascontiguousarray(group_data,       dtype=np.float32)
    group_offsets    = np.ascontiguousarray(group_offsets,    dtype=np.uint64)
    doc_offsets      = np.ascontiguousarray(doc_offsets,      dtype=np.uint64)
    group_doc_starts = np.ascontiguousarray(group_doc_starts, dtype=np.uint64)
    Q    = np.ascontiguousarray(Q,    dtype=np.float32)
    Qcum = np.ascontiguousarray(Qcum, dtype=np.float32)
    order_u32 = np.ascontiguousarray(order, dtype=np.uint32)
    fetch     = np.ascontiguousarray(DEFAULT_FETCH, dtype=np.uint32)

    m = Q.shape[0]
    D = Q.shape[1]
    n_groups = len(group_offsets) - 1
    n_docs   = len(doc_offsets) - 1
    n_fetch  = len(fetch)

    ids    = np.empty(K, dtype=np.uint32)
    scores = np.empty(K, dtype=np.float32)
    stats  = np.zeros(3,  dtype=np.uint64)

    if collect_block_stats:
        block_doc_live   = np.zeros(n_fetch, dtype=np.uint64)
        block_token_live = np.zeros(n_fetch, dtype=np.uint64)
        bdl_ptr = _lp(block_doc_live)
        btl_ptr = _lp(block_token_live)
    else:
        block_doc_live = None
        block_token_live = None
        bdl_ptr = None
        btl_ptr = None

    fn(
        _fp(group_data), _lp(group_offsets), _csz(n_groups),
        _lp(doc_offsets), _csz(n_docs),
        _lp(group_doc_starts),
        _fp(Q), _csz(m), _csz(D),
        _up(order_u32), _up(fetch), _csz(n_fetch),
        _fp(Qcum), ctypes.c_float(shrink), ctypes.c_float(tau_seed), _csz(K),
        _up(ids), _fp(scores), _lp(stats),
        bdl_ptr, btl_ptr,
    )
    return ids, scores, stats, block_doc_live, block_token_live


def run_wide_block_accounting(
    lib: ctypes.CDLL,
    group_data: np.ndarray,
    group_offsets: np.ndarray,
    doc_offsets: np.ndarray,
    group_doc_starts: np.ndarray,
    Q: np.ndarray,
    order: np.ndarray,
    Qcum: np.ndarray,
    shrink: float,
    tau_seed: float,
    K: int,
    collect_block_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Call wide_block_maxsim_accounting.

    Parameters
    ----------
    lib              : CDLL from load_wide_block_kernel()
    group_data, group_offsets, doc_offsets, group_doc_starts :
        see bondmaxsim.data.packing.pack_corpus_wide
    Q                : float32 [m, D] — query token matrix
    order            : int/uint [D] — dimension scan order
    Qcum             : float32 [m, D+1] — cumulative squared norms along order
    shrink           : float — recall knob (1.0 = exact-safe)
    tau_seed         : float — seed for the document-pruning threshold
                       (tau = max(tau_seed, topk.threshold())); -inf = self_bound
    K                : int — number of results to return
    collect_block_stats : if True, also return the per-fetch-boundary
                       live-doc/live-token accumulators (else (None, None))

    Returns
    -------
    ids               : uint32 [K]
    scores             : float32 [K]
    stats              : uint64 [3] — [cells_scanned, docs_pruned, tokens_pruned]
    block_doc_live     : uint64 [n_fetch] or None
    block_token_live   : uint64 [n_fetch] or None
    """
    return _run_wide_block(
        lib.wide_block_maxsim_accounting,
        group_data, group_offsets, doc_offsets, group_doc_starts,
        Q, order, Qcum, shrink, tau_seed, K, collect_block_stats,
    )


def run_wide_block_throughput(
    lib: ctypes.CDLL,
    group_data: np.ndarray,
    group_offsets: np.ndarray,
    doc_offsets: np.ndarray,
    group_doc_starts: np.ndarray,
    Q: np.ndarray,
    order: np.ndarray,
    Qcum: np.ndarray,
    shrink: float,
    tau_seed: float,
    K: int,
    collect_block_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Call wide_block_maxsim_throughput.  Same parameters as
    run_wide_block_accounting; uses the dense-warmup throughput kernel.
    stats[0] (cells_scanned) is NOT the true algorithmic work in this mode
    (Stage 1 §6).
    """
    return _run_wide_block(
        lib.wide_block_maxsim_throughput,
        group_data, group_offsets, doc_offsets, group_doc_starts,
        Q, order, Qcum, shrink, tau_seed, K, collect_block_stats,
    )
