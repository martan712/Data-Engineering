"""ctypes bindings to the wide-block MaxSim BOND C++ kernel (Stage 2 deliverable).

Single responsibility: locate and load the shared library produced by
cpp/wide_block_maxsim_bond/, expose its entry points with correct ctypes
signatures, and raise a clear error if the library has not been built yet.

Ported artifact: C++ kernel ABI from
  cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.cpp header comment for the
  full layout/algorithm description, Stage 1 §5.3.
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §6, §8 item 3.

ABI (accounting/throughput variants share the same signature):
  uint64 wide_block_maxsim_{accounting,throughput}(
      const float* group_data, const uint64_t* group_offsets, size_t n_groups,
      const uint64_t* doc_offsets, size_t n_docs,
      const uint64_t* group_doc_starts,
      const float* query, size_t m, size_t D,
      const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
      const float* Qcum, float shrink, float tau_seed, size_t K,
      uint32_t* topk_id, float* topk_score, uint64_t* stats,
      uint64_t* block_doc_live, uint64_t* block_token_live)

(The former wide_block_maxsim_brute baseline was removed in Stage 3b; the
fused panel kernel in cpp/fused_panel_maxsim/ is the dense wall-clock
baseline now.)
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.packing import DEFAULT_FETCH
from bondmaxsim.kernels._ctypes_util import (
    NativeInputError,
    PackedCorpusWide,
    csz,
    f32p,
    fp,
    load_library,
    lp,
    u32p,
    u64p,
    up,
    validate_k,
    validate_order,
    validate_qcum,
    validate_query,
)

_WIDE_BLOCK_LIB_PATH = REPO_ROOT / "cpp" / "wide_block_maxsim_bond" / "wide_block_maxsim_bond.so"


# ---------------------------------------------------------------------------
# Library loader
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
    lib = load_library(_WIDE_BLOCK_LIB_PATH, "make -C cpp/wide_block_maxsim_bond")

    # Shared wide-block signature (accounting and throughput use the same ABI):
    #   uint64 fn(f32*, u64*, size_t, u64*, size_t, u64*,
    #             f32*, size_t, size_t, u32*, u32*, size_t,
    #             f32*, float, float, size_t,
    #             u32*, f32*, u64*, u64*, u64*)
    _wide_argtypes = [
        f32p, u64p, csz,            # group_data, group_offsets, n_groups
        u64p, csz,                  # doc_offsets, n_docs
        u64p,                       # group_doc_starts
        f32p, csz, csz,             # query, m, D
        u32p, u32p, csz,            # order, fetch_schedule, n_fetch
        f32p, ctypes.c_float, ctypes.c_float, csz,  # Qcum, shrink, tau_seed, K
        u32p, f32p, u64p,           # topk_id, topk_score, stats
        u64p, u64p,                 # block_doc_live, block_token_live (nullable)
    ]
    for fname in ("wide_block_maxsim_accounting", "wide_block_maxsim_throughput"):
        fn = getattr(lib, fname)
        fn.argtypes = _wide_argtypes
        fn.restype  = ctypes.c_uint64

    return lib


# ---------------------------------------------------------------------------
# Typed Python wrappers
# ---------------------------------------------------------------------------

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
    Q = validate_query(Q)
    corpus = PackedCorpusWide.from_arrays(
        group_data, group_offsets, doc_offsets, group_doc_starts, Q.shape[1]
    )
    return _run_wide_block_validated(
        fn, corpus, Q, order, Qcum, shrink, tau_seed, K, collect_block_stats
    )


def _run_wide_block_validated(
    fn,
    corpus: PackedCorpusWide,
    Q: np.ndarray,
    order: np.ndarray,
    Qcum: np.ndarray,
    shrink: float,
    tau_seed: float,
    K: int,
    collect_block_stats: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Shared calling convention with reusable corpus validation."""
    if not isinstance(corpus, PackedCorpusWide):
        raise TypeError("corpus must be PackedCorpusWide")
    Q = validate_query(Q, corpus.dimension)
    order_u32 = validate_order(order, corpus.dimension)
    Qcum = validate_qcum(Qcum, Q, order_u32)
    K = validate_k(K, corpus.n_documents)
    if not np.isfinite(shrink) or not 0.0 <= shrink <= 1.0:
        raise NativeInputError("shrink must be finite and in [0, 1]")
    if np.isnan(tau_seed) or tau_seed == np.inf:
        raise NativeInputError("tau_seed must be finite or -inf")
    if not isinstance(collect_block_stats, (bool, np.bool_)):
        raise NativeInputError("collect_block_stats must be boolean")
    fetch     = np.ascontiguousarray(DEFAULT_FETCH, dtype=np.uint32)

    m = Q.shape[0]
    D = corpus.dimension
    n_groups = corpus.n_groups
    n_docs   = corpus.n_documents
    n_fetch  = len(fetch)

    ids    = np.empty(K, dtype=np.uint32)
    scores = np.empty(K, dtype=np.float32)
    stats  = np.zeros(3,  dtype=np.uint64)

    if collect_block_stats:
        block_doc_live   = np.zeros(n_fetch, dtype=np.uint64)
        block_token_live = np.zeros(n_fetch, dtype=np.uint64)
        bdl_ptr = lp(block_doc_live)
        btl_ptr = lp(block_token_live)
    else:
        block_doc_live = None
        block_token_live = None
        bdl_ptr = None
        btl_ptr = None

    status = fn(
        fp(corpus.data), lp(corpus.group_offsets), csz(n_groups),
        lp(corpus.doc_offsets), csz(n_docs),
        lp(corpus.group_doc_starts),
        fp(Q), csz(m), csz(D),
        up(order_u32), up(fetch), csz(n_fetch),
        fp(Qcum), ctypes.c_float(shrink), ctypes.c_float(tau_seed), csz(K),
        up(ids), fp(scores), lp(stats),
        bdl_ptr, btl_ptr,
    )
    if status == np.iinfo(np.uint64).max:
        raise RuntimeError(f"{fn.__name__} rejected the native call")
    return ids, scores, stats, block_doc_live, block_token_live


def run_wide_block_accounting_validated(
    lib, corpus, Q, order, Qcum, shrink, tau_seed, K, collect_block_stats=False
):
    """Accounting call with reusable corpus validation."""
    return _run_wide_block_validated(
        lib.wide_block_maxsim_accounting,
        corpus, Q, order, Qcum, shrink, tau_seed, K, collect_block_stats,
    )


def run_wide_block_throughput_validated(
    lib, corpus, Q, order, Qcum, shrink, tau_seed, K, collect_block_stats=False
):
    """Throughput call with reusable corpus validation."""
    return _run_wide_block_validated(
        lib.wide_block_maxsim_throughput,
        corpus, Q, order, Qcum, shrink, tau_seed, K, collect_block_stats,
    )


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
