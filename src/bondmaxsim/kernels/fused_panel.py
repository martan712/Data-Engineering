"""ctypes bindings to the fused panel MaxSim kernel (Stage 3b deliverable).

Single responsibility: locate and load the shared library produced by
cpp/fused_panel_maxsim/, expose its entry point with correct ctypes
signatures, and raise a clear error if the library has not been built yet.

Design and rationale: docs/stage3b_fused_panel_maxsim_kernel.md.  Layout
producer: bondmaxsim.data.packing.pack_corpus_panels (16-token panel-major,
duplicate-last-token document padding).

ABI:
  uint64 fused_panel_maxsim_brute(
      const float* panel_data,
      const uint64_t* group_offsets, size_t n_groups,
      const uint64_t* doc_offsets,          // PADDED, len n_docs+1
      const uint64_t* group_doc_starts,     // len n_groups+1
      const float* query, size_t m, size_t D,
      size_t K, int n_threads,
      uint32_t* topk_id, float* topk_score)

  uint64 fused_panel_maxsim_bond(
      ... same corpus/query arguments ...,
      const uint32_t* order, const float* Qcum,   // [D], [m, D+1]
      const uint32_t* checkpoints, size_t n_checkpoints,  // NULL/0 = {32, 64}
      float shrink, float tau_seed, size_t K, int n_threads,
      uint32_t* topk_id, float* topk_score, uint64_t* stats)  // stats: u64[2]

  uint64 fused_panel_maxsim_bond_cheap(
      ... identical signature to fused_panel_maxsim_bond ...)   // stats: u64[2]
      — same document-level pruning with the QUERY-ONLY cheap bound
      UB = sum_i max_j P_ij + sum_i resq_i (resd_j <= 1 relaxation; BOND
      SIGMOD-2002 H_q lesson, docs/bond2002_bound_cost_analysis.md).

  uint64 fused_panel_maxsim_bond_token(
      ... identical signature to fused_panel_maxsim_bond ...)   // stats: u64[3]
      — adds the Stage 1 §2.4 token-level domination test (lane masks,
      dead-panel skip); the three-arm comparison instrument.
"""

from __future__ import annotations

import ctypes

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.kernels._ctypes_util import f32p, fp, load_library, lp, u32p, u64p, up, csz

_FUSED_PANEL_LIB_PATH = REPO_ROOT / "cpp" / "fused_panel_maxsim" / "fused_panel_maxsim.so"


def load_fused_panel_kernel() -> ctypes.CDLL:
    """Load the fused panel MaxSim shared library (Stage 3b deliverable).

    Raises
    ------
    FileNotFoundError
        If the library has not been built.  Run::

            make -C cpp/fused_panel_maxsim

        or ``./setup.sh`` from the repo root.
    """
    lib = load_library(_FUSED_PANEL_LIB_PATH, "make -C cpp/fused_panel_maxsim")

    lib.fused_panel_maxsim_brute.argtypes = [
        f32p,                       # panel_data
        u64p, csz,                  # group_offsets, n_groups
        u64p,                       # doc_offsets (padded)
        u64p,                       # group_doc_starts
        f32p, csz, csz,             # query, m, D
        csz, ctypes.c_int,          # K, n_threads
        u32p, f32p,                 # topk_id, topk_score
    ]
    lib.fused_panel_maxsim_brute.restype = ctypes.c_uint64

    _bond_argtypes = [
        f32p,                       # panel_data
        u64p, csz,                  # group_offsets, n_groups
        u64p,                       # doc_offsets (padded)
        u64p,                       # group_doc_starts
        f32p, csz, csz,             # query, m, D
        u32p, f32p,                 # order, Qcum
        u32p, csz,                  # checkpoints (NULL = default {32,64}), n_checkpoints
        ctypes.c_float, ctypes.c_float, csz, ctypes.c_int,  # shrink, tau_seed, K, n_threads
        u32p, f32p, u64p,           # topk_id, topk_score, stats
    ]
    for fname in ("fused_panel_maxsim_bond", "fused_panel_maxsim_bond_cheap",
                  "fused_panel_maxsim_bond_token"):
        fn = getattr(lib, fname)
        fn.argtypes = _bond_argtypes
        fn.restype  = ctypes.c_uint64

    return lib


def run_fused_panel_brute(
    lib: ctypes.CDLL,
    panel_data: np.ndarray,
    group_offsets: np.ndarray,
    doc_offsets: np.ndarray,
    group_doc_starts: np.ndarray,
    Q: np.ndarray,
    K: int,
    n_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Call fused_panel_maxsim_brute — fused dense MaxSim over panel layout.

    Parameters
    ----------
    lib             : CDLL from load_fused_panel_kernel()
    panel_data, group_offsets, doc_offsets, group_doc_starts :
        from bondmaxsim.data.packing.pack_corpus_panels (doc_offsets = the
        PADDED offsets, i.e. the third return value)
    Q               : float32 [m, D] — query token matrix (any m; the kernel
                      tiles internally in units of <= 24 query tokens)
    K               : int — number of results to return
    n_threads       : int — OpenMP threads over groups; <= 0 = OpenMP default

    Returns
    -------
    ids    : uint32 [K]
    scores : float32 [K]
    """
    panel_data       = np.ascontiguousarray(panel_data,       dtype=np.float32)
    group_offsets    = np.ascontiguousarray(group_offsets,    dtype=np.uint64)
    doc_offsets      = np.ascontiguousarray(doc_offsets,      dtype=np.uint64)
    group_doc_starts = np.ascontiguousarray(group_doc_starts, dtype=np.uint64)
    Q = np.ascontiguousarray(Q, dtype=np.float32)

    m = Q.shape[0]
    D = Q.shape[1]
    n_groups = len(group_offsets) - 1

    ids    = np.empty(K, dtype=np.uint32)
    scores = np.empty(K, dtype=np.float32)

    lib.fused_panel_maxsim_brute(
        fp(panel_data),
        lp(group_offsets), csz(n_groups),
        lp(doc_offsets),
        lp(group_doc_starts),
        fp(Q), csz(m), csz(D),
        csz(K), ctypes.c_int(n_threads),
        up(ids), fp(scores),
    )
    return ids, scores


def run_fused_panel_bond(
    lib: ctypes.CDLL,
    panel_data: np.ndarray,
    group_offsets: np.ndarray,
    doc_offsets: np.ndarray,
    group_doc_starts: np.ndarray,
    Q: np.ndarray,
    order: np.ndarray,
    Qcum: np.ndarray,
    shrink: float,
    tau_seed: float,
    K: int,
    n_threads: int = 1,
    level: str = "doc",
    checkpoints: np.ndarray | list[int] | None = None,
    bound: str = "tight",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call fused_panel_maxsim_bond (level="doc") or
    fused_panel_maxsim_bond_token (level="token") — fused MaxSim with bound
    checkpoints at the given dims (default {32, 64}; Stage 3b §5.5, R3).

    level="doc"   : document-level pruning only.
    level="token" : additionally applies the Stage 1 §2.4 token-level
                    domination test at each checkpoint (lane masks; a panel
                    whose 16 lanes all die is skipped for the remaining
                    dimension segments).  Same kernel in every other respect —
                    the three-arm comparison isolates the mechanism.
    bound="tight" : Cauchy-Schwarz envelope UB = sum_i max_j (P_ij +
                    resq_i*resd_j) — needs doc-side residuals (sumsq + sqrt).
    bound="cheap" : query-only bound UB = sum_i max_j P_ij + sum_i resq_i
                    (resd_j <= 1 relaxation; still exact-safe at shrink=1 —
                    it can only raise UB).  Doc-level only (e09 instrument);
                    BOND SIGMOD-2002 H_q lesson, see
                    docs/bond2002_bound_cost_analysis.md.

    Parameters
    ----------
    lib, panel_data, group_offsets, doc_offsets, group_doc_starts, Q, K,
    n_threads : as run_fused_panel_brute
    order     : int/uint [D] — dimension scan order (permutation of 0..D-1);
                the panel packing itself is order-agnostic (natural storage,
                permuted access within the L1-resident panel)
    Qcum      : float32 [m, D+1] — cumulative squared norms along order
                (bondmaxsim.data.packing.build_qcum)
    shrink    : float — recall knob (1.0 = exact-safe)
    tau_seed  : float — pruning-threshold seed (-inf = self_bound; the kernel
                additionally shares a rising threshold across threads)
    checkpoints : optional int sequence — bound-checkpoint dims (R3 e08
                ablation).  None = kernel default {32, 64}.  The kernel
                clamps to (0, D), sorts, dedupes, and appends D itself.

    Returns
    -------
    ids    : uint32 [K]
    scores : float32 [K]
    stats  : uint64 [3] — [cells_scanned (padded-token wall-clock convention),
             docs_pruned, tokens_pruned (0 for level="doc")]
    """
    if level not in ("doc", "token"):
        raise ValueError(f"Unknown bond level: {level!r}. Expected 'doc' or 'token'.")
    if bound not in ("tight", "cheap"):
        raise ValueError(f"Unknown bound: {bound!r}. Expected 'tight' or 'cheap'.")
    if bound == "cheap" and level != "doc":
        raise ValueError("bound='cheap' is only implemented for level='doc'.")

    panel_data       = np.ascontiguousarray(panel_data,       dtype=np.float32)
    group_offsets    = np.ascontiguousarray(group_offsets,    dtype=np.uint64)
    doc_offsets      = np.ascontiguousarray(doc_offsets,      dtype=np.uint64)
    group_doc_starts = np.ascontiguousarray(group_doc_starts, dtype=np.uint64)
    Q         = np.ascontiguousarray(Q,     dtype=np.float32)
    Qcum      = np.ascontiguousarray(Qcum,  dtype=np.float32)
    order_u32 = np.ascontiguousarray(order, dtype=np.uint32)

    m = Q.shape[0]
    D = Q.shape[1]
    n_groups = len(group_offsets) - 1

    ids    = np.empty(K, dtype=np.uint32)
    scores = np.empty(K, dtype=np.float32)
    stats  = np.zeros(3, dtype=np.uint64)

    if checkpoints is not None:
        cps_u32 = np.ascontiguousarray(checkpoints, dtype=np.uint32)
        cps_ptr, n_cps = up(cps_u32), len(cps_u32)
    else:
        cps_ptr, n_cps = None, 0

    if level == "token":
        fn = lib.fused_panel_maxsim_bond_token
    elif bound == "cheap":
        fn = lib.fused_panel_maxsim_bond_cheap
    else:
        fn = lib.fused_panel_maxsim_bond
    fn(
        fp(panel_data),
        lp(group_offsets), csz(n_groups),
        lp(doc_offsets),
        lp(group_doc_starts),
        fp(Q), csz(m), csz(D),
        up(order_u32), fp(Qcum),
        cps_ptr, csz(n_cps),
        ctypes.c_float(shrink), ctypes.c_float(tau_seed), csz(K), ctypes.c_int(n_threads),
        up(ids), fp(scores), lp(stats),
    )
    return ids, scores, stats
