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
