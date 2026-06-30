"""ctypes bindings to the compiled C++ kernels.

Single responsibility: locate and load the shared libraries produced by
cpp/per_document_oracle/ (and eventually cpp/wide_block_maxsim_bond/), expose
their entry points with correct ctypes signatures, and raise a clear error if
the library has not been built yet.

Ported artifact: C++ kernel ABI from
  research/preliminaries/09_maxsim_pruning/maxsim_kernels.cpp (accounting mode,
  exact-safe oracle) and
  research/preliminaries/10_maxsim_pruning_opt/maxsim_kernels.cpp (throughput mode).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §6 (accounting mode
  counts only live-set FMAs; throughput mode uses dense warmup then positional
  survivor scan; the two must be called separately), §8 item 3.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from bondmaxsim.config import REPO_ROOT

_PER_DOC_LIB_PATH = REPO_ROOT / "cpp" / "per_document_oracle" / "per_document_oracle.so"
_WIDE_BLOCK_LIB_PATH = REPO_ROOT / "cpp" / "wide_block_maxsim_bond" / "wide_block_maxsim_bond.so"


def load_per_document_oracle() -> ctypes.CDLL:
    """Load the per-document-oracle shared library (accounting + throughput modes).

    Raises
    ------
    FileNotFoundError if the library has not been built (run ./setup.sh or
    `make` in cpp/per_document_oracle/).
    """
    raise NotImplementedError


def load_wide_block_kernel() -> ctypes.CDLL:
    """Load the wide-block MaxSim BOND shared library (Stage 2 deliverable).

    Raises
    ------
    FileNotFoundError if the library has not been built.
    NotImplementedError if this kernel is not yet implemented (Stage 2 in progress).
    """
    raise NotImplementedError
