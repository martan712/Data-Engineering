"""Shared ctypes helpers for the kernel bindings.

Single responsibility: numpy->ctypes pointer casts and the shared "load a
compiled kernel .so or raise a clear build hint" pattern used by both
bondmaxsim.kernels.per_document and bondmaxsim.kernels.wide_block.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# ctypes type aliases
# ---------------------------------------------------------------------------
f32p = ctypes.POINTER(ctypes.c_float)
u32p = ctypes.POINTER(ctypes.c_uint32)
u64p = ctypes.POINTER(ctypes.c_uint64)
csz  = ctypes.c_size_t


def fp(a: np.ndarray) -> ctypes.POINTER:
    return a.ctypes.data_as(f32p)


def up(a: np.ndarray) -> ctypes.POINTER:
    return a.ctypes.data_as(u32p)


def lp(a: np.ndarray) -> ctypes.POINTER:
    return a.ctypes.data_as(u64p)


# ---------------------------------------------------------------------------
# Library loader
# ---------------------------------------------------------------------------

def load_library(path: Path, build_hint: str) -> ctypes.CDLL:
    """Load a compiled kernel shared library, or raise with a build hint.

    Parameters
    ----------
    path       : path to the expected .so file
    build_hint : the `make -C ...` command that builds it, shown in the error

    Raises
    ------
    FileNotFoundError
        If the library has not been built yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Kernel library not found: {path}\n"
            "Build it with:\n"
            f"    {build_hint}\n"
            "or run ./setup.sh from the repo root."
        )
    return ctypes.CDLL(str(path))
