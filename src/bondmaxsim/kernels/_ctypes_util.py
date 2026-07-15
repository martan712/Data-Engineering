"""Shared ctypes helpers for the kernel bindings.

Single responsibility: numpy->ctypes pointer casts and the shared "load a
compiled kernel .so or raise a clear build hint" pattern used by both
bondmaxsim.kernels.per_document and bondmaxsim.kernels.wide_block.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# ctypes type aliases
# ---------------------------------------------------------------------------
f32p = ctypes.POINTER(ctypes.c_float)
u32p = ctypes.POINTER(ctypes.c_uint32)
u64p = ctypes.POINTER(ctypes.c_uint64)
csz  = ctypes.c_size_t
SIZE_T_MAX = 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1
UINT32_MAX = np.iinfo(np.uint32).max


class NativeInputError(ValueError):
    """A public native call received structurally unsafe input."""


def _owned(array: np.ndarray) -> np.ndarray:
    """Return immutable storage owned by a validated reusable object."""
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


def _float_array(name: str, value, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise NativeInputError(f"{name} must have rank {ndim}, got shape {array.shape}")
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise NativeInputError(f"{name} contains non-finite values")
    return result


def _unsigned_array(name: str, value, dtype, *, ndim: int = 1) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise NativeInputError(f"{name} must have rank {ndim}, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise NativeInputError(f"{name} must contain integers")
    if array.size and (np.min(array) < 0 or int(np.max(array)) > np.iinfo(dtype).max):
        raise NativeInputError(f"{name} is not representable as {np.dtype(dtype)}")
    return np.ascontiguousarray(array, dtype=dtype)


def _offsets(name: str, value) -> np.ndarray:
    result = _unsigned_array(name, value, np.uint64)
    if result.size < 1 or result[0] != 0:
        raise NativeInputError(f"{name} must start at zero")
    if np.any(result[1:] < result[:-1]):
        raise NativeInputError(f"{name} must be monotone")
    if int(result[-1]) > SIZE_T_MAX:
        raise NativeInputError(f"{name} exceeds native size_t")
    return result


def validate_query(value, dimension: int | None = None) -> np.ndarray:
    query = _float_array("query", value, ndim=2)
    if query.shape[0] == 0 or query.shape[1] == 0:
        raise NativeInputError("query must contain at least one token and dimension")
    if dimension is not None and query.shape[1] != dimension:
        raise NativeInputError(
            f"query dimension {query.shape[1]} does not match corpus dimension {dimension}"
        )
    if query.size > SIZE_T_MAX:
        raise NativeInputError("query size exceeds native size_t")
    return query


def validate_order(value, dimension: int) -> np.ndarray:
    order = _unsigned_array("order", value, np.uint32)
    if order.shape != (dimension,):
        raise NativeInputError(f"order must have shape ({dimension},)")
    if not np.array_equal(np.sort(order), np.arange(dimension, dtype=np.uint32)):
        raise NativeInputError("order must be a permutation of [0, D)")
    return order


def validate_qcum(value, query: np.ndarray, order: np.ndarray) -> np.ndarray:
    qcum = _float_array("Qcum", value, ndim=2)
    expected = (query.shape[0], query.shape[1] + 1)
    if qcum.shape != expected:
        raise NativeInputError(f"Qcum must have shape {expected}, got {qcum.shape}")
    if not np.all(qcum[:, 0] == 0):
        raise NativeInputError("Qcum must start with a zero column")
    if np.any(np.diff(qcum, axis=1) < -8 * np.finfo(np.float32).eps):
        raise NativeInputError("Qcum must be monotone along dimensions")
    expected_values = np.zeros(expected, dtype=np.float32)
    expected_values[:, 1:] = np.cumsum(query[:, order] ** 2, axis=1)
    if not np.allclose(qcum, expected_values, rtol=8e-6, atol=8e-7):
        raise NativeInputError("Qcum is inconsistent with query and order")
    return qcum


def validate_k(k: int, n_documents: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise NativeInputError("K must be an integer")
    if k <= 0 or k > n_documents:
        raise NativeInputError(f"K must be in [1, {n_documents}], got {k}")
    return int(k)


@dataclass(frozen=True)
class PackedCorpusDimMajor:
    data: np.ndarray
    doc_offsets: np.ndarray
    dimension: int

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(
            self.dimension, (int, np.integer)
        ):
            raise NativeInputError("corpus dimension must be an integer")
        dimension = int(self.dimension)
        if dimension <= 0:
            raise NativeInputError("corpus dimension must be positive")
        if dimension > UINT32_MAX:
            raise NativeInputError("corpus dimension exceeds uint32 dimension indices")
        packed = _float_array("packed document data", self.data, ndim=1)
        offsets = _offsets("doc_offsets", self.doc_offsets)
        n_documents = len(offsets) - 1
        if n_documents == 0:
            raise NativeInputError("doc_offsets must describe at least one document")
        if n_documents > UINT32_MAX:
            raise NativeInputError("document IDs exceed uint32")
        if int(offsets[-1]) > SIZE_T_MAX // dimension:
            raise NativeInputError("packed document extent overflows native size_t")
        if packed.size != int(offsets[-1]) * dimension:
            raise NativeInputError("packed document data size disagrees with offsets and D")
        object.__setattr__(self, "data", _owned(packed))
        object.__setattr__(self, "doc_offsets", _owned(offsets))
        object.__setattr__(self, "dimension", dimension)

    @classmethod
    def from_arrays(cls, data, doc_offsets, dimension: int):
        return cls(data, doc_offsets, dimension)

    @property
    def n_documents(self) -> int:
        return len(self.doc_offsets) - 1


@dataclass(frozen=True)
class PackedCorpusWide:
    data: np.ndarray
    group_offsets: np.ndarray
    doc_offsets: np.ndarray
    group_doc_starts: np.ndarray
    dimension: int

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(
            self.dimension, (int, np.integer)
        ):
            raise NativeInputError("corpus dimension must be an integer")
        packed = _float_array("wide group data", self.data, ndim=1)
        groups = _offsets("group_offsets", self.group_offsets)
        docs = _offsets("doc_offsets", self.doc_offsets)
        group_docs = _offsets("group_doc_starts", self.group_doc_starts)
        dimension = int(self.dimension)
        _validate_grouped_corpus(packed, groups, docs, group_docs, dimension)
        object.__setattr__(self, "data", _owned(packed))
        object.__setattr__(self, "group_offsets", _owned(groups))
        object.__setattr__(self, "doc_offsets", _owned(docs))
        object.__setattr__(self, "group_doc_starts", _owned(group_docs))
        object.__setattr__(self, "dimension", dimension)

    @classmethod
    def from_arrays(cls, data, group_offsets, doc_offsets, group_doc_starts, dimension):
        return cls(data, group_offsets, doc_offsets, group_doc_starts, dimension)

    @property
    def n_documents(self) -> int:
        return len(self.doc_offsets) - 1

    @property
    def n_groups(self) -> int:
        return len(self.group_offsets) - 1


@dataclass(frozen=True)
class PackedCorpusPanels(PackedCorpusWide):
    panel_tokens: int = 16

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.panel_tokens, bool) or not isinstance(
            self.panel_tokens, (int, np.integer)
        ):
            raise NativeInputError("panel_tokens must be an integer")
        panel_tokens = int(self.panel_tokens)
        if panel_tokens <= 0:
            raise NativeInputError("panel_tokens must be positive")
        if np.any(self.group_offsets % panel_tokens) or np.any(
            self.doc_offsets % panel_tokens
        ):
            raise NativeInputError("panel offsets must be aligned to complete panels")
        object.__setattr__(self, "panel_tokens", panel_tokens)

    @classmethod
    def from_arrays(
        cls, data, group_offsets, doc_offsets, group_doc_starts, dimension, panel_tokens=16
    ):
        return cls(
            data, group_offsets, doc_offsets, group_doc_starts, dimension, panel_tokens
        )


def _validate_grouped_corpus(data, groups, docs, group_docs, dimension) -> None:
    if dimension <= 0:
        raise NativeInputError("corpus dimension must be positive")
    if dimension > UINT32_MAX:
        raise NativeInputError("corpus dimension exceeds uint32 dimension indices")
    if len(groups) < 2:
        raise NativeInputError("group_offsets must describe at least one group")
    if len(group_docs) != len(groups):
        raise NativeInputError("group metadata lengths disagree")
    n_documents = len(docs) - 1
    if n_documents == 0:
        raise NativeInputError("doc_offsets must describe at least one document")
    if n_documents > UINT32_MAX:
        raise NativeInputError("document IDs exceed uint32")
    if int(group_docs[-1]) != n_documents:
        raise NativeInputError("group_doc_starts terminal value must equal n_documents")
    if int(groups[-1]) != int(docs[-1]):
        raise NativeInputError("group and document terminal offsets disagree")
    if int(groups[-1]) > SIZE_T_MAX // dimension:
        raise NativeInputError("packed corpus extent overflows native size_t")
    if data.size != int(groups[-1]) * dimension:
        raise NativeInputError("packed corpus size disagrees with offsets and D")
    indices = group_docs.astype(np.intp, copy=False)
    if not np.array_equal(docs[indices], groups):
        raise NativeInputError("group boundaries do not coincide with document boundaries")


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
