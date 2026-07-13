"""Python entry point for the optional exact MaxSim extension."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from . import _exact_maxsim
except ImportError as error:
    _exact_maxsim = None
    _IMPORT_ERROR: ImportError | None = error
else:
    _IMPORT_ERROR = None


def extension_available() -> bool:
    """Return whether the compiled extension can be imported."""
    return _exact_maxsim is not None


def exact_maxsim_scores(
    document_values: NDArray[np.float32],
    document_offsets: NDArray[np.int64],
    query_values: NDArray[np.float32],
    query_offsets: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Return exact MaxSim scores with shape `(queries, documents)`.

    Arrays are passed directly to C++ so that the native boundary can reject
    wrong dtypes, dimensions, layouts, and offsets without implicit copies.
    """
    if _exact_maxsim is None:
        message = (
            "exact MaxSim extension is unavailable; run "
            "cpp/exact_maxsim/build_wsl.sh in WSL"
        )
        raise ImportError(message) from _IMPORT_ERROR

    result: Any = _exact_maxsim.maxsim_scores(
        document_values,
        document_offsets,
        query_values,
        query_offsets,
    )
    return result


def exact_maxsim_scores_f64(
    document_values: NDArray[np.float32],
    document_offsets: NDArray[np.int64],
    query_values: NDArray[np.float32],
    query_offsets: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return exact MaxSim scores using float64 products and accumulation."""
    if _exact_maxsim is None:
        message = (
            "exact MaxSim extension is unavailable; run "
            "cpp/exact_maxsim/build_wsl.sh in WSL"
        )
        raise ImportError(message) from _IMPORT_ERROR

    result: Any = _exact_maxsim.maxsim_scores_f64(
        document_values,
        document_offsets,
        query_values,
        query_offsets,
    )
    return result
