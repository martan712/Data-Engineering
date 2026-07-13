"""Python entry point for the optional exact-safe BOND-MaxSim extension."""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

try:
    from . import _bond_maxsim
except ImportError as error:
    _bond_maxsim = None
    _IMPORT_ERROR: ImportError | None = error
else:
    _IMPORT_ERROR = None


class BondMaxSimSearchResult(TypedDict):
    """Native search outputs and aggregate work diagnostics."""

    ids: NDArray[np.int64]
    scores: NDArray[np.float64]
    documents_pruned: int
    documents_exactly_scored: int
    component_products: int
    full_component_products: int
    work_ratio: float
    pruned_by_checkpoint: NDArray[np.int64]


def extension_available() -> bool:
    """Return whether the compiled BOND-MaxSim extension can be imported."""
    return _bond_maxsim is not None


class BondMaxSimIndex:
    """Offline document index for exact checkpoint-pruned MaxSim search."""

    __slots__ = ("_native",)

    def __init__(
        self,
        document_values: NDArray[np.float32],
        document_offsets: NDArray[np.int64],
        checkpoints: NDArray[np.int64],
    ) -> None:
        if _bond_maxsim is None:
            message = (
                "BOND-MaxSim extension is unavailable; run "
                "cpp/bond_maxsim/build_wsl.sh in WSL"
            )
            raise ImportError(message) from _IMPORT_ERROR
        self._native = _bond_maxsim.BondMaxSimIndex(
            document_values,
            document_offsets,
            checkpoints,
        )

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return cast(int, self._native.document_count)

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return cast(int, self._native.dimension)

    @property
    def checkpoints(self) -> tuple[int, ...]:
        """Validated checkpoint dimensions."""
        return cast(tuple[int, ...], self._native.checkpoints)

    def search(
        self,
        query_values: NDArray[np.float32],
        query_offsets: NDArray[np.int64],
        k: int,
        seed_count: int,
    ) -> BondMaxSimSearchResult:
        """Return exact top-k IDs/scores and aggregate pruning diagnostics."""
        return cast(
            BondMaxSimSearchResult,
            self._native.search(query_values, query_offsets, k, seed_count),
        )

    def search_with_seed_ids(
        self,
        query_values: NDArray[np.float32],
        query_offsets: NDArray[np.int64],
        k: int,
        seed_ids: NDArray[np.int64],
    ) -> BondMaxSimSearchResult:
        """Return exact top-k after fully scoring explicit seeds per query."""
        return cast(
            BondMaxSimSearchResult,
            self._native.search_with_seed_ids(
                query_values,
                query_offsets,
                k,
                seed_ids,
            ),
        )
