"""Port archive ColBERT caches to the token-major .npz dataset format.

Single responsibility: one-shot conversion from object-array .npy caches
(archive/preliminaries/02_bond_variance/.cache/) to the flat token-major .npz
format consumed by bondmaxsim.data.loader.  Idempotent: re-running overwrites
the output cleanly and verifies unit-norm before writing.

Ported artifact: embedding files from
  archive/preliminaries/02_bond_variance/.cache/ (ColBERT / PyLate,
  GTE-ModernColBERT-v1, D=128, unit-norm per token).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §1.3 (doc_offsets
  partition), §4.1 (unit-norm must be verified before any downstream use).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.oracle.normalization import check_unit_norm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR: Path = (
    REPO_ROOT / "archive" / "preliminaries" / "02_bond_variance" / ".cache"
)
OUT_DIR: Path = REPO_ROOT / "data" / "embeddings"

# Archive base-name → output name mapping.
DATASETS: dict[str, str] = {
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "arguana": "ArguAna",
    "scidocs": "SCIDOCS",
}

# Number of doc tokens to sample for the unit-norm sanity check.
_NORM_SAMPLE: int = 2000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_token_array(path: Path, dataset_name: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a .npy embedding file (object array or 3-D) into (flat, starts).

    Each element / row of the loaded array is coerced to float32 [n, 128] and
    stacked into a single flat [T, 128] array.  Non-conforming elements raise
    ValueError immediately so the caller never processes corrupt data silently.

    Parameters
    ----------
    path         : path to the .npy file
    dataset_name : human-readable name for error messages
    kind         : 'doc' or 'query' for error messages

    Returns
    -------
    flat   : float32 [T, 128] — all tokens concatenated
    starts : int64  [N]       — start offset of each sequence in flat
    """
    arr = np.load(path, allow_pickle=True)

    chunks: list[np.ndarray] = []
    starts: list[int] = []
    offset = 0

    for i in range(len(arr)):
        mat = np.asarray(arr[i], dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != 128:
            raise ValueError(
                f"Dataset {dataset_name!r} {kind}[{i}]: "
                f"expected shape [*, 128] but got {mat.shape}"
            )
        starts.append(offset)
        chunks.append(mat)
        offset += mat.shape[0]

    flat = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.empty((0, 128), dtype=np.float32)
    starts_arr = np.array(starts, dtype=np.int64)
    return flat, starts_arr


# ---------------------------------------------------------------------------
# Public porting function
# ---------------------------------------------------------------------------


def port_dataset(out_name: str, archive_name: str) -> dict[str, int]:
    """Convert one dataset from the archive cache to .npz.

    Always overwrites the output file so the function is idempotent.  Verifies
    unit-norm on a random sample of document tokens before writing.

    Parameters
    ----------
    out_name     : lowercase dataset name used for the output file (e.g. 'scifact')
    archive_name : base name in the archive cache (e.g. 'SciFact')

    Returns
    -------
    Summary dict with keys: num_docs, total_doc_tokens, num_queries.
    """
    out_path = OUT_DIR / f"{out_name}.npz"

    doc_path = CACHE_DIR / f"{archive_name}_doc_embs.npy"
    qry_path = CACHE_DIR / f"{archive_name}_qry_embs.npy"

    print(f"  Loading {archive_name} docs  ... ", end="", flush=True)
    doc_values, doc_starts = _load_token_array(doc_path, archive_name, "doc")
    print(f"{len(doc_starts)} docs, {len(doc_values)} tokens")

    print(f"  Loading {archive_name} queries ... ", end="", flush=True)
    query_values, query_starts = _load_token_array(qry_path, archive_name, "query")
    print(f"{len(query_starts)} queries, {len(query_values)} tokens")

    # Unit-norm check on a sample of document tokens.
    sample_n = min(_NORM_SAMPLE, len(doc_values))
    sample_idx = np.linspace(0, len(doc_values) - 1, sample_n, dtype=int)
    stats = check_unit_norm(doc_values[sample_idx])
    if stats["n_violating"] > 0:
        raise AssertionError(
            f"{archive_name} doc tokens failed unit-norm check "
            f"(sample={sample_n}): {stats}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {out_path.name} ...", end="", flush=True)
    np.savez(
        out_path,
        doc_values=doc_values,
        doc_starts=doc_starts,
        query_values=query_values,
        query_starts=query_starts,
    )
    print(" done.")

    return {
        "num_docs": int(len(doc_starts)),
        "total_doc_tokens": int(len(doc_values)),
        "num_queries": int(len(query_starts)),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Port all four datasets from archive cache to data/embeddings/."""
    print("=== BOND-MaxSim Stage 3: Port archive embeddings to .npz ===\n")

    summaries: dict[str, dict[str, int]] = {}
    for out_name, archive_name in DATASETS.items():
        print(f"[{out_name}]")
        summaries[out_name] = port_dataset(out_name, archive_name)
        print()

    # Summary table.
    header = f"{'dataset':<12}  {'num_docs':>10}  {'total_doc_tokens':>18}  {'num_queries':>12}"
    print("--- Summary " + "-" * (len(header) - 12))
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        print(
            f"{name:<12}  {s['num_docs']:>10}  "
            f"{s['total_doc_tokens']:>18}  "
            f"{s['num_queries']:>12}"
        )


if __name__ == "__main__":
    main()
