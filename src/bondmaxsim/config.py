"""Module-level constants and path configuration for bondmaxsim.

Single responsibility: define shared constants (embedding dimension, default K,
pinned external commits) and Path constants that locate the repo root, extern/,
and results/ directories.  No algorithm logic lives here.

Ported artifact: path/commit information from
  docs/project_structure.md ("Where Each Used Artifact Lives" table).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §1.2 (A3: fixed
  embedding dimension D=128), §5.1 (pinned commits for PDX and PDX-sigmod).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Embedding / retrieval constants
# ---------------------------------------------------------------------------

EMBED_DIM: int = 128
"""Embedding dimension D for all token vectors (ColBERT / GTE-ModernColBERT-v1).
See Stage 1 §1.2, assumption A3."""

DEFAULT_K: int = 10
"""Default top-k for retrieval experiments."""

# ---------------------------------------------------------------------------
# Pinned external commits (git submodules under extern/)
# ---------------------------------------------------------------------------

PDX_COMMIT: str = "93531b9"
"""Pinned commit for extern/PDX (evolved layout, ADSampling only, hybrid 25/75
float32 layout with IndexPDXIVFTreeSQ8).  See Stage 1 §5.1."""

PDX_SIGMOD_COMMIT: str = "fdc62f2"
"""Pinned commit for extern/PDX-sigmod (SIGMOD snapshot; ships BOND, BSA,
ADSampling as PDXearch variants, IndexPDXBONDFlat, bench_bond harness).
See Stage 1 §5.1."""

CORECT_COMMIT: str = "fedf8bb2"
"""Pinned commit for extern/CoRECT (IR evaluation framework from Univ. Passau /
OWS.EU partners)."""

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# Repo root is three levels above this file:
#   src/bondmaxsim/config.py  ->  src/bondmaxsim/  ->  src/  ->  <root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

EXTERN_DIR: Path = REPO_ROOT / "extern"
"""Root of pinned git submodules (PDX, PDX-sigmod, CoRECT)."""

RESULTS_JSON_DIR: Path = REPO_ROOT / "results" / "json"
"""Directory for ResultRecord JSON outputs (tracked on branch)."""

RESULTS_FIGURES_DIR: Path = REPO_ROOT / "results" / "figures"
"""Directory for final figures used in the paper (tracked on branch)."""
