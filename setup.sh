#!/usr/bin/env bash
set -euo pipefail

# setup.sh — one-shot dev environment bootstrap.
#
# Standard toolchain: uv (https://docs.astral.sh/uv/).  This script:
#   1. initialises pinned git submodules,
#   2. creates a uv-managed virtualenv (.venv),
#   3. installs bondmaxsim (editable) + dev tools,
#   4. builds the C++ per-document-oracle kernel.
#
# Re-run after cloning or after updating .gitmodules.

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found.  Install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "==> Initialising git submodules at pinned commits..."
git submodule update --init --recursive
echo "    Pinned commits are recorded in .gitmodules and the gitlink entries."
echo "    - extern/PDX       : cwida/PDX @ 93531b9  (ADSampling, no BOND)"
echo "    - extern/PDX-sigmod: cwida/PDX @ fdc62f2  (SIGMOD snapshot: BOND/BSA/ADSampling)"
echo "    - extern/CoRECT    : padas-lab-de/CoRECT @ fedf8bb2"

echo ""
echo "==> Creating uv virtualenv (.venv)..."
uv venv --python 3.12

echo ""
echo "==> Installing bondmaxsim (editable) + dev tools into .venv..."
# Core install is NumPy-only (the mechanism testbed).  Add the heavy retrieval
# stack for Stages 3-5 with:  uv pip install -e ".[dev,retrieval,faiss]"
uv pip install -e ".[dev]"

echo ""
echo "==> Building the C++ per-document-oracle kernel..."
make -C cpp/per_document_oracle

echo ""
echo "Setup complete."
echo "  Activate the venv:  source .venv/bin/activate"
echo "  Run the test gate:  uv run pytest"
echo "  (Stage 2 blocking checks: unit-norm guard + shrink=1 exact-agreement.)"
