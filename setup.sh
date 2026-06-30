#!/usr/bin/env bash
set -euo pipefail

# setup.sh — initialise pinned submodules, install the package, and (TODO)
# build the C++ kernels.  Run once after cloning, or after updating .gitmodules.

echo "==> Initialising git submodules at pinned commits..."
git submodule update --init --recursive
echo "    Pinned commits are recorded in .gitmodules and the gitlink entries."
echo "    - extern/PDX      : cwida/PDX @ 93531b9  (ADSampling, no BOND)"
echo "    - extern/PDX-sigmod: cwida/PDX @ fdc62f2  (SIGMOD snapshot: BOND/BSA/ADSampling)"
echo "    - extern/CoRECT   : padas-lab-de/CoRECT @ fedf8bb2"

echo ""
echo "==> Installing bondmaxsim package (editable)..."
pip install -e .

# ---------------------------------------------------------------------------
# TODO: build C++ kernels
# ---------------------------------------------------------------------------
# The per-document MaxSim kernels live in cpp/ and must be compiled before
# running any experiment that uses ctypes bindings in src/bondmaxsim/kernels/.
# Build instructions are in cpp/README.md.  Example (adjust compiler flags):
#
#   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -- -j$(nproc)
#   cp build/libmaxsim.so src/bondmaxsim/kernels/
#
# This section is not automated yet.
# ---------------------------------------------------------------------------

echo ""
echo "Setup complete.  Run 'pytest' to verify the package installation."
echo "See cpp/README.md for C++ kernel build instructions (TODO above)."
