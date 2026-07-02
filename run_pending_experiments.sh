#!/usr/bin/env bash
# Run all pending Stage 2 and Stage 3 experiments.
# Execute from the repo root:
#   bash run_pending_experiments.sh
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Rebuild wide-block kernel.
# ---------------------------------------------------------------------------
echo "=== Rebuild wide-block kernel ==="
make -C cpp/wide_block_maxsim_bond

# ---------------------------------------------------------------------------
# Stage 2: s02 exact-agreement gate
# Runs oracle kernel (scifact only, already committed) + wide-block kernel
# (all four datasets).  Results go to results/json/.
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 2 s02: exact-agreement gate ==="

echo "--- scifact ---"
uv run python -m experiments.stage2_testbed.s02_exact_agreement scifact

echo "--- nfcorpus ---"
uv run python -m experiments.stage2_testbed.s02_exact_agreement nfcorpus

echo "--- arguana ---"
uv run python -m experiments.stage2_testbed.s02_exact_agreement arguana

echo "--- scidocs ---"
uv run python -m experiments.stage2_testbed.s02_exact_agreement scidocs

# ---------------------------------------------------------------------------
# Stage 2: s03 two-mode smoke
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 2 s03: two-mode smoke (scifact, oracle + wide) ==="
uv run python -m experiments.stage2_testbed.s03_two_mode_smoke

# ---------------------------------------------------------------------------
# Stage 3: e02 pruning-rate survival curves
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e02: pruning-rate curves (scifact + nfcorpus) ==="
uv run python -m experiments.stage3_mechanism.e02_pruning_rate

echo ""
echo "=== All done ==="
