#!/usr/bin/env bash
# Run all completed-but-not-yet-run experiments for Stage 2 and Stage 3.
# Execute from the repo root:
#   bash run_pending_experiments.sh
# or:
#   chmod +x run_pending_experiments.sh && ./run_pending_experiments.sh
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Rebuild wide-block kernel (applies the UB_EPSILON float32 precision fix).
# ---------------------------------------------------------------------------
echo "=== Rebuild wide-block kernel ==="
make -C cpp/wide_block_maxsim_bond

# ---------------------------------------------------------------------------
# Stage 2: s02 exact-agreement gate
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 2 s02: exact-agreement gate ==="

# nfcorpus: results exist but were produced before the UB_EPSILON kernel fix.
echo "--- nfcorpus ---"
uv run python -m experiments.stage2_testbed.s02_exact_agreement nfcorpus

# arguana: natural order failed (recall=0.9985); bond + ada never ran.
echo "--- arguana ---"
uv run python -m experiments.stage2_testbed.s02_exact_agreement arguana

# scidocs: never run.
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
