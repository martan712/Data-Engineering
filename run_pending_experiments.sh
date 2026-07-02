#!/usr/bin/env bash
# Run all pending Stage 3 experiments (e03–e07).
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
# Stage 3: e03 dimension-order ablation
# accounting + throughput, all 4 datasets, natural/bond/pca, oracle policy
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e03: dimension-order ablation (all datasets) ==="
uv run python -m experiments.stage3_mechanism.e03_order_ablation

# ---------------------------------------------------------------------------
# Stage 3: e04 exact-safe cells/latency sweep across corpus sizes
# all 4 datasets, natural order, oracle policy
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e04: exact-safe pruning sweep (all datasets) ==="
uv run python -m experiments.stage3_mechanism.e04_exact_safe_pruning

# ---------------------------------------------------------------------------
# Stage 3: e05 approximate recall sweep (shrink 0.5–1.0)
# scifact + nfcorpus, all 3 orders, self_bound policy
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e05: approximate recall sweep (scifact + nfcorpus) ==="
uv run python -m experiments.stage3_mechanism.e05_approximate_recall_sweep

# ---------------------------------------------------------------------------
# Stage 3: e06 threshold-policy ablation
# all 4 datasets, natural order, all 3 policies
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e06: threshold-policy ablation (all datasets) ==="
uv run python -m experiments.stage3_mechanism.e06_threshold_policy_ablation

# ---------------------------------------------------------------------------
# Stage 3: e07 cache/layout sensitivity
# all 4 datasets, all 3 orders, oracle policy, all queries, 10 repeats
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e07: cache/layout sensitivity (all datasets) ==="
uv run python -m experiments.stage3_mechanism.e07_cache_layout_sensitivity

echo ""
echo "=== All done ==="
