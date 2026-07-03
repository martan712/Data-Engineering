#!/usr/bin/env bash
# Run the Stage 3 experiments still pending: e07, e08, e09.
#
# e01–e06 ran 2026-07-03 on the revised fused kernels and are committed.
# e07 reruns because its earlier results timed oracle tau_seed resolution
# (a full exact scan) inside the pre-processing bar, swamping the reorder
# cost it is meant to measure — the prep timer now excludes it.  e08 reruns
# because the kernel revision (hoisted per-checkpoint resq) moves its
# wall-clock numbers.  e09 is the new bound-tightness ablation (R11).
#
# Execute from the repo root:
#   bash run_pending_experiments.sh
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Rebuild both kernels (accounting = wide-block; wall-clock = fused panel).
# ---------------------------------------------------------------------------
echo "=== Rebuild kernels ==="
make -C cpp/wide_block_maxsim_bond
make -C cpp/fused_panel_maxsim

echo ""
echo "=== Gate tests (must be green before results count) ==="
uv run pytest tests/test_fused_panel_gate.py tests/test_checkpoint_sim.py -q

# ---------------------------------------------------------------------------
# Stage 3: e08 checkpoint-set ablation (R3 — closes RQ3 with numbers)
# all 4 datasets ("wall-clock-optimal C per dataset"); fused doc-level BOND,
# 8 checkpoint sets x {natural, bond} orders, oracle policy; R2 simulator-
# predicted cells% alongside measured wall-clock
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e08: checkpoint-set ablation (all datasets) ==="
uv run python -m experiments.stage3_mechanism.e08_checkpoint_ablation scifact nfcorpus arguana scidocs

# ---------------------------------------------------------------------------
# Stage 3: e09 bound-tightness ablation (R11 — BOND-2002 H_q lesson)
# all 4 datasets; tight Cauchy-Schwarz vs cheap query-only bound on the SAME
# fused doc-level kernel x {natural, bond, pca} orders, oracle policy,
# dense_fused baseline (docs/bond2002_bound_cost_analysis.md)
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e09: bound-tightness ablation (all datasets) ==="
uv run python -m experiments.stage3_mechanism.e09_bound_tightness_ablation

echo ""
echo "=== All done ==="
