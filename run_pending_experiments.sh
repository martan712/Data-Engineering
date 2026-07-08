#!/usr/bin/env bash
# Overnight job: R12a — e09 bound-tightness ablation at the e08-winning LATE
# checkpoint sets.
#
# ============================================================================
# STATUS: COMPLETED 2026-07-04 (log results/logs/r12a_e09_20260704_004845.log,
# 4957 s, all 4 datasets, recall 1.0). Verdict: adopt the TIGHT bound (cheap
# prunes 0% at every late checkpoint). R12b/R12c analysis also done — see the
# plan doc R12. DO NOT re-run unless reproducing e09; it just regenerates the
# same JSON/figures. Nothing is pending in this script.
# ============================================================================
#
# Background (docs/project_b_analysis_and_research_plan.md, R12a; and
# docs/bond2002_bound_cost_analysis.md §6): e09 first ran at the default
# checkpoints C={32,64}, where e02's survival curves say almost nothing is
# prunable — and indeed NEITHER the tight nor the cheap bound pruned, so both
# lost to dense (the "third outcome"). e08 then showed the interesting regime
# is LATE checkpoints (C={112} and supersets), where the exact-safe kernel
# prunes 88-98% of documents and beats dense on 3 of 4 datasets. R12a reruns
# the tight-vs-cheap comparison at those winning sets — that is where pruning
# fires, so that is where the bound choice actually decides the doc-level
# default (bond2002 §6 criteria).
#
# The extended e09 driver sweeps CHECKPOINT_SETS = {32,64} (early control),
# {112}, {64,112}, {32,64,96,112}  x  {natural,bond,pca} orders  x
# {tight,cheap} bounds = 24 arms/dataset, best-of-10 over ALL queries, all
# cores. This is ~4x the arm count of the original e09 and does NOT fit in a
# 15-min cap; run it overnight WITHOUT a timeout. Expect ~15-20 min for
# scifact and substantially longer for scidocs (25.7k docs).
#
# Analysis is deferred: a fresh agent tomorrow reads the JSON/figures below,
# picks the winning bound per bond2002 §6, and completes R12a/b/c.
#
# Execute from the repo root (recommended: leave it running detached):
#   bash run_pending_experiments.sh 2>&1 | tee results/logs/r12a_e09_$(date +%Y%m%d_%H%M).log
# or simply
#   bash run_pending_experiments.sh
# (the script already tees its own log to results/logs/).
set -euo pipefail
cd "$(dirname "$0")"

# Stream per-arm progress live (Python buffers stdout when it is not a tty).
export PYTHONUNBUFFERED=1

mkdir -p results/logs
LOG="results/logs/r12a_e09_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== R12a overnight run -> logging to $LOG ==="
date

# ---------------------------------------------------------------------------
# Rebuild both kernels (accounting = wide-block; wall-clock = fused panel).
# ---------------------------------------------------------------------------
echo ""
echo "=== Rebuild kernels ==="
make -C cpp/wide_block_maxsim_bond
make -C cpp/fused_panel_maxsim

echo ""
echo "=== Gate tests (must be green before results count) ==="
uv run pytest tests/test_fused_panel_gate.py tests/test_checkpoint_sim.py -q

# ---------------------------------------------------------------------------
# Stage 3: e09 bound-tightness ablation at LATE checkpoints (R12a)
# all 4 datasets; tight Cauchy-Schwarz vs cheap query-only bound on the SAME
# fused doc-level kernel x {natural,bond,pca} orders x 4 checkpoint sets,
# oracle policy, dense_fused baseline.
# Writes results/json/stage3_mechanism_e09_bound_tightness_ablation_<ds>.json
# and results/figures/stage3_mechanism/e09_bound_tightness_ablation_<ds>.png
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 3 e09 (R12a): bound-tightness ablation at late checkpoints ==="
uv run python -m experiments.stage3_mechanism.e09_bound_tightness_ablation

echo ""
echo "=== All done ==="
date
