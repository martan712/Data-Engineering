#!/usr/bin/env bash
# Pending job: R9 — Stage 5 CoRECT IR evaluation, the paper's FINAL SECTION.
# One driver (e01_ir_evaluation) evaluates the best method per family from the
# Stage 3–4 verdicts as retrieval systems in the proper IR framework. scifact
# and nfcorpus are already done; this runs the 2 remaining datasets (arguana,
# scidocs), all-cores then 1T:
#   e01 IR evaluation — dense_fused / bond_exact_safe / partitioned@{16,32} /
#                       faiss_ivf@B / plaid@B, at retrieval depth k=100.
#   Reports qrels metrics (nDCG@10, recall@100, MRR@10), CoRECT RC metrics,
#   recall-vs-exact, and interleaved wall-clock latency; index build cost and
#   memory are reported separately. Quality is over every evaluable test query
#   (encoded queries that carry test qrels): all test queries for scifact/
#   nfcorpus, the ported first-200 subset for arguana/scidocs.
#
# Gate: the driver runs the CoRECT wrapper smoke test on the real dense_fused
# run per dataset (agreement with our independent ranx metrics to CoRECT's own
# round(., 5)) BEFORE any RC metric is reported — a hard abort if it fails.
#
# Prereqs (the driver builds these on first use; Stage 4 already cached them):
#   data/faiss_indexes/<ds>.faiss      — FAISS-IVF token index caches
#   data/plaid_indexes/<ds>/           — PyLate FastPlaid indexes
#   data/beir_ids/<ds>/                — BEIR ID + qrels sidecars (test queries)
# Deps: uv pip install faiss-cpu; torch CPU wheel + pylate (pyproject
# [retrieval]/[faiss] extras); ranx + pytrec_eval (via extern/CoRECT).
#
# Writes:
#   results/json/stage5_corect_e01_ir_evaluation_<ds>.json
#     (each with mt/1t sections merged across the two invocations below)
#   results/figures/stage5_corect/e01_ir_evaluation_<ds>.png
#     (quality-latency frontier)
#
# NOTE: this is a long run (~15 min/dataset all-cores, longer 1T). Stdout is
# unbuffered + line-buffered so progress streams live into the tee'd log.
#
# Execute from the repo root:
#   bash run_pending_experiments.sh
# (the script tees its own log to results/logs/).
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1

mkdir -p results/logs
LOG="results/logs/r9_stage5_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== R9 Stage 5 run -> logging to $LOG ==="
date

echo ""
echo "=== Rebuild fused kernel ==="
make -C cpp/fused_panel_maxsim

echo ""
echo "=== Gate tests (must be green before results count) ==="
uv run pytest tests/test_stage4_baselines.py tests/test_fused_panel_gate.py \
    tests/test_eval_qrels.py -q

echo ""
echo "=== Stage 5 e01 (R9): CoRECT IR evaluation, all cores (arguana, scidocs) ==="
uv run python -m experiments.stage5_corect.e01_ir_evaluation 0 arguana scidocs

echo ""
echo "=== Stage 5 e01 (R9): CoRECT IR evaluation, 1T (arguana, scidocs) ==="
uv run python -m experiments.stage5_corect.e01_ir_evaluation 1 arguana scidocs

echo ""
echo "=== All done — Stage 5 complete ==="
date
