#!/usr/bin/env bash
# Pending job: R8 — Stage 4 integration experiments on all four datasets,
# all-cores then 1T:
#   e01 fixed candidate arms  — dense_fused / seeded BOND / faiss_ivf@B /
#                               plaid@B / pdx_ivf@B (Mikel) / partitioned@B
#   e02 seeded-tau recovery   — self_bound / seed_partial / ivf_seed_* / oracle
#   e03 partitioned frontier  — brute+bond scanners over nprobe sweep,
#                               + exact-safe partition-bound probe
#
# Prereqs (the drivers build these on first use, but building them up front in
# separate processes keeps peak memory down on scidocs):
#   data/faiss_indexes/<ds>.faiss      — FAISS-IVF token index caches
#   data/plaid_indexes/<ds>/           — PyLate FastPlaid indexes
# Deps: uv pip install faiss-cpu; torch CPU wheel + pylate (pyproject
# [retrieval]/[faiss] extras); pdx_ivf arm additionally needs
#   CXX=g++ uv pip install ./extern/PDX-sigmod
# (Eigen nested submodule + Fedora linker note: extern/patches/README.md);
# e01 skips the pdx arm gracefully if pdxearch is missing.
#
# Writes:
#   results/json/stage4_integration_e0{1,2,3}_*_<ds>.json
#     (each with runs.mt and runs.1t sections merged across invocations)
#   results/figures/stage4_integration/e0{1,3}_*_<ds>.png
#
# Execute from the repo root:
#   bash run_pending_experiments.sh
# (the script tees its own log to results/logs/).
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1

mkdir -p results/logs
LOG="results/logs/r8_stage4_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== R8 Stage 4 run -> logging to $LOG ==="
date

echo ""
echo "=== Rebuild fused kernel ==="
make -C cpp/fused_panel_maxsim

echo ""
echo "=== Gate tests (must be green before results count) ==="
uv run pytest tests/test_stage4_baselines.py tests/test_fused_panel_gate.py -q

echo ""
echo "=== Stage 4 e02 (R8): seeded-tau recovery, all cores ==="
uv run python -m experiments.stage4_integration.e02_seeded_tau_recovery 0

echo ""
echo "=== Stage 4 e01 (R8): fixed candidate arms, all cores ==="
uv run python -m experiments.stage4_integration.e01_fixed_candidate_arms 0

echo ""
echo "=== Stage 4 e03 (R8): partitioned fused scan frontier, all cores ==="
uv run python -m experiments.stage4_integration.e03_partitioned_fused_scan 0

echo ""
echo "=== Stage 4 e02 (R8): seeded-tau recovery, 1T ==="
uv run python -m experiments.stage4_integration.e02_seeded_tau_recovery 1

echo ""
echo "=== Stage 4 e01 (R8): fixed candidate arms, 1T ==="
uv run python -m experiments.stage4_integration.e01_fixed_candidate_arms 1

echo ""
echo "=== Stage 4 e03 (R8): partitioned fused scan frontier, 1T ==="
uv run python -m experiments.stage4_integration.e03_partitioned_fused_scan 1

echo ""
echo "=== All done ==="
date
