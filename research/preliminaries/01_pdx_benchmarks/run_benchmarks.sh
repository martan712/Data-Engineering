#!/usr/bin/env bash
# Usage: bash run_benchmarks.sh [--rerun]
#   --rerun  ignore all sentinels and re-run everything from scratch
set -euo pipefail

export PDX_ARCH="ZEN5-Martan"

DATASETS=("agnews-mxbai-1024-euclidean" "openai-1536-angular")
PROJECT_DIR="/home/martan/Data Engineering/PDX-sigmod"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
BENCH="${PROJECT_DIR}/benchmarks"
PDX_RESULTS="${BENCH}/results/${PDX_ARCH}"   # where binaries write
RESEARCH_RESULTS="/home/martan/Data Engineering/research/preliminaries/benchmarks/${PDX_ARCH}"
SCRIPTS="${BENCH}/python_scripts"
DATASETS_DIR="${BENCH}/datasets"

MEM_SETUP="16G"   # preprocessing loads raw data + FAISS index simultaneously
MEM_BENCH="8G"    # each benchmark run loads only one index at a time

RERUN=false
for arg in "$@"; do [[ "$arg" == "--rerun" ]] && RERUN=true; done

mkdir -p "${PDX_RESULTS}" "${RESEARCH_RESULTS}"

# ── Helpers ───────────────────────────────────────────────────────────────────

# Returns 0 (true) if the CSV already contains rows for the given dataset
bench_has_results() {
    local csv="${PDX_RESULTS}/$1"
    local dataset="$2"
    [[ -f "${csv}" ]] && grep -q "^${dataset}," "${csv}"
}

# Patch setup_settings.py and setup_data.py for the current dataset/step
patch_setup() {
    local dataset="$1" generate_gt="$2" generate_ivf="$3" algo="$4"
    sed -i "s/^DATASETS = .*/DATASETS = ['${dataset}']/"   "${SCRIPTS}/setup_settings.py"
    sed -i "s/^GENERATE_GT = .*/GENERATE_GT = ${generate_gt}/"   "${SCRIPTS}/setup_data.py"
    sed -i "s/^GENERATE_IVF = .*/GENERATE_IVF = ${generate_ivf}/" "${SCRIPTS}/setup_data.py"
    if [[ -n "${algo}" ]]; then
        sed -i "s/^ALGORITHMS = .*/ALGORITHMS = ['${algo}']/" "${SCRIPTS}/setup_data.py"
    else
        sed -i "s/^ALGORITHMS = .*/ALGORITHMS = []/"          "${SCRIPTS}/setup_data.py"
    fi
}

run_setup_step() {
    local label="$1" sentinel="$2"
    echo ""
    echo "========================================"
    echo "SETUP: ${label}"
    echo "========================================"
    if [[ "${RERUN}" == false && -e "${sentinel}" ]]; then
        echo "  SKIPPED — already exists: $(basename "${sentinel}")"
        return
    fi
    systemd-run --scope -p "MemoryMax=${MEM_SETUP}" --user \
        "${PYTHON}" "${SCRIPTS}/setup_data.py"
}

run_bench() {
    local label="$1" csv="$2" dataset="$3"
    shift 3   # remaining args are the full command to execute
    echo ""
    echo "========================================"
    echo "BENCHMARK: ${label} [${dataset}]"
    echo "========================================"
    if [[ "${RERUN}" == false ]] && bench_has_results "${csv}" "${dataset}"; then
        echo "  SKIPPED — results already in ${csv}"
        return
    fi
    systemd-run --scope -p "MemoryMax=${MEM_BENCH}" --user "$@"
}

# ── Per-dataset loop ──────────────────────────────────────────────────────────

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "###################################################"
    echo "# DATASET: ${DATASET}"
    echo "###################################################"

    GT_FILE="${DATASETS_DIR}/ground_truth/${DATASET}_10"
    IVF_FILE="${DATASETS_DIR}/nary/${DATASET}-ivf"

    # Ground truth (openai already has pre-computed GT; mxbai does not)
    if [[ "${RERUN}" == false && -e "${GT_FILE}" ]]; then
        echo "Ground truth already exists, skipping."
    else
        patch_setup "${DATASET}" "True" "False" ""
        run_setup_step "Ground truth [${DATASET}]" "${GT_FILE}"
    fi

    # Core IVF index (also generates queries binary and nary layout)
    patch_setup "${DATASET}" "False" "True" ""
    run_setup_step "Core IVF index [${DATASET}]" "${IVF_FILE}"

    # Algorithm indexes
    patch_setup "${DATASET}" "False" "False" "adsampling"
    run_setup_step "ADSampling [${DATASET}]" "${DATASETS_DIR}/adsampling_pdx/${DATASET}-ivf"

    patch_setup "${DATASET}" "False" "False" "bsa"
    run_setup_step "BSA/DDC [${DATASET}]" "${DATASETS_DIR}/bsa_pdx/${DATASET}-ivf"

    patch_setup "${DATASET}" "False" "False" "bond"
    run_setup_step "BOND [${DATASET}]" "${DATASETS_DIR}/pdx/${DATASET}-flat"

    # Benchmarks
    run_bench "N-ary IVF Linear Scan" "IVF_BRUTEFORCE.csv"          "${DATASET}" \
        "${BENCH}/BenchmarkNaryIVFLinearScan" "${DATASET}"

    run_bench "N-ary ADSampling SIMD" "IVF_NARY_ADSAMPLING_SIMD.csv" "${DATASET}" \
        "${BENCH}/BenchmarkNaryIVFADSamplingSIMD" "${DATASET}"

    run_bench "PDX ADSampling"        "IVF_PDX_ADSAMPLING.csv"       "${DATASET}" \
        "${BENCH}/BenchmarkPDXADSampling" "${DATASET}"

    run_bench "PDX BSA (DDC)"         "IVF_PDX_BSA.csv"              "${DATASET}" \
        "${BENCH}/BenchmarkPDXBSA" "${DATASET}"

    run_bench "PDX BOND"              "IVF_PDX_BOND.csv"             "${DATASET}" \
        "${BENCH}/BenchmarkPDXIVFBOND" "${DATASET}" "0" "5"

    run_bench "FAISS IVF baseline"    "IVF_FAISS.csv"                "${DATASET}" \
        "${PYTHON}" "${SCRIPTS}/ivf_faiss.py" "${DATASET}"
done

# ── Sync results to research folder ──────────────────────────────────────────
cp "${PDX_RESULTS}"/*.csv "${RESEARCH_RESULTS}/" 2>/dev/null || true

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "RESULTS (${PDX_ARCH})"
echo "========================================"
for f in "${RESEARCH_RESULTS}"/*.csv; do
    [[ -e "${f}" ]] || continue
    count=$(( $(wc -l < "${f}") - 1 ))
    echo "  ${count} rows  →  $(basename "${f}")"
done
echo ""
echo "Done. Full CSVs at: ${RESEARCH_RESULTS}/"
