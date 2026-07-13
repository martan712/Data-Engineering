#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-/home/telle/data-engineering-pdx-clean/.venv-pdx/bin/python}"
CXX="${CXX:-clang++}"
OPT_FLAGS="${OPT_FLAGS:--O3 -march=native}"
USE_OPENMP="${USE_OPENMP:-auto}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python interpreter is not executable: ${PYTHON}" >&2
    exit 1
fi
if ! command -v "${CXX}" >/dev/null 2>&1; then
    echo "C++ compiler is not available: ${CXX}" >&2
    exit 1
fi

case "${USE_OPENMP}" in
    auto|0|1) ;;
    *)
        echo "USE_OPENMP must be auto, 0, or 1" >&2
        exit 1
        ;;
esac

read -r -a PYTHON_INCLUDES <<< "$("${PYTHON}" -m pybind11 --includes)"
read -r -a OPTIMIZATION_FLAGS <<< "${OPT_FLAGS}"
EXT_SUFFIX="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"

BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_DIR="${REPO_ROOT}/experiments/kernels"
BUILD_OUTPUT="${BUILD_DIR}/_exact_maxsim${EXT_SUFFIX}"
OUTPUT_PATH="${OUTPUT_DIR}/_exact_maxsim${EXT_SUFFIX}"
mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

OPENMP_FLAGS=()
if [[ "${USE_OPENMP}" != "0" ]]; then
    PROBE_OUTPUT="${BUILD_DIR}/openmp_probe"
    if printf '%s\n' '#include <omp.h>' \
        'int main() { return omp_get_max_threads() < 1; }' | \
        "${CXX}" -x c++ -std=c++17 -fopenmp -o "${PROBE_OUTPUT}" - \
        >/dev/null 2>&1; then
        OPENMP_FLAGS=(-fopenmp)
    elif [[ "${USE_OPENMP}" == "1" ]]; then
        echo "OpenMP was required but ${CXX} could not compile and link it" >&2
        exit 1
    else
        echo "OpenMP unavailable; building a single-threaded kernel" >&2
    fi
fi

"${CXX}" \
    "${OPTIMIZATION_FLAGS[@]}" \
    -std=c++17 \
    -fPIC \
    -Wall \
    -Wextra \
    "${OPENMP_FLAGS[@]}" \
    "${PYTHON_INCLUDES[@]}" \
    "${SCRIPT_DIR}/exact_maxsim.cpp" \
    -shared \
    "${OPENMP_FLAGS[@]}" \
    -o "${BUILD_OUTPUT}"

cp "${BUILD_OUTPUT}" "${OUTPUT_PATH}"
cd "${REPO_ROOT}"
"${PYTHON}" -c \
    'from experiments.kernels import _exact_maxsim; print(f"built { _exact_maxsim.__file__ } (OpenMP={_exact_maxsim.openmp_enabled})")'
