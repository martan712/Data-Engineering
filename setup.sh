#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./setup.sh --dev [--frozen]
  ./setup.sh --full --frozen

--dev   Synchronize the core/test environment and build native kernels.
--full  Initialize pinned submodules, synchronize all extras, check the paper
        toolchain, build native kernels, and emit an environment report.
--frozen  Refuse to change uv.lock. Required for --full.

Set BUILD=portable (default), paper-native, or sanitize for native builds.
EOF
}

mode=""
frozen=0
for argument in "$@"; do
  case "$argument" in
    --dev|--full)
      if [[ -n "$mode" ]]; then
        echo "ERROR: choose exactly one of --dev or --full" >&2
        exit 2
      fi
      mode="${argument#--}"
      ;;
    --frozen) frozen=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $argument" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$mode" ]]; then
  usage >&2
  exit 2
fi
if [[ "$mode" == "full" && "$frozen" -ne 1 ]]; then
  echo "ERROR: --full requires --frozen" >&2
  exit 2
fi

for command in uv git make; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $command" >&2
    exit 1
  fi
done

cxx="${CXX:-}"
if [[ -z "$cxx" ]]; then
  cxx="$(command -v clang++ || command -v g++ || true)"
fi
if [[ -z "$cxx" ]]; then
  echo "ERROR: no C++ compiler found (tried CXX, clang++, and g++)" >&2
  exit 1
fi

build="${BUILD:-portable}"
case "$build" in
  portable|paper-native|sanitize) ;;
  *) echo "ERROR: BUILD must be portable, paper-native, or sanitize" >&2; exit 2 ;;
esac

if [[ "$mode" == "full" ]]; then
  for command in latexmk lscpu; do
    if ! command -v "$command" >/dev/null 2>&1; then
      echo "ERROR: --full requires $command" >&2
      exit 1
    fi
  done
  available_kib="$(df -Pk . | awk 'NR == 2 {print $4}')"
  memory_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
  echo "Available workspace storage: $((available_kib / 1024 / 1024)) GiB"
  echo "Physical memory: $((memory_kib / 1024 / 1024)) GiB"
  echo "Full data generation needs substantially more storage than artifact verification."
  git submodule update --init --recursive
fi

sync_args=(sync)
if [[ "$frozen" -eq 1 ]]; then
  sync_args+=(--frozen)
fi
if [[ "$mode" == "full" ]]; then
  sync_args+=(--all-extras)
else
  sync_args+=(--extra dev)
fi

echo "Synchronizing Python environment with uv ($(uv --version))..."
uv "${sync_args[@]}"

echo "Building native kernels with BUILD=$build..."
make -C cpp/per_document_oracle BUILD="$build" CXX="$cxx"
make -C cpp/wide_block_maxsim_bond BUILD="$build" CXX="$cxx"
make -C cpp/fused_panel_maxsim BUILD="$build" CXX="$cxx"

mkdir -p artifacts/environment
.venv/bin/python -m bondmaxsim.experiments.environment \
  --output artifacts/environment/setup.json \
  --build-profile "$build"

echo "Setup complete: mode=$mode, frozen=$frozen, build=$build"
