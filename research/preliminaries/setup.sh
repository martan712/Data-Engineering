#!/usr/bin/env bash
# Sets up CoRECT, PDX (main), and PDX-sigmod (sigmod branch) from their
# GitHub remotes and applies all local modifications present in this repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CXX="${CXX:-/usr/bin/clang++}"
export CXX

# ── CoRECT ────────────────────────────────────────────────────────────────────

echo "==> CoRECT"
CORECT="$ROOT/CoRECT"
if [ ! -d "$CORECT/.git" ]; then
    git clone https://github.com/padas-lab-de/CoRECT "$CORECT"
fi
cd "$CORECT"
uv venv --python 3.11 .venv
uv pip install . --python .venv

# ── PDX ───────────────────────────────────────────────────────────────────────

echo "==> PDX"
PDX="$ROOT/PDX"
if [ ! -d "$PDX/.git" ]; then
    git clone https://github.com/cwida/PDX.git "$PDX"
fi
cd "$PDX"
git submodule update --init

# kernels.py: make shared-lib extension platform-aware (.dylib on macOS, .so on Linux)
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("benchmarks/kernels_playground/kernels.py")
src = p.read_text()
old = "cppyy.load_library('./kernels.dylib')"
new = (
    "import sys\n"
    "_lib_ext = '.dylib' if sys.platform == 'darwin' else '.so'\n"
    "cppyy.load_library('./kernels' + _lib_ext)"
)
if old in src:
    p.write_text(src.replace(old, new))
PYEOF

# setup_data.py: use arxiv, openai, wiki instead of mxbai, arxiv, openai
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("benchmarks/python_scripts/setup_data.py")
src = p.read_text()
old = "    'mxbai',\n    'arxiv',\n    'openai',"
new = "    'arxiv',\n    'openai',\n    'wiki',"
if old in src:
    p.write_text(src.replace(old, new))
PYEOF

uv venv --python 3.12 .venv
uv pip install . --python .venv
cmake -B build -DPDX_COMPILE_BENCHMARKS=ON
cmake --build build --target benchmarks -j"$(nproc)"

# ── PDX-sigmod ────────────────────────────────────────────────────────────────

echo "==> PDX-sigmod"
PDX_SIGMOD="$ROOT/PDX-sigmod"
if [ ! -d "$PDX_SIGMOD/.git" ]; then
    git clone -b sigmod https://github.com/cwida/PDX "$PDX_SIGMOD"
fi
cd "$PDX_SIGMOD"
git submodule update --init

# setup.py: drop -static-libstdc++ (not available on Fedora; dynamic linking works)
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("setup.py")
src = p.read_text()
old = '    link_args.append("-static-libstdc++")'
new = '    pass  # static-libstdc++ not available on Fedora; dynamic linking used instead'
if old in src:
    p.write_text(src.replace(old, new))
PYEOF

# benchmark_utils.hpp: register agnews-mxbai-1024-euclidean dataset
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("include/utils/benchmark_utils.hpp")
src = p.read_text()
if '"agnews-mxbai-1024-euclidean"' not in src:
    src = src.replace(
        '"openai-1536-angular"\n    };',
        '"openai-1536-angular",\n            "agnews-mxbai-1024-euclidean"\n    };'
    )
    src = src.replace(
        '{"openai-1536-angular", 16},',
        '{"openai-1536-angular", 16},\n            {"agnews-mxbai-1024-euclidean", 12},'
    )
    p.write_text(src)
PYEOF

# setup_data.py: disable IVF generation and reduce to bond algorithm only
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("benchmarks/python_scripts/setup_data.py")
src = p.read_text()
if "GENERATE_IVF = True" in src:
    src = src.replace("GENERATE_IVF = True  # Creates IVF indexes with FAISS", "GENERATE_IVF = False")
    p.write_text(src)
if "GENERATE_GT = False  # Creates ground truth with sklearn" in src:
    src = p.read_text()
    src = src.replace("GENERATE_GT = False  # Creates ground truth with sklearn", "GENERATE_GT = False")
    p.write_text(src)
old_algs = "ALGORITHMS = [  # Choose the pruning algorithms for which indexes are going to be created\n    'adsampling',\n    'bsa',\n    'bond'\n]"
if old_algs in src:
    p.write_text(p.read_text().replace(old_algs, "ALGORITHMS = ['bond']"))
PYEOF

# setup_settings.py: use openai-1536-angular; add agnews-mxbai-1024-euclidean to DIMENSIONALITIES
python3 - <<'PYEOF'
import pathlib, re
p = pathlib.Path("benchmarks/python_scripts/setup_settings.py")
src = p.read_text()

# Collapse DATASETS list to single active entry
src = re.sub(
    r"DATASETS = \[[\s\S]*?\]",
    "DATASETS = ['openai-1536-angular']",
    src,
    count=1
)

# Add agnews to DIMENSIONALITIES if not already there
if "'agnews-mxbai-1024-euclidean'" not in src:
    src = src.replace(
        "'openai-1536-angular': 1536",
        "'openai-1536-angular': 1536,\n    'agnews-mxbai-1024-euclidean': 1024,"
    )

p.write_text(src)
PYEOF

uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv
uv pip install . --python .venv
