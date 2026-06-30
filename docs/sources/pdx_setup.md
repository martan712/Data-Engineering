<!-- Preserved from the local PDX checkout (cwida/PDX); this file is NOT present in upstream commit 93531b9. -->

# PDX Experiment Replication Setup

Machine: AMD Ryzen AI 7 445, AVX-512, Fedora 43, Clang 21, CMake 3.31

---

## Phase 1 — Prerequisites

### 1.1 Initialize git submodules
Populates `extern/Eigen`, `extern/SuperKMeans`, `extern/pybind11`, `extern/findFFTW`.
```sh
git submodule update --init
```

### 1.2 OpenBLAS
Already installed on Fedora (`openblas-devel`). No action needed.
To verify: `ldconfig -p | grep libopenblas.so`

### 1.3 Install PDX Python bindings (uv venv, Python 3.12)
Python 3.14 is incompatible with faiss-cpu/usearch wheels. Pin to 3.12 via uv.
```sh
uv venv --python 3.12 .venv
CXX=/usr/bin/clang++ uv pip install . --python .venv
```
Verify: `.venv/bin/python -c "import pdxearch; print(pdxearch.compiled)"`

### 1.4 Build C++ benchmarks
```sh
CXX=/usr/bin/clang++ cmake -B build -DPDX_COMPILE_BENCHMARKS=ON
cmake --build build --target benchmarks -j$(nproc)
```
Binaries land in `build/benchmarks/`: `BenchmarkPDXIVF`, `BenchmarkEndToEnd`, etc.

Note: `#pragma clang loop` warnings from SuperKMeans are harmless — the system clang
wrapper invokes GCC headers. Code compiles and runs correctly.

---

## Phase 2 — Kernel Benchmarks (no datasets required)

Compares raw PDX vs SIMD distance kernels on synthetic float32 vectors.

### Setup
```sh
cd benchmarks/kernels_playground
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv
clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -o ./kernels.so ./kernels.cpp
```

**Bug fix required:** `kernels.py` hardcodes `.dylib` (macOS). On Linux, change line 28:
```python
# Before:
cppyy.load_library('./kernels.dylib')

# After:
import sys
_lib_ext = '.dylib' if sys.platform == 'darwin' else '.so'
cppyy.load_library('./kernels' + _lib_ext)
```

### Run
```sh
# Run for each dimensionality of interest (from repo root or kernels_playground dir)
.venv/bin/python kernels.py --count 16384 --ndims 128  --k 10 --query_count 100 --warmup 10
.venv/bin/python kernels.py --count 16384 --ndims 768  --k 10 --query_count 100 --warmup 10
.venv/bin/python kernels.py --count 16384 --ndims 1024 --k 10 --query_count 100 --warmup 10
.venv/bin/python kernels.py --count 16384 --ndims 1536 --k 10 --query_count 100 --warmup 10
```

### Results (AMD Ryzen AI 7 445, AVX-512, 100 queries, 16384 vectors)

| Metric | dim | SIMD (ms) | PDX (ms) | Speedup |
|--------|-----|-----------|----------|---------|
| IP     | 128 | 16.76     | 14.61    | 1.15x   |
| L2     | 128 | 19.56     | 14.31    | 1.37x   |
| IP     | 768 | 176.17    | 110.16   | 1.60x   |
| L2     | 768 | 117.44    | 108.49   | 1.08x   |
| IP     | 1024| 202.14    | 150.08   | 1.35x   |
| L2     | 1024| 154.62    | 152.12   | 1.02x   |
| IP     | 1536| 345.26    | 219.93   | 1.57x   |
| L2     | 1536| 266.40    | 217.80   | 1.22x   |

PDX consistently faster, with the largest gains on IP metric and higher dimensionalities.

---

## Phase 3 — Datasets and Indexes

### Install benchmark Python dependencies
```sh
# From repo root
uv pip install -r benchmarks/python_scripts/requirements.txt --python .venv
```

### Download datasets
The full-bundle Google Drive link is rate-limited for programmatic access. Download the
4 required datasets individually by file ID using gdown. Do NOT use `gdown.download_folder`
— it creates a stray folder and starts downloading everything.

```python
# Run from repo root with .venv active
import gdown, os

dest = 'benchmarks/datasets/downloaded'
datasets = {
    'openai-1536-angular.hdf5':               '1cwIY76n_HEbbZAANJVWiVkjBsIy3b9jB',  # 6.1 GB
    'instructorxl-arxiv-768.hdf5':            '1k6FCpqQiUUXYo8J158dojQG1Bquz1yWg',  # 6.9 GB
    'agnews-mxbai-1024-euclidean.hdf5':       '1SUHhKq0J_PmpRiwz7n7Wxcj4CS9-0qCf',  # 3.2 GB
    'simplewiki-openai-3072-normalized.hdf5': '1730j9FT1q1Vn4BM1JghRzrtS90KlFV7e',  # 3.2 GB
}

for name, fid in datasets.items():
    out = os.path.join(dest, name)
    if os.path.exists(out):
        print(f'Already exists: {name}')
        continue
    print(f'Downloading {name}...')
    gdown.download(id=fid, output=out, quiet=False)
```

Or as a one-liner per file:
```sh
.venv/bin/python -m gdown 1cwIY76n_HEbbZAANJVWiVkjBsIy3b9jB -O benchmarks/datasets/downloaded/openai-1536-angular.hdf5
```

### Generate indexes
`setup_data.py` already has `DOWNLOAD = False` and `GENERATE_GT = False` by default.
Set `DATASETS_TO_USE` to match what was downloaded, then run from repo root:

```sh
# Edit DATASETS_TO_USE in benchmarks/python_scripts/setup_data.py:
#   DATASETS_TO_USE = ['openai', 'arxiv', 'mxbai', 'wiki']

.venv/bin/python benchmarks/python_scripts/setup_data.py
```

This generates 4 PDX index variants (f32, u8, tree_f32, tree_u8) and 2 FAISS index
variants per dataset. Indexes land in `benchmarks/datasets/pdx/` and `benchmarks/datasets/faiss/`.

---

## Phase 4 — Main Benchmarks

All benchmarks are driven by `../preliminary/run_benchmarks.py`. Run from the PDX repo root:

```sh
cd /home/martan/Data\ Engineering/PDX

# All datasets
.venv/bin/python ../preliminary/run_benchmarks.py

# Single dataset (faster iteration)
.venv/bin/python ../preliminary/run_benchmarks.py openai
```

The script runs (in order):
1. `BenchmarkPDXIVF` for all 4 PDX index types (`pdx_f32`, `pdx_tree_f32`, `pdx_u8`, `pdx_tree_u8`)
2. `BenchmarkEndToEnd` with `nprobe=0` (exhaustive) for all index types
3. `ivf_faiss.py` and `ivf_faiss_sq8.py` (FAISS baselines)

All benchmarks run single-threaded (`OMP_NUM_THREADS=1`) to match paper conditions.

### Output CSV format
C++ benchmarks write to `benchmarks/results/DEFAULT/` with columns:
```
dataset, algorithm, avg, max, min, recall, ivf_nprobe, epsilon,
knn, n_queries, selectivity, num_measure_runs, avg_all, max_all, min_all
```
Python (FAISS) benchmarks write the same columns plus `ef_search, M, ef_construction`.

The key column is `avg` — mean per-query time in ms with IQR outlier trimming.

Results are also copied to `preliminary/results/` with descriptive names:
- `pdx_pdx_f32_adsampling.csv`, `pdx_pdx_tree_u8_adsampling.csv`, …
- `pdx_exhaustive.csv`
- `faiss_ivf_f32.csv`, `faiss_ivf_u8.csv`

---

## Phase 5 — Visualization

```sh
.venv/bin/python ../preliminary/visualize.py
```

Reads from `preliminary/results/`, writes PNG bar charts to `preliminary/plots/`:

| File | Experiment |
|---|---|
| `ivf_pdx_vs_faiss.png` | IVF — PDX f32 vs FAISS f32 |
| `ivf2_pdx_tree_u8_vs_faiss.png` | Two-level IVF (IVF₂) — PDX tree u8 vs FAISS u8 |
| `exhaustive_pdx_vs_faiss.png` | Exhaustive search + IVF |
| `all_index_types.png` | All 4 PDX variants + both FAISS baselines |

Chart layout: x-axis = recall targets (R@10 = 0.99 / 0.95 / 0.90), y-axis = avg query
time in ms, grouped bars per dataset. Matches the paper's figure style.
