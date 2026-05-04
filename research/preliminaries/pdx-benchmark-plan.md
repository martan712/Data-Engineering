# PDX-SIGMOD Benchmark Plan

## Datasets

| Dataset | Vectors | Dims | Metric | Why |
|---|---|---|---|---|
| `agnews-mxbai-1024-euclidean` | 769K | 1024 | Euclidean | Smallest available; representative dimensionality |
| `openai-1536-angular` | 999K | 1536 | Angular | Better fit for BOND pruning (higher dims, angular distance, already in config) |

Both are run by `run_benchmarks.sh`. Add more datasets by extending the `DATASETS` array in the script and following the config steps in Phase 1.

---

## Memory strategy

Index preprocessing loads raw data (~3–6GB) and the FAISS index (~3–6GB) simultaneously — this exceeds 8GB. Benchmark runs only load one index at a time and are fine at 8GB.

Limits enforced via `systemd-run` cgroups v2 (`ulimit -v` is wrong — it limits virtual, not physical memory):

| Step | Limit |
|---|---|
| Ground truth generation | 16GB |
| IVF index building | 16GB |
| Algorithm index preprocessing | 16GB |
| Each benchmark run | 8GB |

---

## Phase 0 — Prerequisites

1. **Verify compiler and cmake:**
   ```bash
   clang++ --version   # need Clang; version 21 confirmed working
   cmake --version     # need 3.26+
   ```

2. **Python environment (`uv` only — no bare pip):**
   ```bash
   cd /home/martan/Data\ Engineering/PDX-sigmod
   uv venv .venv   # skip if .venv already exists
   uv pip install -r benchmarks/python_scripts/requirements.txt
   uv pip install "setuptools<70"   # required: pymilvus uses pkg_resources, broken in setuptools>=70
   ```

3. **Verify dataset symlinks:**
   ```bash
   ls benchmarks/datasets/downloaded/
   # Must show all 4 HDF5 files as symlinks to PDX/benchmarks/datasets/downloaded/
   ```

---

## Phase 1 — Configure new datasets

Any dataset not already in the configs needs to be added in two places.

**`benchmarks/python_scripts/setup_settings.py`** — add to `DIMENSIONALITIES`:
```python
'agnews-mxbai-1024-euclidean': 1024,   # added
```
`openai-1536-angular` is already present.

Also ensure `DATASETS` is a single-line list (required for script patching):
```python
DATASETS = ['agnews-mxbai-1024-euclidean']   # single-line format
```

**`include/utils/benchmark_utils.hpp`** — add to both `DATASETS[]` and `BSA_MULTIPLIERS_M`:
```cpp
// In DATASETS[]:
"agnews-mxbai-1024-euclidean"   // added; openai already present

// In BSA_MULTIPLIERS_M:
{"agnews-mxbai-1024-euclidean", 12},   // added; 12 matches similar-dim datasets
```

After editing `benchmark_utils.hpp`, rebuild (Phase 2).

---

## Phase 2 — Build

```bash
cd /home/martan/Data\ Engineering/PDX-sigmod
export CXX="/usr/bin/clang++"
cmake . -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"
make -j$(nproc)
```

---

## Phase 3–6 — Data setup and benchmarks

Everything from here is handled by the script. Run it once and it skips steps already done:

```bash
bash /home/martan/Data\ Engineering/research/preliminaries/run_benchmarks.sh
```

To re-run everything from scratch:
```bash
bash /home/martan/Data\ Engineering/research/preliminaries/run_benchmarks.sh --rerun
```

The script handles, per dataset:
- **Phase 3** — Ground truth generation (`datasets/ground_truth/<dataset>_10` as sentinel). Skipped for `openai-1536-angular`, which has pre-computed GT.
- **Phase 4** — Core IVF index (`datasets/nary/<dataset>-ivf` as sentinel). Also generates the queries binary.
- **Phase 5** — Algorithm indexes, one at a time (16GB each):
  - ADSampling → `datasets/adsampling_pdx/<dataset>-ivf`
  - BSA/DDC → `datasets/bsa_pdx/<dataset>-ivf`
  - BOND → `datasets/pdx/<dataset>-flat`
- **Phase 6** — Benchmark runs (8GB each), skipped if the CSV already contains rows for this dataset:
  - `BenchmarkNaryIVFLinearScan` → `IVF_BRUTEFORCE.csv`
  - `BenchmarkNaryIVFADSamplingSIMD` → `IVF_NARY_ADSAMPLING_SIMD.csv`
  - `BenchmarkPDXADSampling` → `IVF_PDX_ADSAMPLING.csv`
  - `BenchmarkPDXBSA` → `IVF_PDX_BSA.csv`
  - `BenchmarkPDXIVFBOND` → `IVF_PDX_BOND.csv`
  - `ivf_faiss.py` → `IVF_FAISS.csv`

Results are written to `research/preliminaries/benchmarks/ZEN5-Martan/` (binaries write to `PDX-sigmod/benchmarks/results/ZEN5-Martan/` first, then synced at end of script).

---

## Fallback: if a step OOM-kills at 16GB

Switch to `simplewiki-openai-3072-normalized` (260K vectors, 3072 dims) instead of mxbai. Fewer vectors = smaller peak memory during GT and IVF building, even though raw file size is similar. Add it to both configs with dim=3072 and BSA multiplier=12.
