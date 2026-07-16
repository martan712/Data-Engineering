# Setup Notes

Last verified: 2026-07-16.

## Host and WSL

- Host Python: 3.11.9.
- CPU: AMD Ryzen 7 5800H, 8 physical / 16 logical cores.
- WSL distribution: Ubuntu, WSL2 kernel 6.6.87.2.
- WSL Python: 3.14.4.
- WSL compiler: clang 21.1.8.
- Controlled benchmark venv:
  `/home/telle/data-engineering-pdx-clean/.venv-pdx`.

### Unified PLAID comparison environment

The controlled PLAID addition needs Python 3.11 because the pinned
`pylate==1.5.0` dependency requires `fast-plaid<=1.3.0.290`, for which no
Python 3.14 Linux wheel was available. A separate environment keeps PLAID,
PDX, FAISS, and both native MaxSim kernels on one interpreter:

```bash
curl -LsSf https://astral.sh/uv/install.sh | \
  env UV_INSTALL_DIR="$HOME/.local/bin" sh
"$HOME/.local/bin/uv" python install 3.11.9
"$HOME/.local/bin/uv" venv --python 3.11.9 \
  "$HOME/data-engineering-plaid/.venv"

UV="$HOME/.local/bin/uv"
PY="$HOME/data-engineering-plaid/.venv/bin/python"
"$UV" pip install --python "$PY" \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.0'
"$UV" pip install --python "$PY" \
  -r requirements.txt -r requirements-pdx.txt
CXX=clang++ "$UV" pip install --python "$PY" \
  /home/telle/data-engineering-pdx-clean/external/PDX

PYTHON="$PY" USE_OPENMP=1 bash cpp/exact_maxsim/build_wsl.sh
PYTHON="$PY" USE_OPENMP=1 bash cpp/bond_maxsim/build_wsl.sh
```

Verified core versions are Python 3.11.9, PyLate 1.5.0,
fast-plaid 1.3.0.290, CPU-only PyTorch 2.9.0, NumPy 2.4.6, FAISS 1.14.2,
and the same pinned PDX commit documented below. PLAID's Rust thread pool is
bounded with `RAYON_NUM_THREADS` in addition to the OpenMP/BLAS controls.

## PDX source

Repository: `https://github.com/cwida/PDX`

Pinned commit:

```text
fdc62f2d22b3793060abf633cb5407438c7f739b
```

The older checkout at `/home/telle/data-engineering-pdx/external/PDX` is dirty.
It contains 136 added/changed lines implementing historical `search_batch` and
`search_batch_shared_scan` prototypes in:

- `include/pdx/lib/lib.hpp`
- `python/lib.cpp`
- `python/pdxearch/index_factory.py`

Those changes do not belong in final controlled measurements. A second clean
checkout was created at `/home/telle/data-engineering-pdx-clean/external/PDX`.

## Clean build commands

```bash
git clone https://github.com/cwida/PDX \
  /home/telle/data-engineering-pdx-clean/external/PDX
git -C /home/telle/data-engineering-pdx-clean/external/PDX \
  checkout --detach fdc62f2d22b3793060abf633cb5407438c7f739b
git -C /home/telle/data-engineering-pdx-clean/external/PDX \
  submodule update --init --recursive

python3 -m venv /home/telle/data-engineering-pdx-clean/.venv-pdx
/home/telle/data-engineering-pdx-clean/.venv-pdx/bin/python \
  -m pip install --upgrade pip
/home/telle/data-engineering-pdx-clean/.venv-pdx/bin/python \
  -m pip install -r requirements-pdx.txt
CXX=clang++ /home/telle/data-engineering-pdx-clean/.venv-pdx/bin/python \
  -m pip install /home/telle/data-engineering-pdx-clean/external/PDX
```

Build result: `pdxearch==0.1` built successfully. The compiled extension imports
from:

```text
/home/telle/data-engineering-pdx-clean/.venv-pdx/lib/python3.14/site-packages/
pdxearch/compiled.cpython-314-x86_64-linux-gnu.so
```

`examples/pdxearch_simple.py` completed successfully. Its printed timing values
were treated only as a smoke test because that example is not part of the
controlled project protocol.

## Independent exact kernel

Build:

```bash
USE_OPENMP=1 bash cpp/exact_maxsim/build_wsl.sh
```

Verified properties:

- compiled with clang and OpenMP enabled;
- linked against `libomp.so.5`;
- accepts packed float32 values and int64 terminal offsets;
- releases the GIL and parallelizes query-document pairs;
- exact scores and deterministic rankings match NumPy on the fixture.

The output `.so` is Python-ABI and CPU specific and is intentionally ignored by
Git. It must be rebuilt after cloning.

## Independent exact-safe BOND-MaxSim kernel

Build:

```bash
USE_OPENMP=1 bash cpp/bond_maxsim/build_wsl.sh
```

The project-local kernel is separate from upstream PDX. It uses a
document-local dimension-major layout, precomputed residual norms, conservative
floating-point bounds, and query-level OpenMP. Index construction is offline;
query preparation, bounds, pruning, exact scoring of survivors, and top-k are
online.

The first implementation used directed `nextafter` rounding for every partial
product and was replaced by ordinary double accumulation plus a conservative
`gamma_n` summation-error allowance at checkpoints. Only the current
implementation's result artifacts are tracked, so no before/after wall-clock
claim is made. Correctness and work diagnostics are covered by automated tests.

The native `.so` is ignored for the same ABI/CPU reasons as the exact kernel.

## Test commands

Windows/reference tests:

```powershell
python -m unittest discover -s tests -v
```

WSL complete tests:

```bash
OMP_NUM_THREADS=2 \
  "$HOME/data-engineering-plaid/.venv/bin/python" \
  -m unittest discover -s tests -v
```

Final verification on 2026-07-16: all 47 tests pass in the unified WSL Python
3.11 environment, including the compiled exact and BOND kernels. On Windows,
the same suite reports 47 tests with 12 expected native-extension skips and no
failures. The skipped tests require Linux extensions and run in the WSL gate.
`uv pip check` reports all 79 packages compatible in the unified WSL
environment. The host's global Windows Python contains unrelated packages with
dependency conflicts, so it is used only for portable reference tests and not
for benchmark claims.

Final controlled runs use `taskset -c 0,2,4,6`, four OpenMP/BLAS threads,
`OMP_PROC_BIND=TRUE`, and `OMP_PLACES=cores`. This selects one hardware thread
from each of four physical cores on the recorded Ryzen topology. OpenMP narrows
the calling thread to one place after native work, so metadata stores both
`initial_cpu_affinity=[0,2,4,6]` and the post-kernel calling-thread affinity.

The checkout is shared between Windows and WSL. Windows Git had checked out
CRLF files while WSL Git initially used `core.autocrlf=false`, making a clean
tree appear dirty inside the benchmark process. The repository-local setting
was aligned before release runs:

```bash
git config core.autocrlf true
```

The repository now also tracks `.gitattributes` with fixed LF endings for JSON,
source, scripts, and Markdown, so future clones preserve result hashes without
depending on that machine-local setting.

All seven `results/final/` artifacts record commit `0a11fda`, `dirty=false`, and
the expected initial affinity. BOND final comparisons use float64 products and
accumulation in both exhaustive and exact-safe kernels; IVF reranking and the
compiled exact reference used beside PLAID retain the established float32
contract. The two selected PLAID arms were frozen from validation before the
held-out runs. The corpus-sized PLAID arms are separately labelled post-hoc
sensitivity results in the manifest and report.
