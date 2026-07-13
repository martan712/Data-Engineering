# Setup Notes

Last verified: 2026-07-13.

## Host and WSL

- Host Python: 3.11.9.
- CPU: AMD Ryzen 7 5800H, 8 physical / 16 logical cores.
- WSL distribution: Ubuntu, WSL2 kernel 6.6.87.2.
- WSL Python: 3.14.4.
- WSL compiler: clang 21.1.8.
- Controlled benchmark venv:
  `/home/telle/data-engineering-pdx-clean/.venv-pdx`.

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
product and was unnecessarily expensive. The current implementation performs
ordinary double accumulation and applies a conservative `gamma_n` summation
error allowance at checkpoints. On the 100-document smoke fixture, this reduced
BOND median latency from roughly 0.49-0.62 seconds to 0.06-0.09 seconds without
changing IDs, scores within tolerance, or work diagnostics.

The native `.so` is ignored for the same ABI/CPU reasons as the exact kernel.

## Test commands

Windows/reference tests:

```powershell
python -m unittest discover -s tests -v
```

WSL complete tests:

```bash
OMP_NUM_THREADS=2 \
  /home/telle/data-engineering-pdx-clean/.venv-pdx/bin/python \
  -m unittest discover -s tests -v
```

Current result: 32 tests pass in WSL. On Windows, 20 reference tests pass; three
exact-kernel and nine BOND-kernel tests are explicitly skipped because Linux
extensions are not built there.

Controlled pilots were launched with `taskset -c 0-3`, four OpenMP/BLAS threads,
`OMP_PROC_BIND=TRUE`, and `OMP_PLACES=cores`. OpenMP may narrow the calling
thread to one core place before metadata capture; the existing pilot JSON can
therefore show `[0, 1]` for `process_cpu_affinity`. The metadata helper now also
captures `initial_cpu_affinity` at import time so clean reruns retain both the
launcher allocation and the post-kernel calling-thread affinity.
