# Exact MaxSim kernel

This directory contains an independent pybind11 C++17 baseline for exact
ColBERT inner-product MaxSim scoring:

```text
score(query, document) = sum_i max_j dot(query_i, document_j)
```

The kernel accepts packed document and query token matrices. Each values array
has shape `(total_tokens, dimension)` and dtype `float32`; the corresponding
offset array has dtype `int64`, starts at zero, and has one terminal offset.
The returned score matrix is ordered as `(queries, documents)`. Documents must
contain at least one token. Empty queries are allowed and score zero.

The extension exposes two output contracts:

- `maxsim_scores`: float32 products, accumulation, and output for the IVF
  reranking pipeline;
- `maxsim_scores_f64`: float64 products, accumulation, and output for a
  precision-matched exhaustive comparison with the exact-safe BOND kernel.

## Build in WSL

From the repository root:

```bash
bash cpp/exact_maxsim/build_wsl.sh
```

The script uses the active `python3` interpreter by default:

```text
python3
```

Override the interpreter or compiler when needed:

```bash
PYTHON=/path/to/python CXX=clang++ bash cpp/exact_maxsim/build_wsl.sh
```

`OPT_FLAGS` defaults to `-O3 -march=native`. `USE_OPENMP=auto` enables OpenMP
when the compiler can compile and link it; use `USE_OPENMP=0` for a serial
build or `USE_OPENMP=1` to require OpenMP. The C++ dot-product loop is marked
for OpenMP SIMD vectorization, while optimized serial builds remain available
to the compiler's normal auto-vectorizer.

The extension is written to `experiments/kernels/_exact_maxsim<suffix>.so` and
is loaded through the Python wrapper in `experiments.kernels`.

## Verify

```bash
python3 -m unittest tests.test_exact_maxsim_kernel -v
```
