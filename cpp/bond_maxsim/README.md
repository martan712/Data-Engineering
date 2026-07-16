# Exact-safe BOND-MaxSim kernel

This directory contains an independently authored pybind11 C++17 index for
exact checkpoint-pruned ColBERT MaxSim search:

```text
score(query, document) = sum_i max_j dot(query_i, document_j)
```

Index construction is offline. It accepts a C-contiguous `float32` document
matrix with shape `(total_document_tokens, dimension)`, a C-contiguous `int64`
offset vector, and positive, strictly increasing `int64` checkpoints ending at
`dimension`. Every document must contain at least one token. The index owns a
document-local dimension-major copy and conservative suffix norms at every
checkpoint, so neither operation is charged to online search.

## Bound and threshold safety

After checkpoint `c`, each query-token/document-token pair uses

```text
pair_ub = partial_dot + ||query[c:]||_2 * ||document[c:]||_2 + FP allowance
doc_ub  = sum_i max_j pair_ub(i, j)
```

The implementation encloses every double partial sum with directed
`nextafter` rounding. Residual squared norms, square roots, products, and bound
sums are rounded upward. These operations provide the floating-point allowance
shown above rather than assuming that the computed partial dot is exact.

The first `seed_count` document IDs are fully scored, and `seed_count` must be
at least `k`. Only fully scored documents contribute threshold candidates. The
threshold is the kth-largest conservative lower bound, never a partial or
approximate score. A document is pruned only when its upper bound is strictly
below that threshold, so an equal-score document is retained. Pair maxima start
at negative infinity because all similarities may be negative. Final ordering
is score descending and document ID ascending.

## Python API

```python
from experiments.kernels import BondMaxSimIndex

index = BondMaxSimIndex(document_values, document_offsets, checkpoints)
result = index.search(query_values, query_offsets, k=10, seed_count=10)

# Diagnostic/candidate-seeded mode: one unique seed-ID row per packed query.
result = index.search_with_seed_ids(
    query_values,
    query_offsets,
    k=10,
    seed_ids=seed_ids,
)
```

`query_values` and `query_offsets` use the same packed dtypes and layouts. The
result dictionary contains `ids`, `scores`, `documents_pruned`,
`documents_exactly_scored`, `component_products`, `full_component_products`,
`work_ratio`, and `pruned_by_checkpoint`. IDs are deterministic `int64` arrays
and scores are double-accumulated `float64` arrays. Diagnostics are aggregated
over all packed queries.

`work_ratio` is `component_products / full_component_products`. It describes
arithmetic work avoided by pruning; it is not a measured latency speedup and
must not be reported as one.

`search_with_seed_ids` fully scores the supplied documents and derives the same
safe threshold internally. Seed selection is outside the method. Supplying the
known exact top-k is therefore a deliberately free oracle diagnostic, not an
implementable end-to-end method.

## Build and verify in WSL

From the repository root:

```bash
bash cpp/bond_maxsim/build_wsl.sh
python3 -m unittest tests.test_bond_maxsim_kernel -v
```

The build defaults to the active `python3`, `clang++`, and `-O3 -march=native`.
Set `PYTHON`, `CXX`, or `OPT_FLAGS` to override them.
`USE_OPENMP=auto` uses OpenMP when compile/link probing succeeds;
`USE_OPENMP=1` requires it and `USE_OPENMP=0` makes a serial build. The output
is `experiments/kernels/_bond_maxsim<extension-suffix>.so`.
