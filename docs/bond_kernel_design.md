# Exact-Safe BOND-MaxSim Kernel Design

## Purpose

The final BOND experiment must answer a narrower question than the historical
token-index benchmarks:

> Can dimension-by-dimension branch-and-bound reduce exact ColBERT MaxSim work
> enough to beat the same project's compiled exhaustive MaxSim kernel?

The kernel described here is independently implemented on the `Mikel` branch.
It does not copy the fused kernels from `final-research-implementation`. Its
layout follows the PDX/BOND idea of storing each document in dimension-major
order, but it is a project-local research kernel rather than an upstream PDX
feature.

## Score and bound

For query token `q_i` and document token `d_j`, after scanning dimensions
`[0, t)`, define:

```text
p_ij(t) = sum_{r < t} q_i[r] d_j[r]
Rq_i(t) = ||q_i[t:]||_2
Rd_j(t) = ||d_j[t:]||_2
```

Cauchy-Schwarz gives a pairwise upper bound:

```text
<q_i, d_j> <= p_ij(t) + Rq_i(t) Rd_j(t)
```

Therefore an upper bound for one document is:

```text
U_d(t) = sum_i max_j (p_ij(t) + Rq_i(t) Rd_j(t))
```

The maxima must start at negative infinity. Initializing them to zero is
incorrect when all token similarities are negative.

## Safe threshold

Partial scores are not used as a pruning threshold. The kernel first scores at
least `k` deterministic seed documents fully. For every fully scored document it
also computes a conservative lower bound on that exact score. The pruning
threshold is the kth-largest of those valid lower bounds and can only increase
as more documents are fully evaluated.

A live document is pruned only when:

```text
inflated_upper_bound < safe_kth_lower_bound
```

The comparison is strict. A document whose upper bound equals the threshold is
kept, preserving score ties and the deterministic document-ID tie break.

This construction is exact-safe because:

1. every seed threshold value is no greater than the corresponding true score;
2. the kth-largest lower bound cannot exceed the global kth-largest true score;
3. `U_d(t)` is no smaller than the unfinished document's true score;
4. a document below that threshold cannot enter the true top-k.

Floating-point accumulation can otherwise invalidate steps 1 or 3 near a
boundary. The implementation uses double accumulation, tracks a conservative
rounding allowance, inflates upper bounds, and deflates exact lower bounds.
Non-finite inputs are rejected.

## Data layout

Index construction is offline. Packed row-major document tokens are copied to a
document-local vertical layout:

```text
document -> dimension -> token position
```

Residual token norms are precomputed for each configured checkpoint. Online
query work includes query transposition, query residual norms, incremental dot
products, bound checks, threshold updates, and final top-k production. Index
construction is reported separately.

## Measurements

The kernel reports, per run:

- exact top-k document IDs and scores;
- documents pruned and documents fully scored;
- pruning counts at each dimension checkpoint;
- component products actually evaluated;
- component products required by exhaustive MaxSim;
- their ratio.

The component-product ratio is a work estimate, not a speedup. Bounds,
branching, heap maintenance, query transposition, and less regular memory access
all add costs that exhaustive SIMD code does not pay.

## Correctness gates

Before corpus timing, the implementation must match a float64 NumPy oracle on:

1. random normalized variable-length documents and queries;
2. all-negative similarities;
3. exact score ties and top-k boundaries;
4. a constructed workload where pruning must occur;
5. invalid dtypes, layouts, offsets, checkpoints, and non-finite values.

No wall-clock result is accepted if top-k IDs differ from exhaustive MaxSim.

## Controlled comparison

The BOND and exhaustive arms will run in one WSL process with the same packed
inputs, CPU affinity, thread count, `k`, warm-up count, and interleaved measured
repetitions. The outer timer starts with resident query embeddings and ends with
ranked document IDs. BOND index construction remains outside the online timer
and is reported separately.

The first decision run uses the existing SciFact validation/test split. A
second dataset is only justified after correctness and a stable SciFact verdict.

