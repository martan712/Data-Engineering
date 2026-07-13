# Controlled BOND-MaxSim Pilot Results

## Status

These are decision-making pilot results from the current dirty worktree. They
are reproducible from tracked scripts and JSON files, but they are not yet a
frozen final-release claim.

The experiment compares two independently authored C++17 pybind11 kernels in
one WSL process:

- exhaustive fused MaxSim;
- exact-safe BOND-MaxSim over a document-local vertical layout.

Both use normalized `float32` ColBERT token vectors, four pinned CPUs, `k=10`,
one warm-up, five measured interleaved repetitions, and the complete online path
from resident query embeddings to top-k document IDs. BOND index construction is
offline and reported separately.

## Correctness

The BOND kernel uses Cauchy-Schwarz residual upper bounds and a threshold formed
only from conservative lower bounds of fully scored seed documents. It prunes
only on a strict inequality. The kernel passed randomized, all-negative,
tie-boundary, near-threshold, deliberately prunable, and invalid-input tests.

On every measured SciFact query, BOND returned exactly the same top-10 document
IDs as exhaustive MaxSim. The maximum score difference between double-accumulated
BOND output and the existing float-accumulated exact kernel was `7.34e-6` on the
held-out queries.

On NFCorpus, all 40 top-10 sets are identical. One query swaps two IDs whose
float exact scores are equal; their double-precision BOND scores differ by only
`5.1e-8`. The runner reports this numerical tie explicitly and rejects any
set-changing or non-tie mismatch.

## Progressive pilots

| data | exact median (s) | BOND configuration | BOND median (s) | pruned document pairs | component-product ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| 100 docs, 2 queries | 0.0052 | seed=50 | 0.0596 | 0.0% | 1.0000 |
| 500 docs, 5 queries | 0.0280 | seed=200 | 0.5441 | 0.0% | 1.0000 |
| 5,183 docs, 10 validation queries | 0.6199 | seed=500 | 11.0535 | 9.7% | 0.9759 |
| 5,183 docs, 40 held-out queries | 3.4043 | seed=500 | 42.0728 | 22.4% | 0.9547 |

The seed count was increased on the validation portion and then fixed at 500
for the 40 held-out queries. All pruning in both full-corpus runs occurred at
dimension 96. No documents were pruned at dimensions 8, 16, 32, or 64.

## Held-out timing detail

| arm | raw samples (s) | median (s) | p95 (s) |
| --- | --- | ---: | ---: |
| compiled exact | 3.5183, 2.8059, 3.4043, 2.5683, 3.6973 | 3.4043 | 3.6615 |
| exact-safe BOND, seed=500 | 40.9268, 39.5419, 45.0130, 42.0728, 42.9318 | 42.0728 | 44.5968 |

Within this runner, BOND has 12.36 times the median latency of exhaustive
MaxSim. This is a slowdown measurement, not an end-to-end retrieval speedup
claim. The offline BOND vertical-index build took 1.57 seconds.

Across 207,320 held-out query-document pairs, BOND pruned 46,526. Because every
prune occurred after 96 of 128 dimensions, it still evaluated 95.47% of the
exhaustive component products. Avoiding 4.53% of multiply-accumulate work cannot
repay residual-norm lookup, bound evaluation, threshold maintenance, branching,
and less regular memory access.

## Interpretation

This compiled result supports the earlier operation-count diagnosis: raw
ColBERT dimensions do not concentrate enough useful score information early in
the scan. A visibly nonzero document-pruning percentage is misleading here,
because pruning after 75% of the dimensions saves little arithmetic.

The result does not imply that every BOND variant must fail. The current kernel
uses natural model dimension order and deterministic prefix seed documents. Two
additional mechanism ablations separate seed quality and dimension order.

## Oracle-seed diagnostic

The explicit-seed API fully scores arbitrary seed IDs and derives the safe
threshold internally. Supplying the known exact top-10 is a deliberately free
oracle: candidate selection and exact-reference acquisition are excluded. It is
not an implementable pipeline.

On the 40 held-out raw queries, oracle seeds increase the pruned-pair fraction
from 22.4% to 36.3%, but all 75,299 prunes still occur at dimension 96. The
component-product ratio falls only from 0.9547 to 0.9243. In that oracle run,
BOND takes 47.2189 seconds versus 2.6676 seconds for its same-run exact arm, or
17.70 times the latency. Better seeds therefore do not rescue the raw dimension
order.

## PCA mechanism ablation

An uncentered global PCA rotation is fitted only on normalized document tokens
and orders components by descending second-moment eigenvalue. It is orthonormal,
so it preserves inner products in exact arithmetic. The float32 rotation has a
maximum orthogonality error of `2.53e-8`; every transformed BOND ranking still
matches its transformed exhaustive reference exactly.

The first 8 PCA components contain 90.91% of measured token energy; the first
64 contain 97.76%. This makes residual bounds useful earlier:

| held-out PCA arm | same-run exact median (s) | BOND median (s) | BOND/exact latency | pruned pairs | component ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefix seed=500 | 2.9804 | 37.1335 | 12.46x | 86.3% | 0.6902 |
| free exact-top-10 oracle seeds | 2.9804 | 36.6227 | 12.29x | 97.9% | 0.6153 |

For prefix seeds, 9 documents are pruned at dimension 32, 83,722 at dimension
64, and 95,177 at dimension 96. For oracle seeds, those counts are 40, 121,166,
and 81,782. PCA therefore produces genuine arithmetic pruning, including much
earlier exits, but the current kernel remains more than 12 times slower than
exhaustive SIMD-friendly MaxSim.

PCA fit took 0.536 seconds, transformation 0.297 seconds, and vertical-index
construction 1.687 seconds. All are offline and excluded from the query timing,
but reported because PCA changes index-build cost and storage preparation.

The combined result is more informative than either timing or work alone:

- raw dimensions fail because bounds become selective too late;
- oracle seeds show that prefix seed quality is not the main raw bottleneck;
- PCA shows that dimension order can remove 31-38% of component products;
- the implementation overhead still dominates, so inverse work ratio is not a
  valid proxy for speedup.

The result also does not compare upstream PDX against an optimized production
ColBERT engine. It answers the narrower project question for this independently
implemented exact-safe MaxSim kernel and workload.

## NFCorpus transfer

The same raw-order, prefix-500 configuration was repeated on 3,633 NFCorpus
documents and 40 held-out queries:

| dataset | exact median (s) | BOND median (s) | BOND/exact | pruned pairs | component ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 3.4043 | 42.0728 | 12.36x | 22.44% | 0.9547 |
| NFCorpus | 1.2954 | 17.3114 | 13.36x | 23.36% | 0.9416 |

NFCorpus prunes 730 document pairs at dimension 64 and 33,210 at dimension 96.
The transfer therefore preserves the mechanism-level conclusion: a nontrivial
document-pruning percentage translates into only a 5.84% component-product
reduction, and the exact-safe kernel remains much slower than regular exhaustive
MaxSim.

## Artifacts

- `results/controlled/scifact_bond_smoke.json`
- `results/controlled/scifact_bond_500d_5q_pilot.json`
- `results/controlled/scifact_bond_full_1q_seed_pilot.json`
- `results/controlled/scifact_bond_validation_10q_pilot.json`
- `results/controlled/scifact_bond_test_40q_pilot.json`
- `results/controlled/scifact_bond_oracle_test_40q_pilot.json`
- `results/controlled/scifact_bond_pca_500d_5q_smoke.json`
- `results/controlled/scifact_bond_pca_validation_10q_pilot.json`
- `results/controlled/scifact_bond_pca_test_40q_pilot.json`
- `results/controlled/nfcorpus_bond_test_40q_pilot.json`

![Controlled BOND result](figures/fig7_controlled_bond.png)

![Controlled transfer check](figures/fig8_controlled_transfer.png)
