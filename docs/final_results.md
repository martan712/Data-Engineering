# Final Controlled Results

## Status and scope

The release was generated from clean commit `0a11fda` on Ubuntu WSL2 with
Python 3.11.9. Every artifact records `dirty=false`, initial CPU affinity
`[0, 2, 4, 6]`, four OpenMP/BLAS/Rayon threads, one warm-up, five measured
interleaved runs, input hashes, package versions, stage times, and raw samples.

The online boundary starts with resident normalized query token vectors and ends
with ranked document IDs. Encoding and index construction are excluded and
reported separately. No result below is an end-to-end retrieval-system speedup.

## IVF candidate generation

Both engines use `L=100`, `nprobe=8`, the same IVF training sample, the
`approx_score` selector, and compiled float32 exact MaxSim reranking.

### SciFact

5,183 documents, 1,193,945 document token vectors, and 40 held-out queries:

| arm | median (s) | p95 (s) | exact R@10 | mean reranked | rerank ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled exact | 2.5774 | 2.9767 | 1.0000 | 5,183 | 1.0000 |
| FAISS-IVF C=50 | 0.3550 | 0.3683 | 0.8600 | 50.0 | 0.0096 |
| PDX-IVF C=50 | 0.3735 | 0.3880 | 0.8600 | 50.0 | 0.0096 |
| FAISS-IVF C=100 | 0.3913 | 0.4183 | 0.9325 | 100.0 | 0.0193 |
| PDX-IVF C=100 | 0.4257 | 0.4324 | 0.9325 | 100.0 | 0.0193 |
| FAISS-IVF C=200 | 0.4770 | 0.5018 | 0.9750 | 200.0 | 0.0386 |
| PDX-IVF C=200 | 0.5060 | 0.5149 | 0.9750 | 200.0 | 0.0386 |
| FAISS-IVF C=400 | 0.6949 | 0.7365 | 0.9900 | 390.9 | 0.0754 |
| PDX-IVF C=400 | 0.7262 | 0.8317 | 0.9900 | 390.9 | 0.0754 |
| FAISS-IVF full pool | 0.9773 | 1.0870 | 1.0000 | 619.3 | 0.1195 |
| PDX-IVF full pool | 1.0322 | 1.2016 | 1.0000 | 619.3 | 0.1195 |

The candidate pool contains every exact top-10 document for all 40 queries, so
finite-C loss is selector loss. FAISS and PDX have identical candidate pools,
retrieved-hit counts, final rankings, and recovery curves. Their latency winner
changes across datasets/configurations; there is no consistent PDX advantage.

### NFCorpus

3,633 documents, 864,703 document token vectors, and 40 held-out queries:

| arm | median (s) | p95 (s) | exact R@10 | mean reranked | rerank ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled exact | 1.2554 | 1.4940 | 1.0000 | 3,633 | 1.0000 |
| FAISS-IVF C=50 | 0.2268 | 0.2585 | 0.8275 | 50.0 | 0.0138 |
| PDX-IVF C=50 | 0.2334 | 0.2648 | 0.8275 | 50.0 | 0.0138 |
| FAISS-IVF C=200 | 0.3460 | 0.3539 | 0.9650 | 198.0 | 0.0545 |
| PDX-IVF C=200 | 0.3598 | 0.4300 | 0.9650 | 198.0 | 0.0545 |
| FAISS-IVF full pool | 0.5698 | 0.5771 | 0.9750 | 388.1 | 0.1068 |
| PDX-IVF full pool | 0.5358 | 0.6432 | 0.9750 | 388.1 | 0.1068 |

The full pool contains every exact top-10 document for 37/40 queries, exposing
candidate-retrieval loss that no selector can repair. FAISS and PDX again have
identical final rankings and quality. Exact qrels Recall@10 is 0.9850 on
SciFact and 0.2191 on NFCorpus; exact-ranking recovery is the more sensitive
architecture metric.

Offline FAISS/PDX builds take 18.15/21.43 seconds on SciFact and 12.37/14.25
seconds on NFCorpus.

## Controlled PLAID comparison

PyLate 1.5.0 with CPU fast-plaid 1.3.0.290 uses `nbits=4` and Triton disabled.
The first 10 SciFact queries select `(nprobe=8, full_scores=100)` as the fastest
arm above validation exact R@10 0.85 and `(8,400)` as the maximum-recall arm.
Validation recovery is 0.89/0.92; it remains 0.92 through full-corpus scoring.
Those choices were frozen before queries 10--49 were evaluated on either
dataset. The validation/test shift is large:

| dataset/arm | median (s) | p95 (s) | exact R@10 | qrels R@10 | configured full-score ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact PLAID F=100 | 14.5497 | 14.7523 | 0.2875 | 0.3250 | 0.0193 |
| SciFact PLAID F=400 | 16.6456 | 16.8475 | 0.4375 | 0.3750 | 0.0772 |
| NFCorpus PLAID F=100 | 5.6825 | 6.0592 | 0.3100 | 0.1434 | 0.0275 |
| NFCorpus PLAID F=400 | 7.2505 | 7.3043 | 0.3950 | 0.1043 | 0.1101 |

`n_full_scores` is a configured budget, not an observed candidate count. The
current API does not expose realized candidates, so the final column is only an
upper-bound budget fraction and is not directly equated with IVF work.

A post-hoc full-score sensitivity analysis asks whether the selected budgets
alone explain the low held-out quality. It is reported separately because it
was run after observing the split shift:

| dataset | PLAID full budget | exact median (s) | PLAID median (s) | PLAID/exact | exact R@10 | qrels R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 5,183 | 3.2697 | 80.6144 | 24.65x | 0.6750 | 0.7850 |
| NFCorpus | 3,633 | 1.2836 | 46.0245 | 35.86x | 0.5975 | 0.1302 |

Full scoring improves exact recovery but still does not reproduce exact MaxSim,
consistent with loss from earlier PLAID stages and compressed scoring. On these
small CPU workloads every selected and full-score PLAID arm is slower and less
accurate than its compiled exact or IVF counterpart. This is evidence about
this pinned implementation/configuration, not a general claim that PLAID is
ineffective at its intended larger scale.

Offline PLAID construction takes 1,415.73 seconds on SciFact and 527.82 seconds
on NFCorpus; the resulting indices are 182.3 MB and 131.0 MB. These costs are
not included in online latency.

## Exact-safe BOND-MaxSim

The BOND and exhaustive arms both use float64 products and accumulation. The
prefix arm seeds its threshold with 500 deterministic documents. Oracle seeds
receive the exact top-10 for free and are diagnostic only.

| dataset/arm | exact median (s) | BOND median (s) | BOND/exact | pruned pairs | products evaluated |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact raw, prefix-500 | 3.9208 | 42.1947 | 10.76x | 22.44% | 95.47% |
| SciFact raw, free oracle | 3.9208 | 45.0854 | 11.50x | 36.32% | 92.43% |
| SciFact PCA, prefix-500 | 4.4111 | 36.9539 | 8.38x | 86.30% | 69.02% |
| SciFact PCA, free oracle | 4.4111 | 37.5548 | 8.51x | 97.91% | 61.53% |
| NFCorpus raw, prefix-500 | 1.9501 | 19.1847 | 9.84x | 23.36% | 94.16% |

Every BOND arm returns the exact top-10. The largest score error is `1.78e-14`.
Raw SciFact pruning occurs only at dimension 96; NFCorpus has 730 prunes at 64
and 33,210 at 96. PCA concentrates 90.91% of token energy in eight components
and removes 31--38% of products, but BOND remains 8.38--8.51 times slower.

## Conclusion

1. Flat token retrieval is useful for candidate generation, not final MaxSim.
2. IVF plus exact reranking gives a strong quality/work curve, but it is
   standard IVF and not a PDX-specific contribution.
3. FAISS-IVF and PDX-IVF have identical quality and no consistent latency
   winner here.
4. The controlled CPU PLAID points are dominated at this small scale; even the
   post-hoc full-score upper bound remains approximate and much slower.
5. Exact-safe raw BOND retains 94--95% of arithmetic and is about 10 times
   slower than precision-matched exhaustive MaxSim.
6. PCA improves pruning substantially but does not overcome kernel overhead.

Raw artifacts and integrity metadata are in `results/final/manifest.json`.
Figures 5--9 are generated directly from these release artifacts.
