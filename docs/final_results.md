# Final Controlled Results

## Status and scope

The release measurements were generated from clean commit `0cc6145` on Ubuntu
WSL2. Every artifact records `dirty=false`, initial CPU affinity
`[0, 2, 4, 6]` (one hardware thread on each of four physical cores), four
OpenMP/BLAS threads, one warm-up, five measured interleaved runs, input hashes,
package versions, stage times, and raw outer-timer samples.

The online boundary starts with resident normalized query token vectors and ends
with ranked document IDs. Encoding and index construction are excluded and
reported separately. Latency comparisons below therefore apply to this online
boundary, not to a complete retrieval service.

## IVF candidate generation

Both engines use `L=100`, `nprobe=8`, the same IVF training sample, the
`approx_score` selector, and compiled float32 exact MaxSim reranking.

### SciFact

5,183 documents, 1,193,945 document token vectors, and 40 held-out queries:

| arm | median (s) | p95 (s) | exact R@10 | mean reranked docs | rerank ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled exact | 2.9333 | 3.1010 | 1.0000 | 5,183 | 1.0000 |
| FAISS-IVF C=50 | 0.3810 | 0.4003 | 0.8600 | 50.0 | 0.0096 |
| PDX-IVF C=50 | 0.3495 | 0.3575 | 0.8600 | 50.0 | 0.0096 |
| FAISS-IVF C=100 | 0.3814 | 0.4025 | 0.9325 | 100.0 | 0.0193 |
| PDX-IVF C=100 | 0.3973 | 0.4757 | 0.9325 | 100.0 | 0.0193 |
| FAISS-IVF C=200 | 0.5051 | 0.5885 | 0.9750 | 200.0 | 0.0386 |
| PDX-IVF C=200 | 0.5411 | 0.5541 | 0.9750 | 200.0 | 0.0386 |
| FAISS-IVF C=400 | 0.8483 | 0.9233 | 0.9900 | 390.9 | 0.0754 |
| PDX-IVF C=400 | 0.8820 | 0.9127 | 0.9900 | 390.9 | 0.0754 |
| FAISS-IVF full pool | 1.2908 | 1.7934 | 1.0000 | 619.3 | 0.1195 |
| PDX-IVF full pool | 1.2467 | 1.5615 | 1.0000 | 619.3 | 0.1195 |

The candidate pool contains every exact top-10 document for all 40 queries, so
finite-C loss is selector loss. Full-pool reranking reproduces exact rankings
while evaluating 11.95% of query-document pairs. Its online latency is lower
than exhaustive MaxSim in this workload, but encoding, build, and service costs
are outside the boundary.

### NFCorpus

3,633 documents, 864,703 document token vectors, and 40 held-out queries:

| arm | median (s) | p95 (s) | exact R@10 | mean reranked docs | rerank ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled exact | 1.4328 | 1.6469 | 1.0000 | 3,633 | 1.0000 |
| FAISS-IVF C=50 | 0.2114 | 0.2541 | 0.8275 | 50.0 | 0.0138 |
| PDX-IVF C=50 | 0.2365 | 0.2712 | 0.8275 | 50.0 | 0.0138 |
| FAISS-IVF C=200 | 0.3615 | 0.4032 | 0.9650 | 198.0 | 0.0545 |
| PDX-IVF C=200 | 0.3610 | 0.4265 | 0.9650 | 198.0 | 0.0545 |
| FAISS-IVF full pool | 0.7019 | 0.7163 | 0.9750 | 388.1 | 0.1068 |
| PDX-IVF full pool | 0.6952 | 0.7510 | 0.9750 | 388.1 | 0.1068 |

The pool itself has mean exact recall@10 of 0.975 and contains all exact top-10
documents for 37/40 queries. NFCorpus therefore exposes candidate-retrieval loss
that no top-C selector can repair. FAISS and PDX produce identical candidate
pools and final rankings. Their latency ordering changes across configurations,
so these results establish no consistent PDX-specific advantage.

Exact qrels Recall@10 is 0.9850 on SciFact and 0.2191 on NFCorpus. On NFCorpus,
C=50, C=200, and full-pool qrels Recall@10 are 0.2150, 0.2172, and 0.2191.
Exact-ranking recovery remains the more sensitive architecture metric.

Offline FAISS/PDX build times are 18.52/21.46 seconds for SciFact and
12.27/14.67 seconds for NFCorpus.

## Exact-safe BOND-MaxSim

The BOND and exhaustive arms both use float64 products and accumulation over the
same float32 embeddings. BOND uses 500 deterministic prefix documents as its
implementable threshold seed. The oracle arm receives the exact top-10 for free
and is only a mechanism upper bound.

| dataset/arm | exact median (s) | BOND median (s) | BOND/exact | pruned pairs | products evaluated |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact raw, prefix-500 | 4.2740 | 46.4861 | 10.88x | 22.44% | 95.47% |
| SciFact raw, free oracle | 4.2740 | 48.1283 | 11.26x | 36.32% | 92.43% |
| SciFact PCA, prefix-500 | 4.0591 | 35.5169 | 8.75x | 86.30% | 69.02% |
| SciFact PCA, free oracle | 4.0591 | 35.4903 | 8.74x | 97.91% | 61.53% |
| NFCorpus raw, prefix-500 | 2.0397 | 20.3855 | 9.99x | 23.36% | 94.16% |

Every BOND arm returns exactly the same ordered top-10 as its exhaustive arm.
The largest score error is `1.78e-14`. Raw SciFact pruning occurs only at
dimension 96; NFCorpus has 730 prunes at dimension 64 and 33,210 at dimension
96. A visible document-pruning percentage therefore saves little arithmetic in
the natural model order.

Offline uncentered PCA concentrates 90.91% of token energy in the first eight
components. It creates genuine early pruning and removes 31-38% of component
products, but the BOND kernel remains about 8.7 times slower. Dimension ordering
helps the bound; bound maintenance, partial state, branching, and irregular
access still dominate the regular exhaustive kernel.

## Conclusion

1. Flat token retrieval is useful as candidate generation, not as a replacement
   for document-level MaxSim.
2. Candidate generation plus exact reranking gives a controllable quality/work
   curve, but this is standard IVF and not a PDX-specific contribution.
3. FAISS-IVF and PDX-IVF have identical quality here and no consistent latency
   winner.
4. Exact-safe raw-order BOND is correct but retains 94-95% of arithmetic and is
   roughly 10 times slower than precision-matched exhaustive MaxSim on both
   datasets.
5. PCA proves that a favorable dimension order improves pruning, but not enough
   to make this BOND kernel competitive.

Raw artifacts and integrity metadata are in `results/final/manifest.json`.
Figures 5-8 are generated directly from those artifacts.
