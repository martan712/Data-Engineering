# Accelerating ColBERT Multi-Vector Search with PDX/BOND

Draft project report. The core compiled comparisons are complete as pilots;
timings remain pending a frozen clean-worktree release run.

## Abstract

ColBERT retrieval represents each document with many token vectors and scores a
query using late interaction: every query token selects its maximum similarity
to any token in a document, and these maxima are summed. This project studies
whether the vertically decomposed PDX layout and BOND branch-and-bound search can
reduce the work of this multi-vector MaxSim computation.

The project first validated PyLate ColBERT embeddings and the PDX-BOND
implementation independently. It then flattened document token vectors into PDX
and mapped token hits back to documents. Token-only aggregation was not a
reliable document scorer, but token retrieval followed by exact MaxSim reranking
produced strong candidate sets. Mechanism instrumentation suggested that
exact-safe BOND dimension pruning is weak on the tested 128-dimensional ColBERT
embeddings, even with favorable thresholds.

An audit found that the first wall-clock comparisons used an inefficient NumPy
exact baseline, unmatched rerank budgets, incomplete timing boundaries, and
cross-machine values. Those speedup claims were withdrawn. A new independently
implemented C++ exact MaxSim kernel and a controlled same-process runner now
measure complete FAISS-IVF and PDX-IVF pipelines. On a held-out SciFact pilot,
both engines have identical ranking-recovery curves. Reranking 200 documents per
query recovers 0.975 of the exact top-10 while evaluating 3.86% of all
query-document pairs; reranking the complete candidate pool recovers the exact
top-10 with about 12% of the comparisons. PDX search is slower than FAISS at low
rerank budgets on the tested system.

The independently implemented exact-safe BOND-MaxSim kernel also reproduces the
exact top-10 on all 40 held-out queries. It prunes 22.4% of query-document pairs,
but all pruning occurs after 96 of 128 dimensions and it evaluates 95.47% of
exhaustive component products. Its median latency is 42.07 seconds versus 3.40
seconds for compiled exhaustive MaxSim in the same interleaved runner. The
result supports a negative answer for raw ColBERT dimensions: the bounds become
useful too late to offset their runtime overhead.

A free exact-top-10 oracle seed policy reduces raw work only to 92.43%. A global
PCA rotation concentrates 90.91% of token energy in the first eight components
and reduces work to 69.02% with deterministic seeds or 61.53% with oracle seeds.
Despite that real arithmetic reduction, both PCA BOND arms remain more than 12
times slower than their same-run exhaustive reference. Dimension ordering helps
the bound, but current bound-maintenance and partial-state costs dominate.

The same raw-order configuration was transferred to NFCorpus. It preserves the
exact top-10 set for all 40 held-out queries but evaluates 94.16% of exhaustive
component products and has 13.36 times the same-run exhaustive median. The IVF
transfer is less forgiving than SciFact: full-pool reranking reaches 0.975 exact
recall@10, demonstrating a candidate-retrieval coverage limit. FAISS and PDX
again have the same recovery curve, with PDX slightly slower in every matched
NFCorpus arm.

## 1. Research objective

The primary question is:

> Can BOND-style dimension pruning accelerate exact ColBERT MaxSim search over
> a strong compiled exact implementation?

Secondary questions are:

1. Why do the BOND bounds become useful or remain loose on ColBERT token
   vectors?
2. Can a flat token-vector index generate document candidates with high exact
   MaxSim recall?
3. At matched rerank budgets, how do PDX-IVF and FAISS-IVF trade ranking
   recovery for online latency?

The project does not claim that IVF is novel. IVF is a practical comparison and
fallback architecture discovered while testing the BOND hypothesis.

## Related work

ColBERT introduced contextualized late interaction, in which independently
encoded query and document token vectors interact through MaxSim at retrieval
time [2]. ColBERTv2 reduced the representation footprint [3], while PLAID added
centroid-based pruning and a highly optimized retrieval engine [4]. These
systems establish that multi-vector retrieval needs methods designed around the
late-interaction operator rather than a single-vector distance alone.

BOND scans vertically decomposed dimensions incrementally and prunes objects
whose best possible final distance cannot enter the top-k [1]. PDX revisits the
dimension-oriented layout inside blocks and combines it with dimension-pruning
methods, including PDX-BOND [5]. ADSampling is related approximate work that
adaptively avoids full distance computations [6]. The present project asks a
different question: whether an exact-safe residual bound remains selective when
the indexed object is a variable-length token matrix and the document score is
a sum of token-wise maxima.

SciFact and NFCorpus are drawn from BEIR, a heterogeneous zero-shot retrieval
benchmark [7]. PyLate supplies the ColBERT-compatible model and embedding API
used in the experiments [8].

## 2. ColBERT representation and scoring

For query token vectors `q_i` and document token vectors `d_j`, normalized
ColBERT MaxSim is:

```text
score(q, d) = sum_i max_j dot(q_i, d_j)
```

PyLate with `lightonai/GTE-ModernColBERT-v1` produces variable-length token
matrices with dimension 128. The project stores them in a packed format:

- one dense `values` matrix containing all token vectors;
- one terminal `offsets` array of length `items + 1`;
- document and query ID arrays.

The PDX Python API exposes squared L2 distance. For unit-normalized vectors:

```text
squared_l2(x, y) = 2 - 2 * dot(x, y)
```

Nearest-neighbor ordering by squared L2 is therefore equivalent to ordering by
cosine similarity or normalized inner product. Retrieved distances are converted
back to similarity with `1 - squared_l2 / 2`.

## 3. System architecture

### 3.1 Exact reference

NumPy remains the readable score oracle. The performance reference is now an
independently implemented pybind11 C++17 kernel over packed float32 inputs. It:

- validates dtype, shape, contiguity, and terminal offsets;
- releases the Python GIL;
- parallelizes query-document pairs with OpenMP;
- vectorizes 128-dimensional dot products;
- returns the full exact query-document score matrix.

Its scores and deterministic rankings are tested against NumPy.

### 3.2 Flat token candidate generation

All document token vectors are indexed independently. Two metadata arrays map a
token-vector ID to its document and token position. For each query token, the
index retrieves its top-L document-token vectors. Hits are aggregated as follows:

1. map every hit back to a document;
2. retain the best retrieved similarity per query-token/document pair;
3. replace missing contributions with zero for the approximate selector score;
4. select at most C documents;
5. compute exact MaxSim for the selected documents;
6. return the final document top-k.

The zero-filled approximate score is a selector heuristic, not a replacement for
MaxSim. The complete unique document pool is also measured as an oracle upper
bound for selection at fixed L.

### 3.3 Candidate engines

The controlled pilot uses:

- FAISS `IndexIVFFlat` with normalized squared L2;
- PDX `IndexPDXBONDIVFFlat` at pinned commit
  `fdc62f2d22b3793060abf633cb5407438c7f739b`;
- the same exact C++ kernel for reranking both candidate sets.

Index construction is an offline cost. Online timing starts with normalized query
token vectors resident in memory and ends with ranked document IDs.

## 4. Experimental progression

The project advanced through increasingly realistic tests:

1. PyLate model loading, toy indexing, and query retrieval.
2. PDX-BOND build and exact L2 smoke test in WSL2.
3. Packed variable-length ColBERT embedding export.
4. NumPy exact MaxSim and flat token-vector PDX retrieval on a toy corpus.
5. Candidate generation and exact reranking on 174-document and 600-document
   local corpora.
6. Candidate-pool versus selector-loss studies.
7. SciFact and NFCorpus exploratory experiments.
8. Batch-call and shared-scan prototypes.
9. PDX-IVF, FAISS-IVF, and PLAID exploratory comparisons.
10. MaxSim-aware BOND work-ratio instrumentation and PCA ordering.
11. Measurement audit and controlled same-stack rebuild.

The archive preserves the early scripts because the failed approaches explain
why the final architecture and research question changed.

## 5. Measurement audit

The first figures cannot support final speedup claims for four central reasons:

1. PDX and FAISS timers omitted token-to-document aggregation and selection.
2. PLAID full-scored up to 8,192 candidates while IVF commonly reranked 50.
3. Exact search used a per-document Python/NumPy loop.
4. Windows and WSL times appeared in the same comparison.

Additional issues included uncontrolled thread counts, no repeated interleaved
runs, parameter selection on evaluation queries, favorable corpus prefixes, and
changing IVF bucket fractions in scaling plots. The historical 30x/39x/32x and
related ratios were therefore withdrawn. Compact source JSON files remain under
`results/legacy/` for quality and mechanism analysis only.

The replacement protocol requires one machine and process, matched work, one
outer online timer, explicit stage timers, pinned threads and affinity, warm-up,
interleaved repetitions, raw samples, input hashes, and Git/environment metadata.

## 6. BOND mechanism evidence

The NumPy MaxSim-aware instrumentation incrementally scans dimensions and applies
a Cauchy-Schwarz residual bound. Under the recorded SciFact workload:

| setting | inverse measured work ratio |
| --- | ---: |
| raw embeddings, oracle threshold | 1.10 |
| PCA rotation, implementable threshold | 1.22 |
| PCA rotation, oracle threshold | 1.68 |

These are not speedups. They omit bound maintenance, threshold updates,
branching, and memory-layout cost. The narrow conclusion is that raw normalized
ColBERT dimensions did not expose enough early signal for extensive exact-safe
pruning under the tested checkpoints. PCA helps the operation count but does not
yet establish a useful implementation.

### 6.1 Compiled exact-safe kernel

The project-local C++17 BOND-MaxSim index stores each document in
dimension-major order and precomputes residual norms. It derives the pruning
threshold only from conservative lower bounds of fully scored seed documents.
Partial-dot rounding error is bounded, residual terms are rounded upward, and a
document is pruned only when its upper bound is strictly below the threshold.

With `seed_count=500`, the first 10 SciFact queries were used as validation and
the remaining 40 were held out. Both arms used four pinned CPUs, one warm-up,
five measured interleaved repetitions, and one complete online timer.

| split/arm | median (s) | p95 (s) | exact top-10 | pruned pairs | component ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation exact | 0.6199 | 0.7644 | 10/10 | - | 1.0000 |
| validation BOND | 11.0535 | 11.1254 | 10/10 | 9.7% | 0.9759 |
| held-out exact | 3.4043 | 3.6615 | 40/40 | - | 1.0000 |
| held-out BOND | 42.0728 | 44.5968 | 40/40 | 22.4% | 0.9547 |

Every held-out prune occurred at dimension 96. BOND therefore avoided only
4.53% of component products while paying for residual norms, upper bounds,
threshold maintenance, branches, and partial-state writes. Its held-out median
latency was 12.36 times the exhaustive latency. This is a controlled slowdown,
not a speedup claim.

The exact kernel accumulates in float while BOND accumulates scores in double;
their held-out top-k IDs were identical and the maximum reported score
difference was `7.34e-6`.

### 6.2 Oracle seeds and PCA

Supplying the exact top-10 as free seed IDs removes seed-selection quality as a
confounder. On raw held-out embeddings, this prunes 36.3% of query-document
pairs but still evaluates 92.43% of component products because all pruning is at
dimension 96. Oracle BOND takes 47.2189 seconds versus 2.6676 seconds for its
same-run exact arm.

Uncentered PCA is fitted offline on document token vectors and applied to both
documents and queries. The transform preserves inner products in exact
arithmetic and has a float32 orthogonality error of `2.53e-8` here.

| held-out arm | BOND median (s) | same-run exact (s) | latency ratio | pruned pairs | component ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw, prefix seed=500 | 42.0728 | 3.4043 | 12.36x | 22.4% | 0.9547 |
| raw, free oracle seeds | 47.2189 | 2.6676 | 17.70x | 36.3% | 0.9243 |
| PCA, prefix seed=500 | 37.1335 | 2.9804 | 12.46x | 86.3% | 0.6902 |
| PCA, free oracle seeds | 36.6227 | 2.9804 | 12.29x | 97.9% | 0.6153 |

Absolute timings from separate runs are not compared directly; every latency
ratio uses the exhaustive arm interleaved in the same run. PCA fit,
transformation, and vertical index construction are offline and separately
recorded as 0.536, 0.297, and 1.687 seconds.

## 7. Controlled SciFact pilot

### 7.1 Setup

- 5,183 documents and 1,193,945 document token vectors.
- First 10 queries used for selector validation.
- Remaining 40 queries used for the held-out decision pilot.
- `L=100`, `nprobe=8`, four pinned CPUs and four OpenMP/BLAS threads.
- One warm-up and five measured interleaved repetitions.
- Fixed `approx_score` selector after validation.

`union_approx_and_count` improved exact recall@10 by 0.01 at C=50, reduced it by
0.02 at C=100, and tied at C=200/400. The simpler selector was retained.

### 7.2 Results

| arm | median (s) | p95 (s) | exact recall@10 | mean C | comparison ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled exact | 2.7766 | 3.1541 | 1.0000 | 5,183 | 1.0000 |
| FAISS-IVF C=50 | 0.3018 | 0.3822 | 0.8600 | 50.0 | 0.0096 |
| PDX-IVF C=50 | 0.3425 | 0.4112 | 0.8600 | 50.0 | 0.0096 |
| FAISS-IVF C=100 | 0.3455 | 0.4309 | 0.9325 | 100.0 | 0.0193 |
| PDX-IVF C=100 | 0.4307 | 0.5770 | 0.9325 | 100.0 | 0.0193 |
| FAISS-IVF C=200 | 0.4968 | 0.5608 | 0.9750 | 200.0 | 0.0386 |
| PDX-IVF C=200 | 0.5664 | 0.6215 | 0.9750 | 200.0 | 0.0386 |
| FAISS-IVF C=400 | 0.8158 | 0.9364 | 0.9900 | 390.9 | 0.0754 |
| PDX-IVF C=400 | 0.7961 | 0.9730 | 0.9900 | 390.9 | 0.0754 |
| FAISS-IVF full pool | 1.2129 | 1.3157 | 1.0000 | 619.3 | 0.1195 |
| PDX-IVF full pool | 1.1142 | 1.4212 | 1.0000 | 619.3 | 0.1195 |

Every candidate pool contained the complete exact top-10. Quality loss at finite
C is therefore selector loss. FAISS and PDX have identical recovery values and
pool sizes. At C=50, PDX token search took a median 0.225 s versus 0.195 s for
FAISS. At large C, reranking dominates and small total-time reversals are not
interpreted as an engine advantage.

Qrels Recall@10 and MRR@10 were identical for every arm despite exact-ranking
recall ranging from 0.86 to 1.00. This confirms that incomplete qrels cannot
diagnose the architecture loss on this small query set.

A one-thread follow-up measured 8.068 s for compiled exact, 1.824 s for FAISS
full-pool reranking, and 1.912 s for PDX full-pool reranking. Candidate pools,
selected documents, hit counts, and rankings were identical between engines.
The exact kernel obtained 2.91x throughput scaling from one to four threads; the
one-thread samples were more stable.

### 7.3 NFCorpus transfer check

The same WSL stack and online boundary were applied to 3,633 NFCorpus documents,
864,703 document token vectors, and 40 held-out queries. The candidate
configuration remained `L=100`, `nprobe=8`, with one warm-up and five measured
interleaved repetitions.

| arm | median (s) | p95 (s) | exact recall@10 | mean C | comparison ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled exact | 1.2874 | 1.4739 | 1.0000 | 3,633 | 1.0000 |
| FAISS-IVF C=50 | 0.2060 | 0.2404 | 0.8275 | 50.0 | 0.0138 |
| PDX-IVF C=50 | 0.2218 | 0.2582 | 0.8275 | 50.0 | 0.0138 |
| FAISS-IVF C=200 | 0.3330 | 0.3356 | 0.9650 | 198.0 | 0.0545 |
| PDX-IVF C=200 | 0.3615 | 0.3728 | 0.9650 | 198.0 | 0.0545 |
| FAISS-IVF full pool | 0.6564 | 0.6842 | 0.9750 | 388.1 | 0.1068 |
| PDX-IVF full pool | 0.6586 | 0.7142 | 0.9750 | 388.1 | 0.1068 |

Candidate pools and final rankings agree between FAISS and PDX. Unlike
SciFact, reranking every retrieved candidate does not recover the full exact
top-10: the remaining 0.025 loss is candidate-pool coverage, not selector loss.
PDX token search remains slower than FAISS at C=50 (0.135 s versus 0.117 s
median), and no PDX-specific advantage is observed.

Raw-order BOND with 500 deterministic prefix seeds takes 17.3114 seconds versus
1.2954 seconds for its interleaved exact arm. It prunes 23.36% of document pairs
but evaluates 94.16% of component products; most pruning occurs at dimension
96. Every top-10 set is exact. One query changes the order of two documents tied
by the float exact reference; the double-precision score gap is `5.1e-8`, and
the runner records this as a numerical tie rather than silently calling the
rankings identical.

## 8. Current conclusions

1. Flat token-vector indexing can generate high-coverage document pools for
   ColBERT MaxSim.
2. Token-hit aggregation is not a reliable final document scorer; exact MaxSim
   reranking is required.
3. Selector budget controls the quality-work trade-off once pool coverage is
   high.
4. PDX-IVF and FAISS-IVF show the same candidate quality in the controlled
   pilot. PDX does not currently provide a PDX-specific retrieval advantage.
5. The candidate architecture is practically useful, but it is standard IVF and
   is not the primary research novelty.
6. Exact-safe BOND dimension pruning is ineffective on the tested raw ColBERT
   dimensions: it preserves correctness but eliminates too little arithmetic,
   too late, and is slower than exhaustive MaxSim.
7. The raw-order BOND diagnosis transfers from SciFact to NFCorpus. Candidate
   retrieval quality does not transfer perfectly: NFCorpus exposes a pool
   coverage limit even when every retrieved document is reranked.

## 9. Limitations and threats to validity

- The controlled JSON was generated from a dirty worktree and must be repeated
  after the implementation is frozen.
- The exact kernel is compiled and tested, but it is an independent baseline,
  not a claim to be the fastest possible MaxSim implementation.
- SciFact and NFCorpus are small. Results may change at larger corpus sizes and
  on different query/document length distributions.
- Only one candidate configuration (`L=100`, `nprobe=8`) is used in the held-out
  pilot.
- IVF parameters originate from earlier exploration; the validation/test split
  prevents new selector tuning but cannot erase all historical knowledge.
- Qrels are incomplete and the query sample is small.
- PDX and FAISS index implementations may use different internal threading and
  layouts even under the same process-level thread limit.
- PLAID has not yet been rerun with a matched full-score budget.
- The BOND kernel is independently authored and exact-safe, but it is a
  project-local document-oriented prototype rather than an upstream PDX block
  kernel or a claim of production-optimal implementation.
- The implementable threshold uses the first 500 documents as deterministic
  seeds. Oracle seeds are intentionally free and PCA is offline, so both are
  mechanism diagnostics rather than end-to-end methods.

## 10. Reproducibility

The repository now contains:

- pinned Python dependency manifests;
- a pinned, clean PDX commit and setup log;
- the independent exact C++ source and build script;
- automated utility, manifest, candidate, and kernel tests;
- a controlled runner with raw interleaved samples and complete stage timing;
- tracked historical and controlled-pilot result JSON files with integrity
  manifests;
- figures generated directly from the controlled result;
- an explicit benchmark protocol and evidence audit.

Large embeddings remain outside Git. Their SHA-256 hashes are stored in each
controlled result artifact.

## 11. Remaining work

1. Freeze the worktree and repeat the held-out SciFact matrix.
2. Repeat the same frozen configurations on NFCorpus.
3. Regenerate final figures and replace pilot labels.
4. Add a matched-budget PLAID point if time permits.
5. Finalize proofreading and artifact provenance for submission.

## References

1. A. P. de Vries, N. Mamoulis, N. Nes, and M. L. Kersten. "Efficient k-NN
   Search on Vertically Decomposed Data." ACM SIGMOD, 2002.
   [doi:10.1145/564691.564729](https://doi.org/10.1145/564691.564729).
2. O. Khattab and M. Zaharia. "ColBERT: Efficient and Effective Passage Search
   via Contextualized Late Interaction over BERT." SIGIR, 2020.
   [arXiv:2004.12832](https://arxiv.org/abs/2004.12832).
3. K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia.
   "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
   Interaction." NAACL, 2022.
   [arXiv:2112.01488](https://arxiv.org/abs/2112.01488).
4. K. Santhanam, O. Khattab, C. Potts, and M. Zaharia. "PLAID: An Efficient
   Engine for Late Interaction Retrieval." CIKM, 2022.
   [arXiv:2205.09707](https://arxiv.org/abs/2205.09707).
5. L. Kuffo, E. Krippner, and P. Boncz. "PDX: A Data Layout for Vector
   Similarity Search." Proceedings of the ACM on Management of Data, 2025.
   [doi:10.1145/3725333](https://doi.org/10.1145/3725333).
6. J. Gao and C. Long. "High-Dimensional Approximate Nearest Neighbor Search:
   with Reliable and Efficient Distance Comparison Operations." 2023.
   [arXiv:2303.09855](https://arxiv.org/abs/2303.09855).
7. N. Thakur, N. Reimers, A. Rueckle, A. Srivastava, and I. Gurevych. "BEIR: A
   Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
   Models." NeurIPS Datasets and Benchmarks, 2021.
   [arXiv:2104.08663](https://arxiv.org/abs/2104.08663).
8. LightOn. "PyLate: Flexible Training and Retrieval for Late Interaction
   Models." Software repository, version 1.5.0 used here.
   [github.com/lightonai/pylate](https://github.com/lightonai/pylate).
