# Historical Controlled Candidate-Generation Pilots

> This document preserves the dirty-worktree decision pilots. It is not the
> source for release numbers. See `docs/final_results.md` and `results/final/`
> for clean-commit results and figures.

## Status

These are decision-making pilot results, not final submission claims. They were
produced from a dirty `Mikel` worktree while the controlled runner was being
implemented. They have since been replaced by the clean release.

The preserved artifact is:

`results/controlled/scifact_test_40q_controlled_pilot.json`

The transfer artifact is:

`results/controlled/nfcorpus_ivf_test_40q_pilot.json`

Figures 5 and 6 originally used the SciFact artifact below. They have since been
regenerated from `results/final/scifact_ivf_test_40q_final.json`:

- `docs/figures/fig5_controlled_quality_latency.png`
- `docs/figures/fig6_controlled_stage_breakdown.png`

## Setup

- Dataset: full prepared SciFact corpus, 5,183 documents.
- Evaluation split: queries 10-49, 40 queries not used in the selector-policy
  validation run.
- Embeddings: 1,193,945 normalized document token vectors, dimension 128.
- Candidate configuration: `L=100`, `nprobe=8`, `approx_score` selector.
- PDX: clean commit `fdc62f2d22b3793060abf633cb5407438c7f739b`.
- Runtime: Ubuntu WSL2, four pinned CPUs, four OpenMP/BLAS threads.
- Schedule: one warm-up and five measured interleaved repetitions.
- Online boundary: resident query token vectors to ranked document IDs.
- Offline build: FAISS-IVF 11.10 s; PDX-IVF 14.20 s.

Online time includes token search, token-to-document aggregation, selection,
compiled exact MaxSim reranking, and final top-k. Index construction and query
encoding are reported separately.

## Held-out results

| arm | median (s) | p95 (s) | exact recall@10 | mean reranked docs | rerank ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled exact | 2.7766 | 3.1541 | 1.0000 | 5,183 | 1.0000 |
| FAISS-IVF, C=50 | 0.3018 | 0.3822 | 0.8600 | 50.0 | 0.0096 |
| PDX-IVF, C=50 | 0.3425 | 0.4112 | 0.8600 | 50.0 | 0.0096 |
| FAISS-IVF, C=100 | 0.3455 | 0.4309 | 0.9325 | 100.0 | 0.0193 |
| PDX-IVF, C=100 | 0.4307 | 0.5770 | 0.9325 | 100.0 | 0.0193 |
| FAISS-IVF, C=200 | 0.4968 | 0.5608 | 0.9750 | 200.0 | 0.0386 |
| PDX-IVF, C=200 | 0.5664 | 0.6215 | 0.9750 | 200.0 | 0.0386 |
| FAISS-IVF, C=400 | 0.8158 | 0.9364 | 0.9900 | 390.9 | 0.0754 |
| PDX-IVF, C=400 | 0.7961 | 0.9730 | 0.9900 | 390.9 | 0.0754 |
| FAISS-IVF, full pool | 1.2129 | 1.3157 | 1.0000 | 619.3 | 0.1195 |
| PDX-IVF, full pool | 1.1142 | 1.4212 | 1.0000 | 619.3 | 0.1195 |

The candidate pool contained every exact top-10 document for all 40 queries.
The loss below full-pool reranking therefore comes from top-C selection, not
candidate retrieval coverage.

## Interpretation

1. **The optimized exact baseline changes the scale of the comparison.** Its
   median is 2.78 s for 40 queries. Historical ratios against the per-document
   NumPy baseline substantially overstated the candidate pipeline advantage.
2. **FAISS and PDX have the same quality curve.** At every matched `C`, both
   engines produced the same exact-recall value and candidate-pool size. A new
   runner field will verify set/ranking equality directly in the clean rerun.
3. **PDX search is slower in the low-C region.** At `C=50`, median search-stage
   time was 0.225 s for PDX and 0.195 s for FAISS. At larger `C`, exact reranking
   dominates total time and the small end-to-end differences are too noisy to
   interpret as an engine advantage.
4. **Candidate generation remains useful.** `C=200` reranks 3.86% of the full
   query-document pairs and recovers 0.975 of the exact top-10. Full-pool
   reranking evaluates about 12% and recovers the exact top-10 completely.
5. **Qrels do not resolve the quality differences.** Exact and every approximate
   arm reported qrels Recall@10 = 0.985 and MRR@10 = 0.902, while exact-ranking
   recovery ranged from 0.86 to 1.00. Incomplete qrels are therefore secondary;
   exact-ranking recovery is the primary architecture metric.

The observed same-stack latency ratios are reported as pilot measurements only.
No final end-to-end speedup is claimed while the worktree is dirty and the
independent exact kernel has not yet been frozen as the final baseline.

## Single-thread follow-up

A separate one-thread run on the same 40 queries recorded:

| arm | median (s) | exact recall@10 |
| --- | ---: | ---: |
| Compiled exact | 8.0684 | 1.0000 |
| FAISS-IVF, full pool | 1.8238 | 1.0000 |
| PDX-IVF, full pool | 1.9120 | 1.0000 |

The exact kernel scales by 2.91x from one to four threads on this workload. Its
one-thread samples ranged from 8.02 to 8.16 s, substantially more stable than
the four-thread samples. Within the one-thread run, FAISS and PDX had identical
candidate pools, selected candidates, retrieved hit counts, and final rankings.
PDX token search remained slower (0.253 s versus 0.195 s median).

## Selector validation

On the first 10 queries, `union_approx_and_count` changed exact recall@10 versus
`approx_score` by +0.01 at `C=50`, -0.02 at `C=100`, and 0.00 at `C=200/400`.
The simpler `approx_score` policy was therefore frozen for the held-out run.

## BOND decision and next steps

The independently authored exact-safe BOND-MaxSim kernel is now implemented and
measured through its own interleaved runner. On 40 held-out queries it preserves
the exact top-10 but evaluates 95.47% of raw component products and takes 42.07
seconds versus 3.40 seconds for same-run exhaustive MaxSim. Free oracle seeds
and PCA improve pruning but do not make the kernel competitive. Full details are
in `docs/controlled_bond_results.md`.

## NFCorpus transfer

The frozen candidate configuration was repeated on 3,633 documents, 864,703
document token vectors, and 40 held-out NFCorpus queries:

| arm | median (s) | exact recall@10 | mean reranked docs | rerank ratio |
| --- | ---: | ---: | ---: | ---: |
| Compiled exact | 1.2874 | 1.0000 | 3,633 | 1.0000 |
| FAISS-IVF, C=50 | 0.2060 | 0.8275 | 50.0 | 0.0138 |
| PDX-IVF, C=50 | 0.2218 | 0.8275 | 50.0 | 0.0138 |
| FAISS-IVF, C=200 | 0.3330 | 0.9650 | 198.0 | 0.0545 |
| PDX-IVF, C=200 | 0.3615 | 0.9650 | 198.0 | 0.0545 |
| FAISS-IVF, full pool | 0.6564 | 0.9750 | 388.1 | 0.1068 |
| PDX-IVF, full pool | 0.6586 | 0.9750 | 388.1 | 0.1068 |

PDX and FAISS produce the same candidate pools and final rankings. The full
pool misses 2.5% of the exact top-10 set on average, so NFCorpus failures include
candidate-retrieval coverage loss rather than only top-C selector loss. PDX is
slightly slower in all three matched arms; this pilot provides no PDX-specific
quality or latency advantage.

The remaining release steps are:

1. Repeat this exact matrix from a clean Git commit and retain the automatic
   FAISS/PDX candidate-equality checks.
2. Repeat the frozen BOND configuration after the implementation is committed.
3. Repeat the frozen exact/candidate/BOND configurations on NFCorpus.
4. Regenerate submission figures from only the frozen result artifacts.
