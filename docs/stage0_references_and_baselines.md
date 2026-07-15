# Stage 0 References And Baselines

Prepared 2026-06-29 for branch `final-research-implementation`.

Stage 0 establishes the vocabulary, source snapshots, and baseline separation
used by the final research implementation. The purpose is to prevent timing or
quality claims from mixing distinct mechanisms: exact scoring, dimension
pruning, candidate generation, kernel/layout acceleration, and reranking.

## Source Pins

| Area | Pinned source | Local artifact | Role in this project |
|---|---|---|---|
| BOND / SIGMOD-2002 | Original BOND branch-and-bound nearest-neighbor algorithm from SIGMOD 2002. Bibliographic details still need to be checked against the paper before final writing. | No original-paper implementation is currently pinned in this repo. | Conceptual source for partial scoring over dimensions and upper-bound pruning. It should be cited separately from the local MaxSim implementation work. |
| PDX-BOND-MaxSim preliminary reformulation | Local reformulation of BOND-style dimension pruning for ColBERT MaxSim using a PDX-like per-document dim-major layout, a shared query-level dimension order, Cauchy-Schwarz residual bounds, token pruning, and document pruning. | `archive/preliminaries/05_pruning_bound/pruning_bound_tightness.py`, `archive/preliminaries/09_maxsim_pruning/`, `archive/preliminaries/10_maxsim_pruning_opt/`, `archive/preliminaries/11_pruning_scale_probe/` | Existing mechanism prototype, not just conceptual source material. Stage 1 should audit and formalize this implementation: exact-safe `shrink = 1`, approximate shrink ramp, threshold policy, token-pruning invariant, and PDX compatibility. |
| PDX | Kuffo, Krippner, Boncz, "PDX: A Data Layout for Vector Similarity Search", SIGMOD/PACMMOD 2025; local repository commit `93531b9`. | `extern/PDX/README.md`, `extern/PDX/Setup.md`, `extern/PDX/include/pdx/`, `extern/PDX/python/pdxearch/` | Target layout and implementation family for fast vector search, IVF variants, ADSampling pruner, and PDX-compatible kernels. |
| ADSampling | PDX implementation of random-rotation or DCT-based dimension sampling and approximate pruning; local implementation at `extern/PDX/include/pdx/pruners/adsampling.hpp`. | `extern/PDX/include/pdx/pruners/adsampling.hpp`, `archive/preliminaries/11_pruning_scale_probe/probe.py` | Separate approximate-pruning baseline. It is not BOND unless the Stage 1 bound and pruning rule match BOND's exact-safe semantics. |
| ColBERT MaxSim | ColBERT late interaction objective, `score(q, d) = sum_i max_j <q_i, d_j>`, implemented through PyLate and `pylate.scores.colbert_scores`. | `archive/colbert_scripts/02b_corect_bruteforce.py`, `archive/colbert_scripts/03_beir_comparison.py` | Exact retrieval oracle, correctness target, and reranking scorer. |
| PLAID | ColBERT-specific ANN/indexing path exposed through PyLate `indexes.PLAID` and `retrieve.ColBERT`. | `archive/colbert_scripts/02_corect_quick.py`, `archive/colbert_scripts/03_beir_comparison.py` | Tuned ColBERT ANN baseline. Its speedup is candidate-generation plus ColBERT-specific indexing, not dimension pruning alone. |
| CoRECT | CoRECT framework repository commit `fedf8bb2`. | `extern/CoRECT/README.md`, `extern/CoRECT/src/corect/eval_utils.py`, `extern/CoRECT/src/corect/corect.py`, `archive/colbert_scripts/02_corect_quick.py`, `archive/colbert_scripts/02b_corect_bruteforce.py` | IR evaluation protocol source for BEIR-style datasets, nDCG/MAP/recall/MRR metrics, and RC metrics. |
| PDX SIGMOD snapshot | Earlier PDX SIGMOD-style repository commit `fdc62f2`. Re-pinned to the public `fdc62f2` (sigmod branch tip): the originally-used `996c714` was a local-only commit never pushed upstream, equal to `fdc62f2` plus local benchmark configs/results — preserved at `results/external_baselines/zen5-mxbai/` and `extern/patches/PDX-sigmod-996c714.diff`; `bond.hpp` and `bench_bond` are byte-identical at both commits. | `extern/PDX-sigmod/include/pdx/{bond,bsa,adsampling}.hpp`, `extern/PDX-sigmod/python/pdxearch/`, `extern/PDX-sigmod/benchmarks/bench_bond/`, `extern/PDX-sigmod/examples/pdxearch_exact_bond.py` | **BOND reference implementation.** Unlike the evolved `PDX` (ADSampling only), this snapshot ships BOND, BSA, and ADSampling as PDXearch variants, with `IndexPDXBONDFlat` (exact, exhaustive pruned), IVF variants, a `bench_bond` harness, and existing local results (`results/external_baselines/zen5-martan/IVF_PDX_BOND.csv`). It is single-vector L2 kNN. Primary base to extend (or forward-port) for the multi-vector MaxSim BOND. |

## Terminology Table

| Name | Algorithmic meaning | Implementation meaning in this repo | Mechanism family | Evidence rules |
|---|---|---|---|---|
| Exact MaxSim | Compute every query-token to document-token inner product needed for `sum_i max_j`. | PyTorch/PyLate scorer in `archive/colbert_scripts/02b_corect_bruteforce.py` and `archive/colbert_scripts/03_beir_comparison.py`. | Exact scoring | Correctness oracle. Timing is a baseline only when run on the same machine, thread settings, dataset, and embedding cache policy as the candidate method. |
| BOND for MaxSim | Exact-safe branch-and-bound over embedding dimensions for ColBERT MaxSim, using partial scores and a valid upper bound for `sum_i max_j <q_i, d_j>`. | Single-vector BOND already exists in `PDX-sigmod` (`bond.hpp`, `IndexPDXBONDFlat`). The MaxSim extension is prototyped per-document under `archive/preliminaries/09_maxsim_pruning/`, `10_maxsim_pruning_opt/`, `11_pruning_scale_probe/`. | Dimension pruning | The per-document prototype is the exact-safe **oracle** (Stage 1 proves `shrink=1` exact), not the target. The Stage 2 deliverable is the multi-vector MaxSim extension of PDX's BOND/PDXearch over a wide token block; see `docs/stage1_bond_maxsim_formalization.md` Section 5. |
| Approximate BOND / shrink | Same partial-score machinery as BOND, but residual bounds are intentionally reduced to trade recall for work. | `shrink` parameter in preliminary MaxSim pruning probes. | Dimension pruning, approximate | Report only on a quality-latency frontier. It must be separated from exact-safe BOND. |
| ADSampling | Rotate or transform vectors, scan a subset or prefix of dimensions, and use an approximate threshold rule. | PDX `ADSamplingPruner`; probe arm named `ada` in `archive/preliminaries/11_pruning_scale_probe/probe.py`. | Approximate dimension pruning and transformation | Baseline or ablation, not synonymous with BOND. Quality must be measured against qrels and exact MaxSim agreement. |
| PDX layout | Columnar or decomposed layout that scans dimensions across many vectors efficiently. | Current PDX repository and Python bindings under `extern/PDX/`. | Kernel/layout acceleration | A layout speedup is not automatically a pruning speedup. Report cells scanned and wall-clock separately. |
| PDX-IVF | IVF candidate generation implemented with PDX indexes and PDX kernels. | PDX `IndexPDXIVFTreeSQ8` family and related wrappers. | Candidate generation plus kernel/layout | Compare at fixed candidate counts or fixed quality. Do not attribute skipped clusters to BOND. |
| FAISS-IVF | FAISS inverted-file candidate generation followed by exact or approximate scoring. | Present in the branch plan as a reusable baseline; not yet a tracked main-branch implementation artifact. | Candidate generation | Strong engineering baseline. Keep reranker fixed when comparing candidate generators. |
| PLAID | ColBERT-specific ANN indexing and retrieval with centroid probing and full-score candidates. | PyLate `indexes.PLAID` and `retrieve.ColBERT` in tracked ColBERT scripts. | Candidate generation plus ColBERT-specific scoring | Must be tuned at realistic scale. Small CPU settings are smoke tests, not final evidence. |
| Exact rerank | Score a fixed candidate set with exact MaxSim. | Available through PyLate scorer in current ColBERT scripts; final shared reranker still needs Stage 4 extraction. | Reranking | Used to isolate candidate-generation quality from final scoring quality. |
| CoRECT standard-metric cross-validation | CoRECT's ordinary qrels-based `evaluate_results` function. | The pinned adapter cross-checks NDCG/MAP/Recall/P/MRR against the local evaluator. | IR evaluation | Used to detect metric-definition drift. Relevance Composition requires controlled pools and is not computed here. |

## Baseline Matrix

| Baseline arm | Exact scoring | Dimension pruning | Candidate generation | Reranking | Current implementation status |
|---|---:|---:|---:|---:|---|
| Full exact MaxSim | Yes | No | No | Not applicable | Tracked: `archive/colbert_scripts/02b_corect_bruteforce.py`; reusable as oracle after standardizing output schema. |
| Preliminary PDX-BOND-MaxSim kernel | Yes when `shrink = 1` and the documented Cauchy-Schwarz assumptions hold; approximate when `shrink < 1` | Yes | No | Not applicable | Tracked C++/Python reformulation and probes. Needs Stage 1 proof/audit, Stage 2 schema, and standardized exact-agreement checks before final use. |
| Approximate shrink sweep | No, unless `shrink = 1` and the bound is exact-safe | Yes | No | Not applicable | Tracked preliminary probes. Final role is quality/work frontier only. |
| ADSampling / rotated scan | No | Approximate | No | Not applicable | Available through PDX and preliminary probes. Final role is ablation or external approximate-pruning baseline. |
| PDX exhaustive search | Yes for the metric it implements, usually L2 or IP over single vectors | No or PDX-native pruning only | No | Not applicable | Available in local PDX. Needs adaptation because ColBERT uses multi-vector MaxSim, not single-vector kNN. |
| FAISS-IVF plus exact MaxSim rerank | Rerank only | No | Yes | Yes | Required final baseline. Current main branch does not yet expose the final shared implementation. |
| PDX-IVF plus exact MaxSim rerank | Rerank only | No | Yes | Yes | Required final baseline. Use current PDX snapshot after validating Python bindings and distance semantics. |
| PLAID | Internal ColBERT scoring path | No BOND-style pruning | Yes | Yes, through PLAID full-score candidates | Tracked: `archive/colbert_scripts/02_corect_quick.py` and `03_beir_comparison.py`; must be tuned and rerun under final controls. |

## Stage 0 Decisions

1. The local preliminary method should be called `PDX-BOND-MaxSim
   preliminary` until Stage 1 audits it. If the proof and implementation match,
   it can become the final `BOND-MaxSim` method; otherwise Stage 1 must name the
   corrected variant explicitly.
2. PDX is treated as a layout and implementation substrate. PDX-IVF and
   ADSampling are separate baselines, not evidence that BOND works for MaxSim.
3. Candidate-set methods must be evaluated with fixed reranking where possible.
   Otherwise a candidate-generation improvement can be mistaken for a dimension
   pruning improvement.
4. Final retrieval claims require qrels metrics. Exact MaxSim agreement is a
   correctness signal, not a substitute for nDCG, recall, and MRR; those
   standard metrics are independently cross-validated through CoRECT.
5. Wall-clock timings must be same-machine, same-OS, same-thread-count, and
   same-dataset. Cross-branch exploratory timings remain hypothesis-generating
   only.

## Handoff To Stage 1

Stage 1 should start from this exact-safe candidate condition:

```text
UB(q, d, scanned_dims) < tau_k
```

For ColBERT, `UB` must upper-bound the full late-interaction score:

```text
score(q, d) = sum_i max_j <q_i, d_j>
```

The preliminary PDX-BOND-MaxSim code is the starting candidate definition, not
just background work. Stage 1 must verify normalization, dimension-order
semantics, threshold policy, the `shrink = 1` exact-safe claim, the approximate
shrink ramp, and whether token-level pruning and document-level pruning can
share one bound state without invalidating top-k correctness.
