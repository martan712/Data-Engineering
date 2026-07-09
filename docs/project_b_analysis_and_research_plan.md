# Project B Analysis And Research Plan

Prepared 2026-06-29. Updated 2026-07-02 (added Status And Checklist section,
pinned sources from the project description, synced with Stage 1 conclusions
and the implemented repository structure).

Project goal: speed up multi-vector search as used in ColBERT. The intended
research direction is to use the recent PDX library from the CWI database group
(Kuffo, Krippner, Boncz, SIGMOD 2025) and re-implement the BOND branch-and-bound
idea — de Vries, Mamoulis, Nes, Kersten, "Efficient k-NN Search on Vertically
Decomposed Data", SIGMOD 2002 — in a way that can accelerate ColBERT-style
MaxSim retrieval. The information-retrieval evaluation should use BEIR-style
datasets and, where possible, the CoRECT evaluation framework introduced at
ECIR 2026 by the University of Passau OWS.EU partners.

Scope: single-node CPU retrieval, matching PDX's SIMD/cache-oriented design.
GPU acceleration is out of scope. Two items from the project description are
tracked as context, not core work: Leonardo Kuffo's PDX blog posts are a
pinned introductory source (Stage 0), and the preliminary PDX-in-DuckDB
integration is a possible later-stage extension only — it becomes relevant
if and only if the BOND-MaxSim mechanism wins on the speed-quality frontier.
The CWI group's recent PDX-based fast k-means paper builds primarily on
ADSampling; it is related work that sharpens the BOND-vs-ADSampling contrast
(Stage 0 terminology), not a method arm of this project.

This document has two parts:

1. A short audit of the existing branch work: what is done, what is useful, and
   what is flawed or non-decisive.
2. A ground-up research plan: questions, methodology, experiments, result
   schema, and final paper structure. This part does not treat the existing
   branches as the research design; it treats them only as possible reusable
   components.

## Part I: Existing Work Audit

The existing work is useful as preliminary exploration, but it should not define
the final research project. The final project should start from the ColBERT,
PDX, BOND, and CoRECT research question and then decide which existing pieces
can be reused.

### What Is Already Done

Two branch-level efforts exist.

| Branch/workstream | What it did | Useful output |
|---|---|---|
| `Mikel` / pipeline work | Built an end-to-end ColBERT retrieval pipeline with PyLate embeddings, exact MaxSim, PDX-IVF, FAISS-IVF, PLAID, and exact reranking. | Reusable data preparation, ColBERT embedding export, exact MaxSim oracle, FAISS-IVF baseline, PDX-IVF prototype, PLAID baseline wrapper, reranking code, and BEIR-style metric plumbing. |
| `martan's-experiment` / mechanism work | Explored BOND-style MaxSim pruning with custom C/C++ kernels and supporting NumPy studies. Measured cells scanned, pruning behavior, dimension ordering, and timing-like quantities. | Reusable mechanism instrumentation, candidate BOND/MaxSim kernels, cells-scanned accounting, exact-vs-pruned comparison logic, and several dimension-ordering ideas. |
| `research/Notes` and `research/bond_maxsim` | Drafted a more formal BOND+MaxSim methodology and staged feasibility setup. | Reusable formulas for the Cauchy-Schwarz MaxSim bound, token and document pruning rules, shrink-based approximation, dimension-order signals, and stage-gate experiment structure. |

The current repository README also reports the following preliminary findings:

- Flat PDX-BOND over token vectors was slower than exact NumPy MaxSim in the
  tested ColBERT setup.
- A MaxSim-aware BOND instrumentation found only a small raw pruning ceiling in
  the tested setting.
- IVF candidate generation followed by exact MaxSim rerank produced large
  practical speedups, and FAISS-IVF reproduced the same pattern.
- PLAID was run as a baseline, but the small CPU-only setup is not a decisive
  PLAID evaluation.

These are valuable exploratory observations. They are not yet a complete
research answer.

### What Is Useful

The following pieces should be retained or reused if they pass a reproducibility
check:

- **Exact MaxSim oracle:** needed for correctness, approximation recall, and
  matched-quality sweeps.
- **Embedding and packing utilities:** useful for producing ColBERT query and
  document token embeddings in a consistent format.
- **PDX smoke tests and API notes:** useful because PDX exposes `l2sq`, while
  normalized ColBERT embeddings allow squared-L2 and inner-product orderings to
  be related.
- **FAISS-IVF plus exact rerank baseline:** useful as a strong engineering
  baseline for candidate-set reduction.
- **PDX-IVF prototype:** useful for testing whether PDX can serve as an
  end-to-end candidate generator or reranking backend.
- **PLAID wrapper:** useful as the ColBERT-specific ANN baseline, but it must be
  retuned and rerun at an appropriate scale.
- **Custom C/C++ mechanism kernels:** useful as the main candidate instrument
  for testing whether BOND-style dimension pruning can reduce MaxSim work.
- **Cells-scanned instrumentation:** useful for separating algorithmic work from
  wall-clock behavior.
- **Existing bound formulas:** useful as Stage 1 starting points. In particular,
  `docs/sources/bond_maxsim_methodology.md` defines a Cauchy-Schwarz residual
  bound for MaxSim, document-level pruning, token-level inner-max pruning, a
  shrink ramp for approximate pruning, and three dimension-ordering signals.

### What Is Not Yet Useful As Evidence

Several current results should be treated as preliminary only:

- **Cross-machine timings:** WSL/Linux PDX numbers and Windows FAISS/PLAID
  numbers cannot support direct speed claims.
- **Small-scale PLAID results:** PLAID is designed for larger ColBERT indexes
  and needs tuning. Small CPU runs with weak probe settings should be labeled
  off-design.
- **IVF speedups as BOND evidence:** IVF skips candidates or clusters. That is
  not the same mechanism as BOND dimension pruning.
- **Cells scanned without wall-clock:** fewer multiply-add cells do not imply a
  faster system if bound bookkeeping, live-set maintenance, dimension ordering,
  and memory access dominate.
- **Approximation recall as IR quality:** agreement with exact MaxSim is useful,
  but the final IR claim needs qrels metrics such as nDCG@10, recall@100,
  MRR@10, and CoRECT RC metrics.
- **Ambiguous method names:** "BOND," "ADSampling," "rotation," "PCA," "shared
  bound," and "IVF" must be separated. An experiment that uses rotation plus a
  shared bound is not automatically ADSampling.

### Main Flaws To Fix

The current work has four structural flaws:

1. **The branch history drives the story.** The research should instead begin
   from the project question and use branch artifacts only when they answer a
   specific subquestion.
2. **Mechanisms are conflated.** Dimension pruning, candidate-set pruning, and
   kernel/layout optimization are different sources of speedup.
3. **Fair timing is missing.** Timed comparisons need the same machine, OS,
   thread count, hardware acceleration mode, dataset, and quality constraint.
4. **CoRECT is not yet central.** Existing work mostly uses BEIR-style data and
   metrics. The final research plan needs an explicit CoRECT-style evaluation
   path for information retrieval quality.

### Conservative Current Takeaway

The preliminary work suggests that naive or current BOND-style dimension pruning
does not yet produce a meaningful ColBERT MaxSim speedup, while IVF plus exact
rerank is a strong practical baseline. That is a starting hypothesis, not the
final research conclusion.

## Part II: Ground-Up Research Plan

### Project Definition

ColBERT represents each query and document as multiple token vectors and scores
them with late interaction:

```text
score(q, d) = sum_i max_j <q_i, d_j>
```

The central systems problem is that exact MaxSim requires many token-level inner
products. The central research idea is to use BOND-style branch-and-bound over
embedding dimensions, implemented in a PDX-compatible layout, to avoid enough of
that work to reduce latency while preserving retrieval quality.

The project must evaluate three distinct acceleration mechanisms:

- **Dimension pruning:** BOND-style partial scoring and upper bounds over
  remaining dimensions.
- **Candidate-set pruning:** IVF, PLAID, or other methods that reduce which
  documents are scored.
- **Kernel/layout acceleration:** PDX-compatible memory layout, fixed or blocked
  columnar access, SIMD, threading, and reduced overhead.

The intended contribution is only established if the BOND/PDX dimension-pruning
path improves the speed-quality frontier for ColBERT-style retrieval. Candidate
pruning and exact reranking remain important baselines.

### Primary Research Question

Can a PDX-compatible implementation of BOND-style dimension pruning accelerate
ColBERT MaxSim retrieval at matched retrieval quality?

### Subquestions

**Algorithmic questions**

- How should the BOND/SIGMOD-2002 branch-and-bound idea be formulated for
  ColBERT MaxSim?
- What upper bound is valid for `sum_i max_j <q_i, d_j>` after only a prefix of
  dimensions has been scanned?
- Does the bound become tight early enough to prune documents before most
  MaxSim work has already been done?
- Which dimension order is best: natural, query-energy `q^2`, distance to means
  `|q - mu|`, variance-aware `q^2 * (mu^2 + var)`, PCA/rotation-based, or
  another MaxSim-specific order?

**Systems questions**

- Can the pruning algorithm be implemented in a PDX-compatible memory layout
  without losing its gains to scattered access or live-set bookkeeping?
- How much of the work reduction survives in wall-clock latency?
- Where is the crossover point as corpus size, document length, embedding
  dimension, query length, and thread count change?

**IR questions**

- At matched quality, does the method beat exact MaxSim, FAISS-IVF plus exact
  rerank, PDX-IVF plus exact rerank, and tuned PLAID?
- Does CoRECT reveal quality loss that is not visible in recall against exact
  MaxSim?
- Is the contribution a general ColBERT retrieval speedup, a negative result
  about BOND for MaxSim, or a narrower kernel/layout result?

### Methodology

The methodology proceeds from theory to mechanism to retrieval evaluation. Each
stage has a stopping condition.

#### Stage 0: Establish References

Define and pin the sources:

- the BOND/SIGMOD-2002 algorithm and its original assumptions (de Vries,
  Mamoulis, Nes, Kersten, SIGMOD 2002 — cite the paper itself, not only the
  PDX re-implementation);
- the PDX paper, implementation, and available search variants, plus Leonardo
  Kuffo's blog posts as the practical entry point to the codebase;
- ADSampling and how it differs from BOND, including the CWI PDX-based
  k-means paper as the group's own ADSampling-based follow-up;
- ColBERT MaxSim and PLAID;
- CoRECT metrics and evaluation protocol (ECIR 2026, University of Passau).

Output:

- a terminology table mapping every method name to an algorithm and
  implementation;
- a baseline list that separates exact scoring, dimension pruning, candidate
  generation, and reranking.

Stage 0 artifact:

- `docs/stage0_references_and_baselines.md`

Existing work that may be reused:

- PDX smoke tests and README API notes;
- existing exact MaxSim and embedding utilities;
- existing PLAID, FAISS, and PDX wrappers after validation.

#### Stage 1: Formalize BOND For MaxSim

Derive the exact scoring objective and the pruning rule.

For a document to be safely skipped in top-k retrieval, its upper bound after a
partial dimension scan must fall below the current kth-best threshold:

```text
UB(q, d, scanned_dims) < tau_k
```

For ColBERT, `UB` must remain valid after the `max_j` over document tokens and
the sum over query tokens. The work in this stage is to prove the chosen bound,
state whether it is exact-safe or approximate, and identify what statistics are
needed to compute it.

Output:

- a mathematical definition of the MaxSim BOND variant;
- exact-safe and approximate variants, clearly separated;
- expected costs for score updates, bound updates, and live-set maintenance.

Stage 1 artifact:

- `docs/stage1_bond_maxsim_formalization.md` — formalization, a proof that the
  preliminary `shrink = 1` kernel is exact-safe (including the token-pruning
  survival invariant), the approximate `shrink < 1` separation, exactness
  preconditions (chiefly unit normalization), the PDX-like-vs-PDX layout
  boundary, a cost model, and the Stage 2 audit checklist.

Existing work that may be reused:

- Martan's MaxSim-aware BOND derivation and instrumentation, if it matches the
  formal definition.
- The formulas in `docs/sources/bond_maxsim_methodology.md`, especially M5
  and M6, as candidate definitions to verify rather than rediscover.

Candidate formulas already present in the existing work:

- **Partial/residual split:** for scanned dimensions `S`,
  `<q_i, d_j> = <q_i[S], d_j[S]> + <q_i[S^c], d_j[S^c]>`.
- **Cauchy-Schwarz residual:** for unit-normalized tokens,
  `|<q_i[S^c], d_j[S^c]>| <= sqrt(1 - sum_{z in S} q_i,z^2) *
  sqrt(1 - sum_{z in S} d_j,z^2)`.
- **Pair interval:** `U_ij = P_ij + residual_ij` and
  `L_ij = P_ij - residual_ij`.
- **Document upper bound:** `UB_d = sum_i max_j U_ij`; document `d` is safely
  pruned when `UB_d < tau_k`.
- **Token lower bound:** `L_i = max_j L_ij`; a document token can be removed
  from the inner MaxSim competition when it cannot beat `L_i` for any query
  token.
- **Approximate shrink ramp:** residuals can be scaled by a depth-dependent
  factor `beta(k)` to sweep a work-recall frontier, with `shrink = 1` retaining
  the exact-safe bound.
- **Dimension-order signals:** `q^2`, `|q - mu|`, and
  `q^2 * (mu^2 + var)` are already proposed. Stage 1 should verify which of
  these is valid, useful, and PDX-compatible.

These formulas should be copied into the final methodology only after checking
their assumptions: unit normalization, whether `S` is a prefix or arbitrary
dimension subset, whether token pruning and document pruning share the same
bound state, and whether the threshold `tau_k` is self-contained, seeded, or
oracle-only.

#### Stage 2: Build A Mechanism Testbed

Implement a small, reproducible mechanism testbed before integrating with a full
retrieval pipeline.

Stage 1 conclusions that shape this stage (see
`docs/stage1_bond_maxsim_formalization.md` §5):

- The preliminary per-document kernel prunes at the wrong granularity
  ("Option B"). It is retained only as the **exact-safe accounting oracle**
  (`cpp/per_document_oracle/`); its wall-clock results are not evidence
  against BOND-in-PDX.
- The Stage 2 deliverable is the **wide-token-block MaxSim BOND**
  (`cpp/wide_block_maxsim_bond/`): document tokens laid out dim-major across
  many documents, scanned dimension-incrementally, with the Section 2 pair
  intervals aggregated to per-document upper bounds and per-document live-token
  sets maintained inside the wide block.
- **Threshold seeding is first-class**, not an ablation: a synchronized
  wide-block scan finalizes no document until the end of the pass, so the
  self-threshold stays at `-inf` mid-pass. Seeded thresholds (IVF, PLAID, or a
  prior-pass lower bound) are the realistic operating mode.

Blocking checks before any Stage 3 result is trusted: the unit-norm guard
(residual bounds use `1 - sumsq` and silently break when `||x|| > 1`) and the
`shrink = 1` exact-agreement gate (`recall_vs_exact@10 == 1.0`).

Required inputs:

- packed document token embeddings;
- document offsets;
- query token embeddings;
- dimension order;
- threshold policy;
- `k`;
- dataset identifiers;
- thread count and compiler settings.

Required outputs:

- top-k document ids and scores;
- exact MaxSim agreement;
- qrels metrics when labels exist;
- cells scanned;
- bound checks;
- pruned documents;
- live-set operations;
- runtime and QPS.

The testbed needs two modes:

- **Accounting mode:** measures algorithmic work precisely.
- **Throughput mode:** measures realistic wall-clock behavior under the intended
  layout and compiler settings.

Stopping condition:

- If the mechanism cannot reduce cells scanned under an exact-safe bound, do not
  integrate it further until a better bound or ordering is found.
- If it reduces cells but not wall-clock time, the research becomes a systems
  diagnosis of where the overhead enters.

Existing work that may be reused:

- Martan's custom C/C++ kernels and cells-scanned instrumentation, after adding
  the standardized inputs and outputs above.

Stage 2 artifacts:

- `src/bondmaxsim/` — research package (config, schema, data, oracle, ordering,
  threshold, kernels, testbed, baselines, eval), one responsibility per module;
- `cpp/per_document_oracle/` — exact-safe accounting/throughput oracle kernels;
- `cpp/wide_block_maxsim_bond/` — the wide-block MaxSim BOND kernel (the
  mechanism under test);
- `experiments/stage2_testbed/` — normalization guard, exact-agreement gate,
  two-mode smoke drivers;
- `tests/` — blocking checks run via `uv run pytest`.

#### Stage 3: Run Mechanism Experiments

Run controlled experiments that explain whether BOND is viable for MaxSim.

Datasets:

- SciFact, NFCorpus, ArguAna, SCIDOCS for BEIR-scale debugging (already
  exported to `data/embeddings/*.npz`, D=128, unit-normalized, 200 queries
  each);
- at least one 100k to 1M document setting for scale — preferred source: the
  CoRECT controlled corpus pools, which supply the scale axis and keep Stage 3
  consistent with the Stage 5 evaluation (fallback: a larger BEIR corpus);
- optional synthetic data to isolate dimension, length, and score-distribution
  effects.

Experiments:

- bound slack versus score dispersion;
- pruning rate over dimension prefixes;
- dimension-order ablation;
- exact-safe pruning;
- approximate matched-recall sweeps;
- PCA/rotation and other PDX-compatible transformations;
- threshold policies including current lower-bound threshold, oracle threshold,
  and IVF-seeded threshold;
- cache and layout sensitivity.

Decision:

- If no exact-safe or matched-quality approximate variant produces a repeatable
  wall-clock win, report a negative result with mechanism evidence.
- If a variant wins, promote only that path to retrieval integration.

Existing work that may be reused:

- current MaxSim-BOND instrumentation as one experiment arm;
- existing PCA/rotation experiments as preliminary points to rerun under the
  shared schema;
- earlier feasibility figures (untracked `research/bond_maxsim/` working
  directory; not on this branch) as plot templates only — every figure used as
  evidence must be regenerated by a driver in `experiments/stage3_mechanism/`
  into `results/figures/stage3_mechanism/`.

Stage 3 artifacts:

- `experiments/stage3_mechanism/e01..e07` — one driver per experiment (see the
  README in that directory for the experiment specifications);
- `results/json/stage3_mechanism_*.json` and
  `results/figures/stage3_mechanism/` — regenerated evidence under the shared
  schema.

#### Stage 4: Integrate Candidate Kernel Path

Integrate only the winning mechanism from Stage 3. Keep exact MaxSim as the
correctness oracle.

Methods to compare:

- exact MaxSim;
- BOND/PDX kernel over the full corpus;
- BOND/PDX kernel as reranker over a fixed candidate set;
- FAISS-IVF plus exact MaxSim rerank;
- PDX-IVF plus exact MaxSim rerank;
- PLAID with tuned settings.

Important control:

Candidate-set size must be held fixed when testing the kernel. Otherwise a
speedup from IVF or PLAID can be mistaken for a BOND kernel speedup.

Existing work that may be reused:

- FAISS-IVF plus exact rerank pipeline;
- PDX-IVF prototype;
- exact reranking utilities;
- the existing threshold modes from
  `archive/reference/05_maxsim_bond_instrumentation.py` (brought on-branch from
  `Mikel`; threshold logic to be reimplemented in `src/bondmaxsim/threshold/`): self-bound,
  oracle-threshold, and seeded-threshold variants. These should become explicit
  method arms if they survive Stage 1 formalization. (Stage 1 outcome: they
  survive; seeded threshold is first-class — see Stage 2 notes above.)

Stage 4 artifacts:

- `src/bondmaxsim/baselines/` — faiss_ivf / pdx_ivf / plaid wrappers;
- `experiments/stage4_integration/` — method arms at fixed candidate sets.

#### Stage 5: CoRECT-Style IR Evaluation

Evaluate final methods as retrieval systems, not just as scoring kernels.

Metrics:

- nDCG@10;
- recall@100;
- MRR@10;
- recall versus exact MaxSim@10;
- CoRECT RC metrics;
- ms/query and QPS;
- memory use and indexing time, reported separately from query latency.

Fairness controls:

- one machine;
- one OS;
- fixed thread count;
- fixed hardware acceleration mode;
- repeated runs with confidence intervals;
- matched retrieval quality or a full quality-latency frontier;
- tuned baselines, especially PLAID.

Existing work that may be reused:

- BEIR preparation scripts;
- qrels metric utilities;
- CoRECT repository code (pinned at `extern/CoRECT/`), after adding or
  specifying a ColBERT/MaxSim wrapper.

Stage 5 artifacts:

- `src/bondmaxsim/eval/` — qrels metrics and the CoRECT adapter;
- `experiments/stage5_corect/` — IR evaluation drivers.

### How The Stages Answer The Questions

| Question | Answered by | Decisive evidence |
|---|---|---|
| Can BOND be formulated correctly for MaxSim? | Stage 1 | Valid upper bound and pruning condition for `sum_i max_j <q_i, d_j>`. |
| Does the bound prune enough work? | Stage 2 and Stage 3 | Cells scanned, pruned documents, bound checks, and exact agreement. |
| Does work reduction become speed? | Stage 2 and Stage 3 | Same-machine throughput-mode wall-clock latency. |
| Is the speedup really BOND rather than IVF/PLAID? | Stage 4 | Fixed candidate sets and method separation. |
| Does it improve IR quality-latency tradeoffs? | Stage 5 | nDCG/recall/MRR/CoRECT-RC frontier at matched quality. |
| Is a negative result publishable? | Stage 1 to Stage 5 | Formal obstruction or empirical mechanism evidence plus fair baselines. |

### Prior And Intermediate Experiments

These experiments should be run before making final claims:

- **Artifact inventory:** map existing branch outputs to script, dataset,
  method, metric, machine, and result file.
- **Exact oracle validation:** verify exact MaxSim implementation, top-k
  semantics, and normalized-vector assumptions.
- **PDX API validation:** confirm metric support, layout choices, and
  BOND/ADSampling/BSA/IVF availability per checkout. `PDX-sigmod` (commit
  `fdc62f2` (public sigmod tip; see Stage 0)) ships BOND, BSA, and ADSampling as PDXearch variants
  (`include/pdx/{bond,bsa,adsampling}.hpp`, `IndexPDXBONDFlat`, `bench_bond`); the
  evolved `PDX` (commit `93531b9`) kept only `pruners/adsampling.hpp` in a hybrid
  25/75 layout, flagship `IndexPDXIVFTreeSQ8`.
- **BOND reference validation:** PDX's existing `bond.hpp` is a *single-vector
  L2* PDXearch variant ("adds no relevant functionality to PDXearch"). Compare
  the SIGMOD-2002 algorithm with this reference, then with the proposed
  multi-vector MaxSim adaptation (inner-product pair bounds, two-level
  token->document pruning). The MaxSim extension is the contribution; BOND itself
  is already implemented for single-vector kNN.
- **Formula inventory:** extract the existing MaxSim bound, token-pruning rule,
  document-pruning rule, shrink ramp, and ordering signals from
  `docs/sources/bond_maxsim_methodology.md` and
  `archive/reference/05_maxsim_bond_instrumentation.py` (brought on-branch from
  `Mikel`; threshold logic to be reimplemented in `src/bondmaxsim/threshold/`); mark each one as
  exact-safe, approximate, diagnostic, or oracle-only.
- **Single-vector sanity check:** run PDX BOND/ADSampling/BSA on a standard
  single-vector dataset to confirm the local setup behaves like the PDX paper
  expects.
- **Small deterministic MaxSim cases:** manually inspect score, upper bound,
  pruning decision, and top-k output.
- **Bound feasibility diagnostic:** plot score dispersion and bound slack over
  dimension prefixes for each dataset.
- **Dimension-order sweep:** natural, `q^2`, `|q - mu|`,
  `q^2 * (mu^2 + var)`, PCA/rotation, and IVF-seeded thresholds.
- **Matched-quality sweep:** report latency only at equal recall versus exact
  MaxSim or equal qrels quality.
- **PLAID retuning:** rerun PLAID at a suitable scale and probe setting.
- **CoRECT wrapper smoke test:** verify standard qrels metrics and RC metrics on
  one small ColBERT run before scaling.

### Shared Result Schema

All experiments write one shared result format. The **canonical definition is
`src/bondmaxsim/schema.py` (`ResultRecord`)**; the JSON below is an
illustrative example, and its keys are spelled Python-style in code
(`recall_vs_exact_at_10`, `nDCG_at_10`, ...). Two extensions are planned and
tracked in the checklist: an explicit `shrink` field (so exact vs approximate
arms are machine-separable, not only encoded in `method`/`notes`) and
token-level pruning counters (`tokens_pruned_pct`, per-block live counts)
required by experiment e02.

```json
{
  "dataset": "scifact",
  "num_docs": 5183,
  "num_queries": 300,
  "method": "bond_pdx_maxsim_exact_safe",
  "candidate_budget": null,
  "dimension_order": "q2_mu_var",
  "threshold_policy": "exact_safe_topk",
  "recall_vs_exact@10": 1.0,
  "nDCG@10": 0.0,
  "recall@100": 0.0,
  "MRR@10": 0.0,
  "CoRECT_RC_metrics": {},
  "ms_per_query": 0.0,
  "qps": 0.0,
  "cells_scanned_pct": 100.0,
  "pruned_docs_pct": 0.0,
  "bound_checks_per_query": 0,
  "machine": "hostname-or-cpu",
  "os": "linux",
  "thread_count": 1,
  "notes": "placeholder values in schema example"
}
```

Use `null` only when a field does not apply. Do not leave quality metrics at
zero in final results; the zeros above are placeholders.

### Final Paper Structure

The final paper should not be organized as a branch report. It should be
organized as a research argument.

1. **Introduction:** State the problem of accelerating ColBERT-style
   multi-vector search and the hypothesis that PDX-compatible BOND can reduce
   MaxSim work.
2. **Background:** Explain ColBERT MaxSim, PDX, BOND/SIGMOD-2002, ADSampling,
   PLAID, IVF, and CoRECT.
3. **Problem Formulation:** Define exact MaxSim, top-k retrieval, exact-safe
   pruning, approximate pruning, matched retrieval quality, and latency metrics.
4. **BOND For MaxSim:** Present the proposed upper bound, dimension orders,
   threshold policies, and PDX layout constraints.
5. **Experimental Methodology:** Describe datasets, models, baselines, hardware,
   result schema, and fairness controls.
6. **Mechanism Results:** Report bound slack, score dispersion, cells scanned,
   pruned documents, ordering overhead, and kernel wall-clock behavior.
7. **Retrieval Results:** Report exact MaxSim, BOND/PDX, FAISS-IVF rerank,
   PDX-IVF rerank, and PLAID on matched-quality frontiers.
8. **CoRECT Evaluation:** Report nDCG@10, recall@100, MRR@10, RC metrics, and
   quality-latency tradeoffs.
9. **Discussion:** Explain whether BOND succeeds, fails because of MaxSim,
   fails because of layout/overhead, or only works in a restricted regime.
10. **Conclusion:** Answer the primary research question and state whether the
    contribution is a new retrieval backend, a negative result, or a reusable
    mechanism testbed.

### Validation Checklist

- Every speedup states baseline, quality constraint, machine, OS, thread count,
  and acceleration mode.
- Every method name maps to an implementation and algorithmic mechanism.
- Approximation recall, qrels metrics, cells scanned, and wall-clock latency are
  reported separately.
- Cross-machine timings and off-design PLAID runs are explicitly non-decisive.
- IVF speedups are never presented as BOND speedups.
- Existing branch artifacts are cited only when they answer a named research
  question or supply a reusable implementation component.
- CoRECT, PDX, BOND, ADSampling, ColBERT MaxSim, PLAID, FAISS-IVF, and custom
  kernels all appear in the final methodology.

## Status And Research Plan (Revised)

Scope decision (2026-07-08): finalize the paper on the exact-safe arm. RQ4
(approximate `shrink < 1` frontier) is rescoped OUT; R5 and R7 (scale) are
deferred to future work. AMENDED 2026-07-09: R9 (Stage 5 CoRECT evaluation)
is rescoped BACK IN as the paper's final section — take the best methods
from Stages 3–4 and show how they perform in the proper IR evaluation
framework (qrels metrics + CoRECT RC, fairness controls). Remaining critical
path: R9 (Stage 5) then R10 (write-up). Contributions that stand: RQ1
(mechanism), RQ2 (fused dense kernel, 3.2x over OpenBLAS), RQ3
(single-dataset exact-safe win), R8 (system), and the Stage 5 IR evaluation.

Last updated 2026-07-09 (sixth pass: R8 / Stage 4 landed — e01/e02/e03 on
all four datasets, MT + 1T, interleaved baseline; methodology + Mikel-chart
dissection in `docs/stage4_comparison_methodology.md`. Headlines: (1) at
matched candidate budgets every token-level candidate pipeline (faiss_ivf /
plaid / pdx_ivf) loses to the exhaustive fused dense scan on 3 of 4
datasets — per-query candidate generation costs more than the whole scan;
only on scidocs (25.7k docs) do faiss/pdx at B=100 win. (2) A strong IVF
tau-seed recovers ~99% of the ORACLE pruning but its seed cost exceeds the
kernel saving everywhere — the R12c oracle margins are reachable but not
monetizable; self_bound stays the honest free policy. (3) The partitioned
fused scan is a real approximate frontier (2–5x at recall 0.87–0.95 on
arguana/scidocs) but its full-probe exact control is SLOWER than the
monolithic scan and the exact-safe partition bound prunes 0% — approximate
only. RQ5 answered; R8 closed. Same day: R9 / Stage 5 rescoped BACK IN as
the paper's final section (see the scope decision above) — remaining
critical path is R9 then R10.
Prior: 2026-07-04 (fifth pass: R12a + R12b + R12c analysis. R12a e09
rerun landed overnight (all 4 datasets, 24 arms, recall 1.0) → adopt the TIGHT
doc-level bound (cheap prunes 0% at every late checkpoint — reverses the
BOND-2002 lesson). R12b: the e08 "wall-clock exceeds cells" finding is a
MEASUREMENT ARTIFACT of a standalone dense baseline — interleaved, arguana's
margin corrects from −19% MT to +8.7%. R12c re-measured all 4 datasets
interleaved at 1T+all-cores (driver `r12c_interleaved_exact_safe.py`):
corrected all-cores margins are arguana +8.7%, scidocs +0.9%, scifact +2.3%,
nfcorpus +0.9% — only arguana wins, so **G1 is NOT met at the honest all-cores
baseline** (single-dataset, not ≥2). All future wall-clock claims must
interleave the baseline. R12 is now CLOSED (a/b/c all done); the TIGHT+C={112}
fold into e05 is carried as an R5 dependency.
Fourth pass, PREP ONLY: extended the e09 driver to
the e08-winning late checkpoint sets for R12a and wired it into
`run_pending_experiments.sh` for an overnight user run — no new results yet;
analysis deferred to a fresh agent. Prior: 2026-07-03 third pass: e07
(scifact/nfcorpus/arguana), e08
(all 4 datasets), and e09 (all 4 datasets) runs landed. Headline: e08 shows
LATE checkpoints (C={96}/{112}) prune 88–98% of documents exact-safe and
beat the fused dense baseline on 3 of 4 datasets — the RQ3 verdict is no
longer uniformly negative; see RQ3 below. (CORRECTED by R12c below: that
"3 of 4" used a standalone dense baseline and was inflated; interleaved,
only arguana wins at all cores and G1 is NOT met.) Also: tie-aware exact-agreement
fix in the fused testbed path (a rank-10 score tie on nfcorpus tripped the
shrink=1 gate; boundary ties are now accepted given oracle scores, as the
wide-block path already did). Second pass: fused-kernel overhead revision —
stage3b doc §6.1.1: fused sumsq, register-folded final segment,
next-panel prefetch — plus the R2 checkpoint simulator, the R3 checkpoint
parameterization + e08 driver, and the R5/R6 driver upgrades. First pass
same day: fused panel kernels replaced the wall-clock instruments, Stage 1
proof extended, first — negative — exact-safe data point on scifact). This
section is the working plan for future sessions/agents. Keep it and the status table in `docs/project_structure.md`
in sync. Conventions (uv toolchain, thin drivers, shared schema,
exact/approx separation) are in `docs/project_structure.md`.

### Research question and decomposition

Can BOND-style dimension-incremental pruning on the PDX columnar layout
accelerate ColBERT MaxSim top-k retrieval on a single CPU node — exactly
(`shrink = 1`) or on a recall/latency frontier (`shrink < 1`)?

- **RQ1 (mechanism)**: how much algorithmic work can the bounds save, and
  WHERE in the dimension scan does pruning become possible?
  → e01 (bound slack), e02 (survival curves). **Answered — e01/e02 now run
  on all four datasets** (arguana + scidocs added 2026-07-03; the analysis
  below cites scifact/nfcorpus): documents survive until late in the scan —
  under the oracle
  threshold on scifact, 98–100% of documents are still alive at dim 64 of
  128; total avoidable work (cells) is only 8–16% even with token-level
  pruning and per-boundary checks. e08 (2026-07-03) confirmed the flip
  side: checkpoints placed AFTER the survival cliff (dims 96–112) do
  prune 88–98% of documents — the cells ceiling stays modest, and the
  wall-clock saving tracks it (~−12%) once the dense baseline is measured
  interleaved rather than standalone (R12b corrected the earlier "wall-clock
  exceeds cells" reading to a measurement artifact).
- **RQ2 (engineering)**: what is the strongest honest dense baseline, and
  does the PDX layout matter in the MaxSim regime?
  → Stage 3b. **Answered**: MaxSim (m≈21) is GEMM-shaped, so the layout wins
  through packing amortization + fused epilogue, not the paper's m=1
  argument; the fused dense kernel runs 13.1 ms/q on scifact (all cores) vs
  42.5 ms/q for 12-thread OpenBLAS — this is the decision-gate baseline.
- **RQ3 (exact-safe gate)**: does `shrink = 1` BOND pruning beat the fused
  dense baseline in wall-clock?
  → e03/e04/e08. **Split verdict (e08, all 4 datasets, 2026-07-03): the
  checkpoint SET decides.** At the default early checkpoints {32, 64}
  pruning fires on <2% of documents (consistent with RQ1's survival curves)
  and BOND never beats dense — the scifact e03 finding generalizes. But at
  LATE checkpoints (C={96}/{112}/supersets) the same exact-safe kernel
  prunes 88–98% of documents (recall 1.0 on all 16 arms × 4 datasets). The
  STANDALONE-baseline e08 numbers looked strong — arguana 29.1 vs 36.1 ms/q
  (−19% MT) and 126.6 vs 179.6 (−29% 1T), scidocs −8%, nfcorpus −3%, scifact
  tie — but **R12b (2026-07-04) showed those margins are largely a
  measurement artifact**: e08/e09 time the dense baseline ONCE in isolation,
  so a thermally unlucky baseline window inflates every pruning arm. Measured
  INTERLEAVED back-to-back (dense and BOND round-robin, drift cancels) on
  arguana: the 1T zero-prune offset collapses from +19% to +0.2% (the
  standalone dense 179.6 ms/q was throttled — interleaved it is 149), and the
  MT saving at C={112} corrects from −19.6% to **+8.7%**, with a −4.3%
  zero-prune checkpoint OVERHEAD. Natural order wins everywhere — bond order
  prunes slightly more but its permuted access costs more than the extra
  pruning saves. Mechanism questions: (a) **RESOLVED (R12b)** — wall-clock
  savings do NOT genuinely exceed the cells prediction; the apparent excess
  was the standalone-baseline artifact, and once the baseline is fair the
  ~−12% cells prediction holds (arguana MT: +8.7% ≈ 12% cells − ~4% checkpoint
  overhead). There is no "disproportionately expensive final segment." (b)
  e09/R12a re-asked the tight-vs-cheap bound at the winning late checkpoints
  (done — see R12/§R12a below). **Corrected all-cores verdict (R12c, all 4
  datasets, interleaved): only arguana clears a real exact-safe margin (+8.7%,
  was +19.6%); scidocs +0.9% (was +5.5%), scifact +2.3%, nfcorpus +0.9% are
  ties. G1 is NOT met at the honest all-cores baseline — a single-dataset win,
  not a ≥2-dataset repeatable margin.** (1T: all four positive, +3.3…+14.1% at
  C={112}, but 1T is not the decision baseline.) Every wall-clock win must be
  re-measured interleaved before it is headlined.
  e03/e04 ran on all four datasets (2026-07-03) and confirm the default-C
  negative across the board — the RQ3 record is complete (e07 on scidocs
  deferred, memory issue).
- **RQ4 (approximate frontier) — OUT OF SCOPE (rescoped 2026-07-08).** The
  project is finalized on the exact-safe arm (`shrink = 1`), whose quality
  guarantee is exact agreement with MaxSim (`recall_vs_exact@10 = 1.0`). The
  approximate `shrink < 1` frontier is deferred to future work.
  → e05 preliminary (scifact + nfcorpus, default C={32,64}, 2026-07-03): NO
  G2 point — shrink=0.8 buys 20–25% latency but only at recall 0.96–0.97; by
  shrink=0.9 recall is back to 1.0 but pruning has collapsed (≤1.5% docs) and
  the win is gone. This preliminary negative stands as motivation for the
  future-work frontier (late-checkpoint operating point); it is no longer a
  decision question for this paper.
- **RQ5 (system context)**: candidate-set seeding (IVF/PLAID) interplay at
  fixed candidate sets — the fused kernel as the reranker in an IVF pipeline.
  → Stage 4 (R8). **Answered (2026-07-09, e01/e02/e03 on all four datasets;
  details under R8 below, controls in `docs/stage4_comparison_methodology.md`):
  at this corpus scale the exhaustive fused dense scan IS the system** — at
  matched budgets every token-level candidate pipeline (faiss_ivf, plaid,
  pdx_ivf) is slower than scanning the whole corpus with the RQ2 kernel on
  scifact/nfcorpus/arguana; the crossover appears only on scidocs (25.7k
  docs, faiss/pdx +56/+62% at B=100, recall 0.92). Realistic IVF tau-seeding
  recovers ~99% of the oracle pruning but never profitably (seed cost >
  kernel saving); the partitioned fused scan gives a genuine approximate
  frontier whose exact control is slower than the monolithic scan.
  IR-quality metrics under CoRECT (Stage 5 / R9) close the loop: rescoped
  back in 2026-07-09 as the paper's final section (exact agreement remains
  the quality guarantee for the exact-safe arms; CoRECT shows how the best
  methods rank as retrieval systems).

A single-dataset RQ3 result with a defensible baseline, a quantified mechanism
explanation (RQ1), the standalone RQ2 dense-kernel contribution, and the
Stage 4 system verdict (the RQ2 kernel already removed the fat that candidate
pipelines are designed to cut — landed 2026-07-09) is a sound, publishable
result; the Stage 5 CoRECT evaluation (R9, rescoped back in) closes the paper
by showing the best methods in the proper IR framework. The approximate
frontier (RQ4) and scale (R7) are future work.

### Instruments (shared bound math, two distinct ALGORITHMS)

The Cauchy-Schwarz bound math is shared (Stage 1 §2, extended §10), but the
two pruning kernels are DIFFERENT ALGORITHMS, not one algorithm with two
meters (Stage 3b §5.7 has the full table): the wide-block kernel is
breadth-first (all group tokens advance through dimensions in lockstep) with
token-level + document-level pruning at every fetch boundary — the faithful
PDX-BOND extension; the fused BOND kernel is depth-first (document by
document) with document-level pruning at sparse checkpoints — architecturally
the "Option B" granularity Stage 1 demoted, deliberately readopted at
wall-clock because e02/e03 showed token pruning fires as late as document
pruning and its bookkeeping costs more than it saves, while depth-first
scanning keeps checkpoint re-scans L2-resident and gives self_bound a
continuously rising τ. Consequence for the paper: "wide token block"
describes the ANALYSIS instrument only; the wall-clock artifact is
doc-at-a-time. Roles after the Stage 3b revision:

| instrument | role | status |
|---|---|---|
| NumPy exact oracle (`oracle/exact_maxsim.py`) | ground truth for recall gates | active |
| per-document oracle kernel (`cpp/per_document_oracle/`) | exact-safe reference implementation of the bound rule | active (tests) |
| wide-block ACCOUNTING kernel (`cpp/wide_block_maxsim_bond/`) | mechanism microscope: true cells, token+doc pruning at EVERY fetch boundary — the upper envelope of what any policy could prune | active (e01/e02/e05/e06 accounting arms) |
| wide-block THROUGHPUT kernel | — | **retired** (its wall-clock numbers measured runtime-`m` codegen artifacts, Stage 3b §1; kept in-tree only for the Stage 2 record + gate tests) |
| wide-block dense scan ("brute_pdx") | — | **removed** (superseded) |
| fused panel BRUTE (`cpp/fused_panel_maxsim/`) | dense wall-clock baseline (decision gate) | active |
| fused panel BOND (doc + token levels) | wall-clock mechanism instrument; checkpoint set C parameterized (R3); revised 2026-07-03 for overhead (§6.1.1) | active |
| fused panel BOND cheap bound (`_bond_cheap`) | doc-level arm with the query-only H_q-analog bound (BOND SIGMOD-2002 lesson; `docs/bond2002_bound_cost_analysis.md`) — bookkeeping-vs-tightness ablation | active (R11/e09 run 2026-07-03; re-test at late checkpoints = R12) |
| NumPy checkpoint simulator (`oracle/checkpoint_sim.py`) | instrument-aligned accounting: docs pruned + cells% under the FUSED doc-checkpoint policy at fixed tau — PREDICTS fused wall-clock savings | active (R2, validated vs kernel stats in `tests/test_checkpoint_sim.py`) |
| NumPy/BLAS dense (1T + all cores) | external reference baseline | active |

Instrument-alignment gap (CLOSED by R2, 2026-07-03): because the two pruning
kernels are different algorithms, the wide-block accounting numbers
(per-boundary, token+document, breadth-first) describe the mechanism's upper
envelope, NOT what the wall-clock algorithm can capture — accounting
`pruned_docs_pct` (99.8%) vs fused `pruned_docs_pct` (<2%) answer different
questions. Both are correct; e02's survival curves reconcile them. The
checkpoint simulator now provides the missing third number: accounting under
the FUSED algorithm itself (doc-at-a-time, document-level, checkpoint set C),
used by e08.

### Done (condensed history)

- [x] Repo restructure, uv toolchain, pinned submodules (PDX @ `93531b9`,
      PDX-sigmod @ `fdc62f2`, CoRECT @ `fedf8bb2`).
- [x] Stage 0: references, terminology, baseline matrix, external ZEN5 runs.
- [x] Stage 1: `shrink = 1` exact-safety proof + §10 addendum (2026-07-03)
      transferring it to the fused panel kernel (coarser-granularity, padding,
      query-tile, shared-threshold lemmas; checkpoint set is a free
      performance parameter).
- [x] Stage 2 (instruments v1): wide-block accounting+throughput kernels,
      per-document oracle, packing, orders, threshold policies, schema,
      Runner; blocking checks green (unit-norm guard, `shrink = 1`
      exact-agreement on all four datasets); s01–s03 smoke results committed.
- [x] Stage 3 e01 (bound slack) + e02 (two-level survival curves) run on
      scifact + nfcorpus; results committed. These are the RQ1 evidence.
- [x] Stage 3b (instruments v2 — fused panel kernels), design + theory in
      `docs/stage3b_fused_panel_maxsim_kernel.md`:
      - K1 `pack_corpus_panels` (16-token panel-major; duplicate-last-token
        padding, proven max-invariant).
      - K2/K3 fused BRUTE kernel (AVX-512 register tile, fused per-doc max,
        OpenMP; 54.9 ms/q 1T / 13.1 ms/q all-cores on scifact).
      - K4 fused BOND kernel (doc checkpoints {32,64}, shared rising τ;
        exact-agreement gate green across orders × policies × threads).
      - K5 testbed integration; `brute_pdx` and wide-throughput retired from
        all experiment surfaces; e03 rebuilt with clean arms and re-run on
        scifact (§6.1 of the Stage 3b doc has the numbers).
- [x] e03 scifact finding recorded: exact-safe BOND does not beat the fused
      dense baseline (16.4–17.1 vs 13.1 ms/q all-cores; <2% docs pruned),
      consistent with e02 survival — the honest RQ3 first data point.
- [x] Fused-kernel overhead revision (2026-07-03, stage3b §6.1.1, after
      external review): sumsq fused into the tile-0 pass; final segment folds
      the per-doc max in registers (no Pt spill/rescan, no wasted sumsq);
      next-panel software prefetch (the old separate sumsq pass was an
      accidental prefetcher for permuted orders — removing it without the
      explicit prefetch REGRESSED bond/pca). BOND 1T overhead vs dense fell
      from ~54–78% to ~15–49%; at all cores natural/pca BOND sits at the
      dense DRAM floor. Scores bit-identical; all gates green; e03 scifact
      re-run on the revised kernel (results JSON refreshed). RQ3 verdict
      unchanged — now attributable to bound math alone.
- [x] R2 instrument (2026-07-03): NumPy checkpoint-accounting simulator
      (`src/bondmaxsim/oracle/checkpoint_sim.py`), fixed-tau policies,
      validated against fused kernel stats[1] across checkpoint sets
      (`tests/test_checkpoint_sim.py`).
- [x] R3 instrument (2026-07-03): checkpoint set parameterized through the
      kernel ABI (`checkpoints`/`n_checkpoints`, NULL = {32,64}), bindings,
      RunConfig.checkpoints, exactness gate for arbitrary C
      (`test_bond_custom_checkpoints_exact`); e08 driver written
      (`experiments/stage3_mechanism/e08_checkpoint_ablation.py`).
- [x] R5/R6 driver upgrades (2026-07-03): e05 gained the fused wall-clock
      shrink frontier vs the dense baseline (gate G2 surface); e06 gained a
      fused confirmation arm for the winning policy; e04 gained the 1T lens +
      fused prune stats; e07's prep/kernel decomposition fixed (Runner times
      kernel-only; total = kernel + prep).
- [x] e01–e06 runs on the upgraded drivers, committed 2026-07-03 ("Add
      Stage 3 e01-e06 results across all datasets"): e01/e02 extended to
      arguana + scidocs; e03/e04/e06 on all four datasets; e05 on
      scifact + nfcorpus. Findings recorded under R4/R5/R6 below and in
      RQ1/RQ3/RQ4 above.
- [x] e07 run (2026-07-03) on scifact/nfcorpus/arguana (scidocs deferred —
      memory issue; see end of queue): per-query reorder cost is negligible
      (reorder_fraction ≈ 0.1% of total; prep is µs-scale vs ms-scale
      kernels); one-time corpus prep is sub-second on all three (packing
      0.12–0.25 s, PCA fit ~1 ms, rotated packing 0.28–0.51 s). The order
      penalty lives IN the kernel: bond order runs 5–25% slower than natural
      at equal (non-)pruning — the permuted-access cost from stage3b §6.1.1.
- [x] e08 run (2026-07-03) on all 4 datasets, 16 arms each, recall 1.0
      everywhere. Headline: LATE checkpoints win — C={112} (or supersets)
      prunes 88–98% of docs and beats the all-cores dense baseline on
      arguana/scidocs/nfcorpus, ties scifact; the default {32,64} prunes
      <2% everywhere. Details in RQ3 above; follow-ups in R12.
- [x] e09 run (2026-07-03) on all 4 datasets, recall 1.0 everywhere (after
      the tie-aware fix below). At C={32,64} NEITHER bound prunes
      meaningfully (cheap 0.00% everywhere; tight ≤5.3%, bond order only)
      and both lose to dense — the bond2002 doc's third outcome
      ("completeness argument") at those checkpoints. Cheap is marginally
      faster where nothing prunes (its bookkeeping is cheaper); tight wins
      only where it prunes (nfcorpus bond order). Decision deferred to R12 —
      RESOLVED in R12a (2026-07-04): at e08's winning late checkpoints cheap
      prunes 0% everywhere, so **adopt TIGHT** (see R12(a) below).
- [x] Stage 4 / R8 instruments + runs (2026-07-09): baselines implemented
      for real — `faiss_ivf` (token IVF + exact rerank at fixed budget),
      `plaid` (PyLate FastPlaid), `pdx_ivf` (the Mikel-branch flat PDX-IVF
      via the `pdxearch` build; `extern/patches/README.md` has the Fedora
      build note) — with gate tests in `tests/test_stage4_baselines.py`;
      `candidate_seed_threshold` policy (Stage 1 §4.4 option b);
      IVF-partitioned fused scan (`src/bondmaxsim/partitioned_scan.py`);
      drivers e01/e02/e03 run on all four datasets, MT + 1T, interleaved
      (results/json/stage4_integration_*). Findings under R8 below;
      fairness controls + Mikel-chart dissection in
      `docs/stage4_comparison_methodology.md`.
- [x] Tie-aware exact-agreement fix (2026-07-03): nfcorpus query 181 has an
      EXACT score tie (7.0799808556, identical in float64) at rank 10; the
      fused kernel picked the other tied doc and tripped the shrink=1 gate
      (recall 0.9995). `run_fused_bond_mode`/`run_fused_brute_mode` now pass
      oracle scores to `exact_agreement` (boundary ties accepted), as the
      wide-block accounting path already did. 61 gate tests green.

### Required next (R-items, in EXECUTION order)

Items are listed in the order they should be executed. R-numbers are
assignment order, NOT execution order — they stay stable because other
documents reference them (`docs/bond2002_bound_cost_analysis.md` §6 →
R12a; RQ1/RQ3/G1 above → R12b/c). Completed items first, then the queue.

Completed:

- [x] **R1 — Stage 1 hygiene**: none outstanding (§10 addendum written
      2026-07-03). Re-audit only if the fused kernel's policy changes shape
      (e.g. panel-level pruning inside documents).
- [x] **R2 — instrument alignment**: DONE 2026-07-03 —
      `src/bondmaxsim/oracle/checkpoint_sim.py` (exact partial scores per
      checkpoint via incremental matmul, fixed-tau policies, both padded and
      unpadded cells conventions), validated against the fused kernel's
      stats[1] across three checkpoint sets in `tests/test_checkpoint_sim.py`.
      self_bound (order-dependent rising τ) deliberately out of scope.
- [x] **R3 — e08 checkpoint ablation**: DONE 2026-07-03 (all 4 datasets, 8
      sets ⊆ {32, 48, 64, 96, 112} × {natural, bond}, recall 1.0 on every
      arm). The expectation was WRONG in an informative way: the "≤16%
      ceiling" was a scifact cells number, and scifact indeed only ties
      dense — but on arguana/scidocs/nfcorpus late checkpoints
      (C={96}/{112}) prune 88–98% of docs and beat dense (arguana −19% MT /
      −29% 1T). RQ3 verdict recorded above; follow-ups split into R12.
- [x] **R4 — e03/e04 on remaining datasets**: DONE 2026-07-03 (committed as
      "Add Stage 3 e01-e06 results across all datasets", post-driver-
      upgrade; that commit also extended e01/e02 to arguana + scidocs, so
      the RQ1 evidence now covers all four datasets). e03 all 4: exact-safe
      BOND at the default C={32,64} loses to or at best ties dense_fused
      everywhere (dense MT 36.6 / 10.5 / 57.7 / 13.9 vs best BOND arm 39.9 /
      10.7 / 60.3 / 13.6 ms/q on arguana/nfcorpus/scidocs/scifact) while the
      ACCOUNTING envelope prunes 99.7–99.96% of docs — envelope vs captured,
      as e02 predicted. e04 all 4: accounting cells scanned FALLS with
      corpus size (scidocs 99.7% at 250 docs → 83.9% at 25.7k) — the R7
      scale question is live. The e07-scidocs remainder is split out to the
      end of the queue (memory issue; run later).
- [x] **R6 — e06 threshold policies**: DONE 2026-07-03, all 4 datasets
      (same commit). Oracle policy wins cells% (83.9–91.9% scanned);
      realistic policies land close behind it (self_bound 88.4–96.0, seed
      87.3–95.3) — the policy choice moves cells only a few points, and
      even the oracle leaves ≤16% on the table at C={32,64}. The fused
      confirmation arm (oracle policy, natural order) prunes ~0% and shows
      no wall-clock win, consistent with e03/e08-at-default.
- [x] **R11 — e09 bound-tightness ablation**: DONE 2026-07-03 (all 4
      datasets, tight vs cheap × {natural, bond, pca}, oracle policy,
      recall 1.0 everywhere). Outcome at the default C={32,64}: the third
      bullet of `docs/bond2002_bound_cost_analysis.md` §6 — BOTH bounds
      lose to dense because neither prunes there (cheap 0.00% everywhere,
      tight ≤5.3% under bond order only). Cheap is marginally faster when
      nothing prunes; tight wins only where its pruning fires. So e07's
      checkpoint overhead is mostly MaxSim's price, not E_v's — but the
      adopt-a-default decision is deferred to R12, because e08 moved the
      interesting regime to late checkpoints that e09 did not test.
- [x] Stage 0 loose end: verify BOND SIGMOD-2002 bibliographic details —
      DONE 2026-07-03: A. P. de Vries, N. Mamoulis, N. Nes, M. Kersten,
      "Efficient k-NN Search on Vertically Decomposed Data", ACM SIGMOD
      2002 (June 4-6, Madison, WI), pp. 322-333 (paper read; lessons in
      `docs/bond2002_bound_cost_analysis.md`).

Queue (execute top to bottom):

- [x] **R12 — late-checkpoint follow-ups (NEW, from the e08 finding)**. DONE
      2026-07-04 — all three deliverables complete: (a) tight bound adopted,
      (b) cells-vs-wall gap resolved as a measurement artifact, (c) corrected
      G1 margins on all 4 datasets. The one carry-over — folding the winning
      config (TIGHT, C={112}) into e05 — is R5's experiment and is already
      listed as an R5 dependency below, so R12 itself is closed.
      **(a) — bound default, DONE 2026-07-04** (e09 rerun, overnight log
      `results/logs/r12a_e09_20260704_004845.log`, 4957 s, all 4 datasets,
      24 arms each, recall 1.0). e09 now sweeps `{32,64}` (early control) +
      `{112}`, `{64,112}`, `{32,64,96,112}` (e08 winners) ×
      {natural,bond,pca} × {tight,cheap}, best-of-10, all cores. **Verdict:
      adopt the TIGHT bound as the doc-level default.** The CHEAP query-only
      bound (`Σ_i max_j P_ij + Σ_i resq_i`, resd→1) prunes 0.00% at EVERY late
      checkpoint on EVERY dataset — too loose to fire even at dim 112 — so it
      only pays checkpoint overhead. Tight prunes 87–98% and is faster in
      10/12 (natural-order) late-checkpoint cases; the two exceptions are the
      single C={112} on nfcorpus/scidocs where tight is marginally slower, but
      at C={64,112} and {32,64,96,112} tight wins on all four datasets. This
      REVERSES the BOND-2002 lesson (their cheap H_q beat the tight bounds):
      in MaxSim the cheap bound does not prune at all, so its "cheaper
      bookkeeping" buys nothing. bond2002 §6 criteria → tight.
      **(b) — cells-vs-wall gap, DONE 2026-07-04 (RESOLVED as a measurement
      artifact).** The premise ("wall-clock savings EXCEED the cells
      prediction; the pruned final segment is disproportionately expensive")
      is WRONG. The −29%/−12% gap is arguana-only (scifact/nfcorpus/scidocs
      save LESS wall-clock than cells predict, amp 0.2–0.3× at 1T), and it is
      driven by e08/e09 timing the dense baseline STANDALONE (once, in
      isolation) rather than interleaved with the arms. Interleaved back-to-
      back probes (`experiments/stage3_mechanism/r12b_interleaved_baseline_probe.py`,
      natural order, all/1 thread):
        - arguana 1T zero-prune (C={32}, 0% pruned): standalone offset +19%
          COLLAPSES to +0.2%; the standalone dense 179.6 ms/q was throttled
          (interleaved 149).
        - arguana MT C={112}: e08 standalone −19.6% corrects to **+8.7%**
          (dense 35.28 → 32.20), matching the e09 run's independent +8.0%
          (dense 39.77 → tight 36.58) — with a −4.3% zero-prune checkpoint
          OVERHEAD. C={112} arguana MT ranged 29–37 ms/q across three runs:
          the standalone across-window spread is the whole "amplification."
      Conclusion: wall-clock tracks the ~−12% cells prediction once the
      baseline is fair; there is NO disproportionately-expensive final
      segment. Methodology fix for all future wall-clock claims: interleave
      the dense baseline with the arms (round-robin, best-of-N per arm), never
      time it standalone. This directly downgrades G1 (see gate below).
      **(c) — corrected G1 margins, DONE 2026-07-04** (e05-fold carried by R5).
      Driver `experiments/stage3_mechanism/r12c_interleaved_exact_safe.py`
      (tight bound, natural order, same 50-query seed-42 subsample as e08,
      interleaved dense, best-of-8; JSON
      `results/json/stage3_mechanism_r12c_interleaved_exact_safe_{mt,1t}.json`).
      Corrected margin vs dense (best late set), interleaved [e08 standalone] (cells):

      | dataset | m | ALL-CORES (G1) | 1T |
      |---|---|---|---|
      | scifact  | 21 | +2.3% {64,112} [−0.3%] | +5.8% {112} [+3.8%] |
      | nfcorpus | 12 | +0.9% {64,112} [+2.8%] | +3.3% {112} [+2.1%] |
      | arguana  | 48 | **+8.7%** {112} [+19.6%] | **+14.1%** {112} [+29.5%] |
      | scidocs  | 17 | +0.9% {64,112} [+5.5%] | +5.6% {112} [+4.0%] |

      Zero-prune C={32} overhead (interleaved): MT −2 to −6%, 1T −6 to −10%
      (arguana 1T ≈ 0). Every dataset's cells-saved ≈ +12%; the corrected wall
      margin ≈ cells − checkpoint-overhead, NO amplification. **Verdict: at all
      cores only arguana (large m=48 → high arithmetic intensity, low relative
      overhead) clears a real margin (+8.7%); the other three are +0.9–2.3%
      ties. G1 is NOT met at the honest all-cores baseline** (needs ≥2 datasets;
      arguana alone). At 1T all four are positive (+3.3…+14.1% at C={112}) but
      1T is not the decision baseline (RQ2 dense is all-cores). MT margins are
      smaller than 1T because at all cores the kernels are more memory-bound, so
      the arithmetic cells saved translate poorly to wall-clock. **Carry-over
      to R5:** fold the winning config (TIGHT, C={112}) into e05 — tracked as
      an R5 dependency, not a reopened R12 item.
- [~] **R5 — e05 approximate frontier, remainder**: DEFERRED TO FUTURE WORK
      (2026-07-08, RQ4 rescoped out). Preliminary record retained: ran on
      scifact + nfcorpus at the default C={32,64} (fused wall-clock frontier +
      accounting frontier, self_bound policy). NO G2 point — wall-clock wins
      only below the recall bar (shrink=0.8: 11.7 vs 15.6 ms/q scifact / 8.6 vs
      10.8 nfcorpus at recall 0.96–0.97); at shrink ≥0.9 recall is 1.0 but
      pruning collapses (≤1.5% docs) and the win vanishes. Future work: the
      late-checkpoint operating point (R12a-winning bound), where the exact-safe
      arm already prunes 88–98% and shrink<1 starts from a winning position.
- [~] **R7 — scale check**: DEFERRED TO FUTURE WORK (2026-07-08; does not fit
      this machine). One 100k–1M doc corpus (CoRECT pools, `[retrieval]` extra)
      exported to the packed format; re-run e03/e05 there. Safe to defer: the
      mechanism verdict rests on the bound looseness of L2-normalized d=128
      embeddings, which is corpus-size-independent (e04 gives the small-scale
      trend); the DRAM-floor argument is expected to hold at scale.
- [x] **R8 — Stage 4**: DONE 2026-07-09 (e01/e02/e03, all four datasets,
      MT + 1T, interleaved dense baseline, only the winning mechanism arm
      promoted — TIGHT bound, natural order, C={112}, shrink=1).
      **(a) e01 fixed-budget method separation:** at matched budgets
      B ∈ {100,500,1000,5000} every token-level candidate pipeline loses to
      the exhaustive fused dense scan on scifact/nfcorpus/arguana (best
      pipeline arm −12% to −160% vs dense, MT) — per-query candidate
      generation costs more than scanning the whole corpus with the RQ2
      kernel. Only on scidocs (25.7k docs) do faiss@100 (+56%) and pdx@100
      (+62%) beat dense, at recall 0.92 — the crossover exists and is a
      scale effect (R7 stays live as future work). The seeded exact-safe
      BOND arm beats dense kernel-only (arguana 48.1 vs 51.7 ms/q MT) but
      goes negative once its seed cost is charged (−1.8% to −16.2% total).
      Under the same controls the Mikel-branch "PDX-IVF ≫ PLAID ≫ exact"
      chart collapses: pdx ≈ faiss ≈ plaid, all above dense (budget/timer/
      stack confounds — methodology doc §3).
      **(b) e02 seeded-tau recovery:** `ivf_seed_strong` recovers ~99% of
      the ORACLE pruning on every dataset (e.g. scifact 87.8% vs 88.2% docs
      pruned; kernel margin +5.9% ≈ oracle +5.8%) — the R12c oracle-tau
      margins ARE reachable by a realistic seed — but the seed pipeline
      costs 7–23 ms/q against kernel savings of ~1–4 ms/q, so every seeded
      arm is net-negative at this scale; `self_bound` (free, 39–77%
      recovery) remains the honest production policy. All arms recall 1.0.
      **(c) e03 partitioned fused scan:** a real approximate frontier —
      scidocs 4.0x at recall 0.92, arguana 5.2x at 0.87, ~2.1x at 0.95–0.96
      on both — but nfcorpus/scifact cross below 1x by recall ~0.88–0.9;
      the full-probe exact control is 2–5x SLOWER than the monolithic scan
      (per-partition slicing overhead), and the exact-safe partition bound
      (UB_p = Σ_i⟨q_i,c_p⟩ + m·R_p) prunes 0% of partitions under oracle
      tau. The arm lives strictly on the approximate frontier
      (Convention 4); the brute-vs-bond scanner ablation is a wash (±5%).
- [ ] **R9 — Stage 5 (RESCOPED BACK IN 2026-07-09; NEXT UP)**: was deferred
      2026-07-08; brought back as the paper's FINAL SECTION — take the best
      methods from Stages 3–4 and show how they perform in the proper IR
      evaluation framework (Stage 5 spec above: qrels metrics + CoRECT RC,
      fairness controls, build cost reported separately). Plan:
      1. Instruments: `src/bondmaxsim/eval/` — qrels metric utilities
         (nDCG@10, recall@100, MRR@10, recall_vs_exact@10) + the CoRECT
         adapter (`extern/CoRECT` pinned @ `fedf8bb2`; needs a
         ColBERT/MaxSim wrapper for the RC metrics).
      2. Arms — the best method per family from the R8 verdict, all on the
         same interleaved/one-stack controls as e01:
         `dense_fused` (the exact production path), exact-safe BOND
         (TIGHT, natural, C={112}, self_bound — the free policy; recall 1.0
         by construction, included to show exactness costs nothing in IR
         quality), the partitioned fused scan at 2–3 operating points from
         the e03 frontier (e.g. recall≈0.9 and ≈0.95), and tuned
         `faiss_ivf@B` / `plaid@B` as external references at matched
         budgets.
      3. Driver: `experiments/stage5_corect/e01_ir_evaluation.py` on all
         four BEIR datasets, MT + 1T, repeated runs with CIs; report the
         quality-latency table/frontier (IR metrics vs ms/q and QPS,
         memory + index time separate).
      4. Expected story: the exact arms inherit exact MaxSim's IR quality
         at the lowest latency (the R8 verdict restated in IR terms); the
         approximate arms show what the qrels metrics hide vs surface
         relative to recall_vs_exact (probing loses true top-k docs —
         does nDCG@10 care?).
- [ ] **R10 — paper (the remaining critical path)**: write up per the Final
      Paper Structure; every claim through the Validation Checklist. Stage 4
      is now FROZEN (2026-07-09), so the stubbed results/discussion sections
      of `report/` can be written. Concrete steps:
      1. Results: RQ1 mechanism numbers (survival curves, cells ceiling),
         RQ2 kernel (3.2x over 12-thread OpenBLAS — standalone contribution
         independent of the RQ3 verdict), RQ3 corrected R12c table (G1 NOT
         met; single-dataset arguana +8.7% win, interleaved), R8 Stage 4
         system verdict (dense scan beats candidate pipelines at matched
         budgets; seeding reachable-but-not-monetizable; partitioned
         approximate frontier), and the Stage 5 CoRECT evaluation (R9) as
         the FINAL SECTION — the best methods in the proper IR framework.
         R10's final section blocks on R9; the rest can be written now.
      2. Discussion: why the bound math (not engineering) caps exact-safe
         BOND for MaxSim; the interleaved-baseline methodology lesson
         (R12b); the scidocs crossover as the scale boundary of the verdict.
      3. Figures: e01 budget-axis separation + e03 latency-recall frontier
         (already in results/figures/stage4_integration/), plus the existing
         Stage 3 e08/R12c evidence.
      4. Future work section: R5 late-checkpoint shrink frontier, R7 scale
         (100k–1M docs — where e01-scidocs says candidate generation starts
         to pay).
      5. Validation Checklist pass over every headline claim before
         submission.
- [ ] **e07 on scidocs (deferred to the back)**: the only R4 remainder.
      Hits a memory issue on the 25.7k-doc corpus — run later once
      resolved; until then the e07 record (reorder cost negligible) rests
      on scifact/nfcorpus/arguana.

Optional (not in the queue; triggered by outcomes above):

- [ ] Optional (only after a positive RQ3): PDX-in-DuckDB integration
      as future work.
- [ ] Optional — trigger condition MET by e08 (2026-07-03): bond order DID
      prune more than natural at C={112} on every dataset (e.g. arguana
      98.1% vs 97.1%) yet lost wall-clock everywhere — so pack-time static
      dimension order is now a live candidate (after R12, if its margin
      would matter). The bond order's 1T
      penalty is the permuted access pattern itself (one scattered 64 B line
      per dim; PDX-sigmod pays the same via its `indices_dimensions`
      translation index — our `order[t]` is the faithful equivalent, incl.
      the DISTANCE_TO_MEANS_IMPROVED per-partition physical sort). A
      corpus-global importance order applied physically at pack time (as the
      pca arm already does with its rotation) would give sequential access +
      early energy concentration, at the cost of query-independence.

### Decision gates (explicit criteria)

- **G1 (exact-safe, RQ3)**: an arm with recall 1.0 beating the fused dense
  all-cores baseline by a repeatable margin on ≥2 datasets. **Status: NOT MET
  at the all-cores baseline (R12c, 2026-07-04, corrected).** The e08 evidence
  (arguana −19% MT, scidocs −8%, nfcorpus −3%, scifact tie) came from a
  STANDALONE dense baseline; R12b/R12c re-measured with the baseline
  INTERLEAVED (tight bound, natural order, same 50-query seed-42 subsample,
  best-of-8). Corrected all-cores margins at the best late checkpoint:
  **arguana +8.7%** (was +19.6%), scidocs +0.9% (was +5.5%), scifact +2.3%,
  nfcorpus +0.9% — i.e. only ONE dataset (arguana, the large-m=48, high
  arithmetic-intensity case) clears a real margin; the other three are
  within ±2% (ties). The checkpoint kernel carries a −2…−6% zero-prune
  overhead that, minus the ~+12% cells saving, leaves ≈0 on the datasets that
  are memory-bound at all cores. So the honest verdict is a SINGLE-dataset
  exact-safe win, not the ≥2-dataset repeatable margin G1 requires. (At 1T all
  four are positive, +3.3…+14.1% at C={112}, but 1T is not the honest
  baseline — RQ2 dense is all-cores.) At the default {32,64} checkpoints G1
  remains negative everywhere. Report this as the negative/single-dataset
  result WITH the RQ1 explanation and the RQ2 baseline contribution — do not
  soften the baseline to manufacture a win, and (the R12b lesson) do not let a
  throttled standalone baseline manufacture one either.
- **G2 (approximate, RQ4) — OUT OF SCOPE (rescoped 2026-07-08).** Deferred to
  future work with RQ4. Preliminary evidence retained: `shrink < 1` frontier
  dominating the dense baseline (lower latency, recall ≥ 0.99) was NOT found
  on e05 (scifact + nfcorpus at C={32,64}, 2026-07-03) — best sub-baseline
  latencies sit at recall 0.96–0.97, and recall recovers to 1.0 only where
  pruning (and the win) has collapsed. The late-checkpoint operating point is
  future work, not a gate for this paper.
