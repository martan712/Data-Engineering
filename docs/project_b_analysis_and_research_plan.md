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

## Status And Checklist

Last updated 2026-07-02. This section is the working to-do list for future
sessions/agents. Keep it and the status table in `docs/project_structure.md`
in sync when a stage advances. Conventions (uv toolchain, thin drivers, shared
schema, exact/approx separation) are in `docs/project_structure.md` — read it
before adding files.

### Done

- [x] Repository restructured as a self-contained research branch
      (`final-research-implementation`): `docs/`, `src/bondmaxsim/`, `cpp/`,
      `experiments/`, `results/`, `archive/`, pinned submodules in `extern/`
      (PDX @ `93531b9`, PDX-sigmod @ `fdc62f2`, CoRECT @ `fedf8bb2`).
- [x] Toolchain standardized on uv (`./setup.sh`, `uv run pytest`); packaging
      via `pyproject.toml`, NumPy-only core with `[retrieval]`/`[faiss]`/`[dev]`
      extras.
- [x] Stage 0: `docs/stage0_references_and_baselines.md` — terminology table,
      source pins, baseline matrix; external ZEN5 BOND/BSA/ADSampling results
      preserved under `results/external_baselines/`.
- [x] Stage 1: `docs/stage1_bond_maxsim_formalization.md` — `shrink = 1`
      exact-safety proof (unit-norm precondition, token-pruning survival
      invariant, set-equality top-k), `shrink < 1` separation, per-document
      "Option B" demoted to exact-safe oracle, wide-token-block PDX-BOND named
      as the Stage 2 deliverable, seeded threshold promoted to first-class.
- [x] Stage 2 complete: wide-block MaxSim BOND kernel implemented in
      `cpp/wide_block_maxsim_bond/` (C++, Makefile, `.so`); ctypes bindings in
      `src/bondmaxsim/kernels/wide_block.py` (accounting + throughput entry
      points); exact-agreement gate passing on all four datasets (scifact,
      nfcorpus, arguana, scidocs); package scaffold with `config`, `schema`
      (`ResultRecord`), `data`, `oracle`, `ordering`, `kernels`, `testbed`,
      `threshold` all implemented.
- [x] Per-document oracle kernel ported and building
      (`cpp/per_document_oracle/`, exp-09 accounting + exp-10 throughput).
- [x] Debug-scale data exported: `data/embeddings/{scifact,nfcorpus,arguana,scidocs}.npz`
      (GTE-ModernColBERT-v1, D=128, unit-norm verified, 200 queries each).
- [x] Test suite green: 68 tests including the two Stage 2 blocking checks
      (unit-norm guard; `shrink = 1` exact-agreement gate on the oracle kernel).
- [x] Schema extensions: `shrink` and `tokens_pruned_pct` added to
      `ResultRecord`; the testbed runner surfaces the accounting kernel's
      `tokens_pruned` stat (stats[2]) as `tokens_pruned_pct`.
- [x] Threshold policies implemented (`src/bondmaxsim/threshold/policies.py`:
      self_bound / oracle / seed) with safety tests
      (`tests/test_threshold_policies.py`).
- [x] Stage 2 smoke drivers (`experiments/stage2_testbed/`): s01 normalization
      guard, s02 exact-agreement gate (all four datasets), s03 two-mode smoke;
      run with `ResultRecord` JSON committed under `results/json/`.
- [x] Stage 3 e01 (bound slack): `bondmaxsim.oracle.bound_trajectory` +
      `experiments/stage3_mechanism/e01_bound_slack.py` + tests committed;
      run on scifact and nfcorpus (50 queries, natural/bond/pca) with JSON +
      figures committed under `results/`; e02 spec extended to two-level
      (document and token) survival curves in
      `experiments/stage3_mechanism/README.md`.
- [x] Stage 3 e02 (pruning-rate survival curves): two-level document and token
      survival curves from the wide-block kernel across all threshold policies
      (self_bound / oracle / seed) and dimension orders (natural / bond / pca);
      run on scifact and nfcorpus; JSON + figures committed under `results/`.
- [x] Stage 3 e03 first run (scifact) + wall-clock baseline forensics: the
      8.2x `brute_pdx` vs `brute_numpy` gap decomposed into 4.06x OpenBLAS
      threading + 2.0x runtime-`m` inner-loop codegen; BOND throughput slower
      than its own dense baseline (bound bookkeeping > 16.3% cells saved).
      Full analysis, theory, and kernel redesign in
      `docs/stage3b_fused_panel_maxsim_kernel.md`.

### In Progress / Next

- [ ] **Stage 3b — fused panel MaxSim kernel** (unplanned interlude; design
      and rationale in `docs/stage3b_fused_panel_maxsim_kernel.md`). The
      wall-clock instruments are rebuilt before e03–e07 are run, because the
      current dense baseline comparison is not thread- or codegen-fair and
      the decision gate would be evaluated against the wrong baseline.
      Prototype evidence: fused register-tiled kernel on panel-major PDX
      layout = 50.4 ms/q single-threaded vs 147.7 ms single-thread OpenBLAS
      (2.9x) on scifact. Work items (doc §6):
      - [x] K1: `pack_corpus_panels` (16-token panel-major, duplicate-last-token
            doc padding) + layout/padding-invariance tests (2026-07-02).
      - [x] K2: `cpp/fused_panel_maxsim/` brute kernel (register-tiled M∈{8,16,24}
            query tiles, fused per-doc max epilogue) + ctypes bindings +
            exact-agreement gate tests (synthetic incl. multi-tile m>24 and
            negative-max padding trap; recall 1.0 on scifact in e03).
      - [x] K3: OpenMP over groups; 1T/8T result invariance test green.
            Measured on scifact: 54.5 ms/q 1T → 15.8 ms/q all-cores
            (vs numpy 43.5 ms/q 12T, 121.8 ms/q pinned 1T) — 2.8x over
            12-thread OpenBLAS, at the predicted DRAM-floor scale.
      - [ ] K4: `fused_panel_maxsim_bond` with panel-granularity bounds at
            fetch-boundary checkpoints; shrink=1 exact-agreement gate re-run.
      - [x] K5 (partial): Runner `brute_force_mode` kinds pdx/numpy/fused with
            `n_threads` (threadpoolctl BLAS pinning for numpy); e03 refreshed
            on scifact with all five baselines (results committed).
            Remaining: `bond_fused` arm (needs K4) + e03 on nfcorpus /
            arguana / scidocs.
      The accounting kernel and e01/e02 results are unaffected and stay.
- [ ] **Stage 3 experiments e03–e07** per
      `experiments/stage3_mechanism/README.md`, run against the Stage 3b
      wall-clock instruments: dimension-order ablation (e03 refresh),
      exact-safe cells/latency sweep (e04), approximate recall frontier (e05),
      threshold-policy ablation (e06), cache/layout penalty (e07).
- [ ] **Stage 3 decision gate**: record explicitly in this document whether any
      exact-safe arm yields a repeatable wall-clock win over brute force — the
      gate baseline is the **multithreaded fused dense kernel** (strongest
      defensible baseline), not NumPy; record the regime finding
      (m≈21 MaxSim is GEMM-shaped; doc §2) alongside the outcome. If no win,
      pivot to the documented negative-result path.

### To Do (Stages 3–5)
- [ ] Scale dataset: obtain one 100k–1M document corpus (prefer CoRECT pools;
      requires the `[retrieval]` extra to embed) and export it to the packed
      token format.
- [ ] Qrels: export qrels for the four debug datasets and the scale corpus so
      nDCG@10 / recall@100 / MRR@10 can be computed (Stage 3 onward);
      implement `src/bondmaxsim/eval/qrels.py` (currently stubs).
- [ ] Stage 4: implement `src/bondmaxsim/baselines/` (faiss_ivf, pdx_ivf,
      plaid — all currently stubs) and the fixed-candidate-set comparison
      drivers in `experiments/stage4_integration/`.
- [ ] Stage 4: retune and rerun PLAID at appropriate scale (small CPU runs are
      off-design and non-decisive).
- [ ] Stage 5: implement the CoRECT adapter (`src/bondmaxsim/eval/corect.py`,
      currently a stub) with a ColBERT/MaxSim wrapper; run RC metrics and the
      matched-quality frontier under the fairness controls (one machine, one
      OS, fixed threads, repeated runs with confidence intervals).
- [ ] Stage 0 loose end: verify the BOND SIGMOD-2002 bibliographic details
      against the paper before final writing (flagged in the Stage 0 doc).
- [ ] Paper: write up following the Final Paper Structure section; every claim
      passes the Validation Checklist above.
- [ ] Optional, only after a positive Stage 3/4 result: assess the PDX-in-DuckDB
      integration path as future work (from the project description).
