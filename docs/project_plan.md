# Final Project Plan

## Objective

Produce a defensible and reproducible evaluation of whether BOND-style
dimension pruning can accelerate ColBERT MaxSim search, and compare it with
candidate-generation alternatives under one controlled benchmark protocol.

The final contribution does not depend on BOND being faster. A well-supported
negative result is valid if correctness, pruning behavior, and wall-clock cost
are measured against a strong exact implementation.

## Current status

- P0-P2: complete in the current worktree.
- P3: exact-safe compiled BOND, free-oracle, and PCA clean release complete.
- P4: matched-budget FAISS-IVF/PDX-IVF clean release complete; controlled
  validation-selected PLAID operating points and post-hoc sensitivity complete.
- P5: SciFact and NFCorpus clean controlled release complete.
- P6: repository report, final artifacts, figures, provenance, and tests are
  complete. Supervisor-specific submission packaging is external to this plan.

## Research questions

1. Can an exact-safe BOND-MaxSim kernel prune enough token-dimension work to
   beat an optimized exact MaxSim kernel?
2. If exact-safe BOND does not win, which property of ColBERT embeddings limits
   pruning?
3. At matched rerank budgets, how do PDX-IVF and FAISS-IVF trade exact ranking
   recovery for online latency, and where do validation-selected PLAID native
   operating points lie under the same timing boundary?
4. Which conclusions transfer from SciFact to a second small BEIR dataset?

## Work packages and acceptance gates

### P0 - Audit and freeze the evidence

- Inventory scripts, result files, figures, environments, and branch origins.
- Mark every existing number as `verified`, `exploratory`, or `invalid for
  comparison`.
- Preserve useful negative results, but remove unsupported speedup claims.

Gate: every reported number maps to a result artifact, command, configuration,
machine description, and Git commit.

### P1 - Reproducible project skeleton

- Add a single environment/setup path for the controlled Linux benchmark.
- Track compact result JSON files and metadata; keep large embeddings ignored.
- Add smoke tests for packing, exact MaxSim, ranking metrics, and candidate
  aggregation.
- Add a small fixture that runs without downloading a dataset.

Gate: a clean clone can run tests and the fixture from documented commands.

### P2 - Strong exact baseline

- Integrate or implement a compiled fused exact MaxSim kernel.
- Verify scores and top-k rankings against the NumPy reference.
- Record threading, compiler flags, warm-up, and repetition policy.

Gate: numerical agreement is within a documented tolerance and ranking
agreement is exact on all test fixtures.

### P3 - Exact-safe BOND-MaxSim mechanism study

- Run the compiled exact-safe BOND kernel and the fused exact kernel through the
  same runner.
- Report bound tightness, work ratio, pruning rate, and wall-clock time.
- Separate oracle diagnostics from implementable policies.
- Keep PCA/ordering ablations as mechanism analysis, not as speedup evidence
  unless their preprocessing and runtime are accounted for.

Gate: exact-safe modes reproduce the exact top-k and all timing comparisons use
the protocol in `docs/benchmark_protocol.md`.

### P4 - Controlled candidate-generation study

- Compare PDX-IVF and FAISS-IVF on the same embeddings and queries.
- Include PLAID only as a clearly labeled native operating point if its full
  score budget and timing boundary can be matched.
- Include token retrieval, token-to-document aggregation, candidate selection,
  exact reranking, and final top-k extraction in online latency.
- Sweep rerank budget rather than comparing one unmatched configuration.
- Report exact ranking recovery and qrels metrics alongside latency.

Gate: methods are compared as quality-latency curves with matched candidate
budgets or clearly labeled native operating points.

### P5 - Dataset evaluation

- Use SciFact as the primary reproducible benchmark.
- Use NFCorpus, or another agreed small BEIR subset, as the transfer check.
- Run CoRECT only if its setup and evaluation can be completed without weakening
  the controlled core experiment.

Gate: at least two datasets have complete exact and approximate result artifacts
from the same machine and software stack.

### P6 - Final report and release

- Replace the current README headline with verified conclusions.
- Produce a short methods/results report, a limitations section, and figures
  generated only from tracked result JSON files.
- Document shared code provenance and commit-level attribution.
- Run the full test suite and a clean reproduction smoke test.

Gate: all tables and figures regenerate from tracked artifacts and no claim
depends on a cross-machine wall-clock ratio.

## Priority order

1. Benchmark correctness and traceability.
2. Compiled exact and exact-safe BOND comparison.
3. Controlled candidate-generation curves.
4. Second dataset.
5. Report polish and optional CoRECT extension.

## Explicit non-goals

- No claim that standard IVF is a novel indexing method.
- No end-to-end speedup claim from component-only timing.
- No ratio between Windows and WSL measurements.
- No comparison that gives methods materially different rerank budgets without
  showing that difference.
- No large scale-up before the controlled small benchmark is stable.

## Release state

1. Core implementation and the PLAID comparison protocol were frozen before
   held-out execution at source commit `0a11fda`.
2. Seven clean-commit result artifacts are accepted by
   `results/final/manifest.json`: five held-out IVF/BOND artifacts and two
   explicitly post-hoc PLAID full-score sensitivity artifacts.
3. Figures 5-9 are generated only from accepted release artifacts.
4. Complete Windows/reference and WSL/native test suites are release gates.
5. Controlled CPU PLAID operating points are now included. Their configured
   work budgets, validation/test shift, and post-hoc sensitivity status are
   reported without claiming a general or end-to-end speedup.
6. Larger-scale evaluation remains future work, not a blocker for the current
   controlled result.
