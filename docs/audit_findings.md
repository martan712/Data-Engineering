# Branch and Evidence Audit

Audit date: 2026-07-13.

## Audited snapshots

- `Mikel`: `a3fb47aa5cab9acb137699cb90c0b63e84e1eb30`
- `final-research-implementation`: `ed01ea2`
- Divergence at audit time: 5 commits unique to `Mikel`, 85 commits unique to
  `final-research-implementation`.

The working branch for this repair is `Mikel`.

## Main finding

The existing `Mikel` pipeline contains useful data preparation, ColBERT
embedding, candidate-generation, reranking, and diagnostic work. Its published
wall-clock comparisons are not suitable as final speedup evidence because the
methods were measured with different timing boundaries, rerank budgets,
machines, implementations, and thread settings.

The final project will retain the reproducible architecture and quality results,
withdraw the historical headline speedups, and rerun a smaller comparison under
`docs/benchmark_protocol.md`.

## Methodology issues found

1. PDX candidate timing measured token search but excluded token-to-document
   aggregation, candidate selection, reranking, and final ranking.
2. FAISS `total_seconds` excluded aggregation, selection, and final ranking.
3. PLAID used up to 8192 full scores while the IVF paths commonly reranked 50
   documents.
4. The exact performance reference was a Python loop around one NumPy matrix
   multiplication per document rather than a compiled fused kernel.
5. Windows and WSL measurements appeared in the same latency figures; the same
   exact workload differed by several times between those environments.
6. Thread budgets were inconsistent or unrecorded. There were no repeated,
   interleaved runs or raw per-query timing distributions.
7. Configurations were selected using exact rankings or qrels from the same
   queries later reported as evaluation results.
8. Corpus subsets placed relevant documents before distractors, and IVF bucket
   counts changed with corpus size. The resulting scaling curve did not isolate
   corpus size.
9. `pool_coverage@10` sometimes meant all exact top-10 documents were present,
   while `agreement@10` meant mean set overlap. These cannot be described as the
   same coverage limit.
10. The NumPy BOND instrumentation reports idealized operation counts. Its
    reciprocal work ratio excludes bound-maintenance and control-flow cost and
    is not a measured kernel speedup.

## Evidence disposition

| Existing evidence | Decision |
| --- | --- |
| PyLate export and packed variable-length token embeddings | Keep |
| Exact NumPy MaxSim scores and rankings | Keep as correctness oracle |
| L2-normalized PDX `l2sq` equivalence to cosine ordering | Keep |
| Candidate-pool coverage and rerank-budget quality curves | Keep after artifact/schema verification |
| Flat PDX-BOND and batch/shared-scan prototype timings | Keep as exploratory engineering observations |
| BOND/PCA/oracle work ratios | Keep as bounded mechanism analysis, with narrower language |
| 30x/39x/32x/8x/7.4x speedup statements | Withdrawn; replaced by clean controlled measurements under `results/final/` |
| Historical PLAID versus IVF latency bars | Withdrawn; replaced by validation-selected same-process points with explicit configured budgets and post-hoc sensitivity labels |

Compact historical JSON files are preserved under `results/legacy/`. They are
not inputs to final speedup claims.

## Reproducibility gaps

At the audited `Mikel` commit, result JSON files were ignored and absent from
Git, four figures were tracked without their inputs, no tests existed, PDX was
checked out by mutable branch name, and result records lacked commit/input hashes
and complete environment metadata.

Initial repairs in the current worktree add:

- a controlled benchmark protocol;
- a staged final-project plan;
- tracked compact legacy result artifacts with explicit status;
- unit tests for MaxSim utilities and benchmark bookkeeping;
- shared warm-up, repetition, percentile, input-hash, and runtime metadata
  helpers.

## Authorship and reuse boundary

The 85 commits unique to `final-research-implementation` are authored by Martan
van der Straaten. This repair may use the branch's documented methodological
lessons and public algorithmic ideas, but must not silently present its code or
results as work from `Mikel`.

Any direct code reuse must retain Git authorship and be identified in the report,
or be independently reimplemented. Because the project is now being completed
separately, the team should confirm the permitted reuse and attribution with the
supervisor before importing substantial implementation from that branch.

## Completed remediation

The repair produced independently implemented compiled exact and exact-safe
BOND-MaxSim kernels, automated NumPy-oracle tests, complete online timing,
matched FAISS/PDX rerank budgets, SciFact and NFCorpus transfer runs, and
clean-commit release artifacts under `results/final/`. The final BOND comparison
uses float64 products and accumulation in both arms. The controlled PLAID
addition uses the same process and complete timing boundary, freezes two points
on validation, and reports corpus-sized score budgets only as post-hoc
sensitivity checks. Historical speedup claims remain withdrawn.
