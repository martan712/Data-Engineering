# Controlled Benchmark Protocol

This protocol is mandatory for final wall-clock comparisons. Existing results
that do not satisfy it may still support correctness, candidate coverage, or
mechanism analysis, but must not be presented as controlled speedups.

## Comparison contract

All methods in one timing comparison must use:

- the same machine, operating system, CPU governor/power setting, dataset,
  embeddings, query order, and process-level thread configuration;
- the same final output contract: ranked document IDs for each query;
- the same exact MaxSim definition and numerical precision;
- the same warm-up and repetition schedule;
- matched rerank budgets where the methods expose comparable candidate stages.

Windows-to-WSL or NumPy-to-compiled ratios are not final speedup measurements.

## Baselines

The primary exact baseline is a compiled fused MaxSim kernel. The NumPy
per-document implementation remains a correctness oracle and readable reference,
not the performance baseline.

Approximate methods are evaluated against the compiled exact ranking using:

- exact top-k set recall at k = 1, 3, 5, and 10;
- exact top-1 match and all-of-top-k recovery;
- qrels Recall@k and MRR@10;
- online latency and throughput.

### Controlled PLAID addition

PLAID is an external ColBERT-specific baseline, not a PDX component. Its
historical result is excluded because it used up to 8,192 full scores while the
IVF arms commonly reranked 50 documents. The replacement comparison uses the
same Linux process, CPU affinity, resident query embeddings, output contract,
warm-up schedule, and five interleaved measured repetitions as exact, FAISS-IVF,
and PDX-IVF.

Parameter selection is isolated from the held-out release queries:

- build one CPU fast-plaid index with `nbits=4`, Triton disabled, and normalized
  document token vectors, then record a content hash of the resulting index;
- on the first 10 SciFact validation queries, sweep `n_ivf_probe` in
  `{8, 16, 32, 64, 128}` and `n_full_scores` in `{50, 100, 200, 400}`;
- retain the fastest validation configuration reaching exact recall@10 of at
  least 0.85, the fastest reaching at least 0.95, and the highest-recall
  configuration (deduplicated, with deterministic ties);
- if no configuration reaches 0.95, hold `n_ivf_probe` at the fastest value
  among the maximum-recall arms and run one validation-only expansion with
  `n_full_scores` in `{800, 1600, 3200, number of documents}`. This expansion
  is permitted once and must still be frozen before viewing test results;
- freeze the selected configurations and evaluate them on queries 10--49 of
  both SciFact and NFCorpus.

`n_full_scores` is PLAID's configured full-scoring budget. PyLate/fast-plaid
does not expose the realized candidate count through the API used here, so it
must not be reported as measured work or treated as exactly equivalent to the
realized IVF rerank count. PLAID is therefore compared through its complete
online latency, exact-ranking recovery, qrels metrics, configured budget, and
offline index size/build time. Quality-matched points and frontiers are valid;
an unqualified same-work claim is not.

## Timing boundaries

Report index construction separately as offline cost. Query encoding is also a
separate stage unless every method includes the same encoder invocation.

Online end-to-end latency starts with query token embeddings already resident in
memory and ends when final ranked document IDs are available. It includes:

1. ANN or BOND index search;
2. token-hit to document mapping;
3. token aggregation and candidate feature construction;
4. top-C candidate selection;
5. exact MaxSim reranking;
6. final top-k selection and ID mapping.

Each stage may also have its own component timer, but component time must not be
labeled end-to-end speedup. Synchronize any asynchronous backend before stopping
a timer.

## Budgets and operating points

For two-stage methods, record at least:

- token-hit budget `L` per query token;
- index probe/search parameters;
- unique candidate-pool size;
- selected exact-rerank budget `C`;
- actual number of reranked documents;
- total retrieved token hits.

Compare methods with the same `C` where possible. Also report quality-latency
curves over several budgets. A method's default configuration may be shown as a
native operating point, but not as a matched-budget comparison.

PLAID's `n_full_scores` must be explicit. Its default must not be compared with
`C=50` IVF pipelines as if they perform equivalent reranking work.

## Repetition and scheduling

- Run at least two untimed warm-up queries or one complete untimed warm-up pass.
- Run at least five measured repetitions for small experiments and three for
  longer experiments.
- Interleave method order per repetition, or rotate it deterministically, to
  reduce thermal and cache-order bias.
- Report median, interquartile range, and p95 where enough samples exist.
- Preserve per-query latency samples in the result artifact.
- Fix and record random seeds. Reuse trained indexes across timed query runs.

## Threading and hardware

Record CPU model, physical/logical core counts, RAM, OS/kernel, Python, compiler,
BLAS/OpenMP libraries, package versions, and Git commit. Explicitly set thread
counts for OpenMP, BLAS, FAISS, and each custom kernel. Do not compare a
single-threaded method with an all-core method without labeling the difference.

## Result artifact requirements

Every final JSON result must include:

- schema version and UTC timestamp;
- Git commit and dirty-worktree flag;
- dataset name, document/query counts, token counts, and embedding dimension;
- input file hashes or an equivalent immutable dataset identifier;
- complete method parameters and random seeds;
- machine/software metadata and thread settings;
- warm-up count, measured repetition count, and raw timing samples;
- stage timings and online end-to-end timings;
- exact-ranking recovery and qrels metrics;
- errors, skipped configurations, and the reason for each skip.

Figures and report tables must be generated from these artifacts, without
hard-coded headline values.

## Status of current evidence

| Evidence | Current use |
| --- | --- |
| PyLate export and packed variable-length embeddings | Reusable input pipeline |
| PDX-BOND smoke test matching brute-force L2 | Correctness evidence |
| Flat token-vector BOND slower than NumPy at small scale | Exploratory negative result; replaced by compiled exact-safe BOND release runs |
| NumPy MaxSim BOND work-ratio/oracle study | Mechanism evidence; not kernel wall-clock evidence |
| IVF candidate-pool and selector-recall measurements | Clean controlled quality-latency evidence under `results/final/` |
| Reported 30x/100x IVF speedups | Withdrawn; clean same-stack measurements replace them |
| Current PLAID versus IVF wall-clock bars | Withdraw pending matched budgets and timing boundaries |

## Claim language

Use `candidate reduction`, `work ratio`, `component latency`, or `ranking
recovery` when that is what was measured. Use `speedup` only for the ratio of two
complete, comparable timing regions collected under this protocol.

No wall-clock result is accepted if the top-k set differs from exhaustive
MaxSim or if an order difference exceeds the declared numerical tie tolerance.
