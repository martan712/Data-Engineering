# Stage 4: Candidate Kernel Integration Experiments (R8)

Stage 4 drivers (see `docs/project_b_analysis_and_research_plan.md`, Stage 4
section and the R8 work item). Only the winning mechanism arm from Stage 3 is
promoted here: fused doc-level BOND, TIGHT bound, natural order, C={112},
shrink=1 (the R12a/R12c operating point).

All drivers import from `bondmaxsim`. Candidate-set size is held FIXED when
comparing candidate-generation methods so IVF/PLAID speedups cannot be
mistaken for BOND speedups (Stage 1 §5.4 / Convention 7). All wall-clock
timing is INTERLEAVED round-robin with the dense fused baseline, best-of-N
per arm (R12b/R12c methodology) — the dense baseline is never timed
standalone.

## Experiments

### `e01_fixed_candidate_arms.py`
Method separation at fixed candidate budgets {100, 500, 1000, 5000}∩corpus:
- `dense_fused` — exact fused MaxSim over the full corpus (production baseline)
- `bond` — fused BOND exact-safe with the realistic IVF-seeded tau
  (`ivf_seed_cheap`); kernel and seed cost reported separately
- `faiss_ivf_rerank@B` — FAISS-IVF token retrieval, doc aggregation to exactly
  B candidates, exact MaxSim rerank (`src/bondmaxsim/baselines/faiss_ivf.py`)
- `plaid@B` — PyLate FastPlaid with n_full_scores=B, n_ivf_probe=8
  (`src/bondmaxsim/baselines/plaid.py`)
- `pdx@B` — the Mikel-branch flat PDX-IVF (`IndexPDXBONDIVFFlat` via the
  optional `pdxearch` build), same aggregation + exact rerank recipe
  (`src/bondmaxsim/baselines/pdx_ivf.py`); the arm is skipped gracefully if
  `pdxearch` is not installed
- `partitioned@B` — our IVF-partitioned fused scan with nprobe chosen so
  probed docs ≈ B (`src/bondmaxsim/partitioned_scan.py`); APPROXIMATE

Reports ms_per_query (interleaved best-of-5) and recall_vs_exact@10 per arm.
The budget axis doubles as the "mechanism vs candidate generation" separation
plot planned as e03 in earlier drafts.

Usage: `uv run python -m experiments.stage4_integration.e01_fixed_candidate_arms <threads:0|1> [dataset ...]`

### `e02_seeded_tau_recovery.py`
R8's tau_seed question (Stage 1 §4.4): how much of the ORACLE-tau exact-safe
margin (the policy behind the e08/R12c G1 numbers) does a realistic seed
recover? Arms: `self_bound` (-inf; kernel's thread-shared rising tau only),
`seed_partial` (§4.4 partial-dimension seed, value-reference only),
`ivf_seed_cheap` / `ivf_seed_strong` (FAISS-IVF candidates + exact scores of
s∈{10,100} docs; seeding cost measured, pinned to 1 thread — fan-out costs
more than it saves at this work size), `oracle`. Every arm is exact-safe
(tie-aware recall gate = 1.0); reports pruned-docs %, kernel ms/q, seed cost,
and recovery fractions vs oracle.

Usage: `uv run python -m experiments.stage4_integration.e02_seeded_tau_recovery <threads:0|1> [dataset ...]`

### `e03_partitioned_fused_scan.py`
The combined system arm (`src/bondmaxsim/partitioned_scan.py`): document
clusters packed per-partition in panel layout at BUILD time, per-query
centroid probing (one m×P GEMM), fused BOND kernel over the top-nprobe
partitions with a rising seeded tau. This is the PDX-IVF architecture with
our fused kernel as the bucket scanner — the design where the RQ2 3.2×
kernel advantage applies to ~every scanned FLOP while IVF probing shrinks
the scan. APPROXIMATE arm (probing can lose true top-k docs): reported as a
latency-recall frontier over nprobe, never mixed with exact-safe results;
nprobe=P is the exact control (gate-tested). Build cost reported separately.

Usage: `uv run python -m experiments.stage4_integration.e03_partitioned_fused_scan <threads:0|1> [dataset ...]`

## Deferred (with R7 / future work)
- Corpus scale: `pdx_ivf` was implemented and run after all
  (`src/bondmaxsim/baselines/pdx_ivf.py`; Fedora build note in
  `extern/patches/README.md`), but the corpus scale where candidate
  generation is expected to dominate (100k–1M docs) does not fit this
  machine — deferred with R7. e01-scidocs (25.7k docs) shows the crossover
  beginning.
- BOND-vs-ADSampling inside the PDX layout (earlier-draft e02): the RQ3
  mechanism verdict closed in Stage 3; an ADSampling arm would require the
  same PDX C++ integration and is not on the R8 critical path.
- "BOND kernel as reranker over a fixed candidate set" (plan method list):
  superseded by the seeded-tau full-corpus arm — at C={112} the kernel
  already prunes 88–98% of docs itself, and a per-query candidate repack
  would time the packer, not the mechanism.

## Index caches
FAISS indexes: `data/faiss_indexes/<dataset>.faiss`; PLAID indexes:
`data/plaid_indexes/<dataset>/` (both gitignored, regenerable from the
embedding blobs; the drivers build them on first use).

## Output
`results/json/stage4_integration_<experiment>_<dataset>.json`, with one
`runs.mt` / `runs.1t` section per thread setting (merged across invocations).
