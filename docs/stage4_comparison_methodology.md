# Stage 4 Comparison Methodology (R8)

What the Stage 4 integration experiments compare, the controls that make the
comparison fair, and a dissection of why the Mikel-branch benchmark results
do not contradict ours — they measured different, uncontrolled quantities.
Two distinct headline numbers circulate from that branch and neither
survives the controls:

- **"~30x on full SciFact"** (his README): the IVF+exact-rerank recipe vs
  HIS exact reference — a per-document Python/NumPy loop at ~703 ms/q. Our
  fused dense kernel computes the identical exact result in ~20 ms/q (35x),
  so the properly-engineered EXACT baseline alone delivers more speedup
  than the approximate recipe did; fully timed at matched budget the same
  recipe is slower than the exhaustive fused scan (§3).
- **PLAID ~60–190x slower than FAISS/PDX** (his fig2 baselines chart): a
  budget/timer/stack confound stack, dissected in §3.

This document is the justification for the "candidate-set size must be held
fixed" control in the plan (Stage 4 "Important control"; Stage 1 §5.4) with
the concrete failure case that motivated it.

## 1. What we compare (e01: `experiments/stage4_integration/e01_fixed_candidate_arms.py`)

Six method families, all answering the same query workload:

| arm | exact? | stages inside the timer |
|---|---|---|
| `dense_fused` | yes | fused panel kernel over the full corpus |
| `bond` (exact-safe, seeded τ) | yes | fused BOND kernel (tight bound, natural order, C={112}) over the full corpus; the IVF τ-seeding pipeline is timed separately and ALSO charged (`ms_per_query_total`) |
| `faiss_ivf_rerank@B` | no | per-token FAISS-IVF search + per-doc aggregation + candidate truncation to B + exact MaxSim rerank of B docs |
| `plaid@B` | no | fast-plaid end-to-end retrieval with `n_full_scores = B` |
| `pdx_ivf@B` (Mikel's flat PDX-IVF) | no | per-token `IndexPDXBONDIVFFlat` search + per-doc aggregation + truncation to B + exact MaxSim rerank of B docs |
| `partitioned@B` (ours) | no | centroid probe (one m×P GEMM) + fused BOND scan of the top-nprobe partitions, nprobe chosen so probed docs ≈ B |

Budgets B ∈ {100, 500, 1000, 5000} ∩ corpus size. Quality metric:
recall@10 against the exact MaxSim oracle on the same 50 queries
(seed 42 — the e08/r12c subsample). Results:
`results/json/stage4_integration_e01_fixed_candidate_arms_<dataset>.json`
(runs.mt / runs.1t) and the budget-axis figure per dataset.

Companion experiments: e02 (`e02_seeded_tau_recovery.py`) ablates ONLY the
τ-seeding policy of the exact-safe kernel (self_bound / partial-dim seed /
IVF-seeded cheap+strong / oracle); e03 (`e03_partitioned_fused_scan.py`)
sweeps the partitioned arm's nprobe as a latency-recall frontier with a
brute-vs-bond scanner ablation and the exact-safe partition-bound probe.

## 2. Why our comparison is pure

1. **Fixed candidate budget.** Every candidate-generation arm is compared at
   the same B — the number of documents eligible for full scoring. A method
   may not gain by silently scoring fewer documents; if it fills its budget
   with better candidates it wins on recall, not on a hidden work reduction.
   (Stage 1 §5.4: an IVF speedup must never be read as a scoring-kernel
   speedup, and vice versa.)
2. **Complete pipelines inside the timer.** Everything a deployed system
   would execute per query is timed: token searches, per-doc aggregation
   (including Python-side hit conversion — faithfully ported from the Mikel
   pipeline), candidate gathering, reranking, top-k extraction. Nothing sits
   between two timers.
3. **Interleaved timing.** All arms and the dense baseline run round-robin,
   best-of-N per arm, in one process on one machine (the R12b/R12c lesson:
   a standalone-timed baseline absorbed a thermal window and inflated
   arguana's margin from +8.7% to −19.6%).
4. **One stack.** Same OS, same venv, same session, same 50-query subsample,
   same exact-oracle reference for every arm. Thread settings are explicit
   per run (all-cores and 1T recorded separately); the τ-seed pipeline is
   pinned to 1 thread because fan-out costs more than it saves at that work
   size (measured: 33.6 vs 8.4 ms/q on scifact) — recorded as
   `seed_threads: 1` in the JSON.
5. **Index build separated from query latency.** FAISS / PLAID / PDX /
   partition builds are reported (or cached) outside the timed loops —
   build cost is a property of indexing, not of a query.
6. **Exact and approximate never mix** (Convention 4). Exact-safe arms carry
   a tie-aware recall gate (must be 1.0); approximate arms report plain
   recall@10 vs exact and live on a frontier, not in the exact table.

## 3. What was wrong with the Mikel-branch chart

Source: branch `Mikel`, `experiments/pipeline/03_beir_ivf_benchmark.py`
(PDX-IVF sweep) and `experiments/pipeline/04_baselines.py` (PLAID/FAISS
baselines, figure inputs). Three confounds stack in PDX-IVF's favour:

1. **A ~164x budget mismatch.** The PLAID baseline ran with its default
   `--plaid-full-scores 8192` — fast-plaid fully re-scores up to 8192
   candidate documents per query — while the PDX-IVF pipeline reranked
   `top_c = 50` documents (sweep C ∈ {20, 50, 100}; the headline row is
   C=50; `maxsim_scores_for_candidate_docs` in `utils_colbert.py` confirms
   top_c truncates the per-query candidate list, one query at a time). On
   the chart's corpus (scifact_full, 5,183 docs) `n_full_scores` exceeds
   the corpus size, so PLAID's effective budget was "everything its probes
   surface, uncapped" against the other arms' 50. The chart therefore
   compares a 50-doc rerank against an effectively unbounded rerank and
   attributes the difference to the engine.
2. **An untimed pipeline stage.** PDX-IVF's `total_seconds =
   candidate_generation_seconds + rerank_seconds`, where generation times
   only the `index.search` calls and rerank times only the MaxSim scoring.
   The per-query Python aggregation loop between them (per-token, per-hit:
   doc mapping, best-similarity bookkeeping, candidate-set construction)
   is outside both timers — and in our end-to-end measurements that stage
   is a material fraction of the pipeline at realistic budgets.
3. **Two runtime stacks.** The PLAID/FAISS baselines ran in the Windows venv
   (CPU torch on Windows; the script header says so explicitly), the
   PDX-IVF numbers in WSL. Cross-OS wall-clock is not comparable
   (the Stage 5 fairness list exists for this reason).

None of this is fabrication — each number is a true measurement of what its
script timed. The chart's *comparison* is what fails: budget, timed
boundary, and stack all differ across bars.

The same applies to the FAISS-IVF bar: `faiss_ivf_pipeline` in
`04_baselines.py` is a carbon copy of the PDX recipe (`top_c = 50`, timers
around `index.search` and the rerank only, the Python aggregation loop
untimed between them, and `faiss.omp_set_num_threads(1)` on top), so FAISS
"beat" PLAID for the same reasons. That the distortion hits FAISS and PDX
identically confirms it is systematic to the harness (budget + timer
boundary), not a property of any engine.

**The chart itself** ("Baselines on full SciFact (5183 docs, 50 queries)",
Mikel branch figures): Exact MaxSim 35.15 s, FAISS-IVF+rerank 0.89 s,
PLAID (CPU) 55.95 s, PDX-IVF+rerank [WSL] 0.29 s; qrels recall@10 = 0.938
for exact/FAISS/PDX vs 0.794 for PLAID. Per query: exact 703 ms, FAISS
17.8 ms, PLAID 1119 ms, PDX 5.8 ms — i.e. PLAID-vs-PDX is 193x on this
figure, composed of the budget mismatch (~10–20x), the Windows-torch vs
WSL stack gap (the subtitle itself concedes "PDX [WSL] absolute time not
directly comparable"), the untimed aggregation, and PLAID's fixed per-call
overhead amortized over only 50 queries. Two internal tells corroborate:
(a) PLAID did the MOST scoring work yet returned the WORST quality —
its scoring is quantized while the IVF arms rerank exactly and inherit the
exact reference's 0.938; (b) the exact reference itself runs at 703 ms/q
where our fused dense kernel computes the identical result in ~20 ms/q
(e01, interleaved) — 35x — so "IVF beats exact" on this corpus was an
artifact of a weak exact baseline. Fully timed at matched budget (e01),
the same PDX pipeline costs ~27 ms/q at B=100 — slower than scanning the
whole corpus with the fused kernel.

**Controlled result.** With all three confounds removed (same budget, same
timer boundary, same stack, interleaved), the three token-level pipelines
collapse onto each other — e.g. at B=100 on scifact (validation run;
authoritative numbers in the results JSONs): pdx_ivf 84 ms/q ≈ faiss_ivf
85 ms/q ≈ plaid 96 ms/q — and ALL of them sit far above the exhaustive
fused dense scan. This is expected: `IndexPDXBONDIVFFlat` trains its coarse
quantizer with faiss k-means internally (`pdxearch/index_core.py` wraps
`faiss.IndexIVFFlat`), so the faiss and PDX arms share the candidate
structure and differ only in the in-bucket scan kernel, which is a small
fraction of the pipeline at these corpus sizes.

## 4. Takeaways for the paper

- Report candidate-generation methods only at matched budgets, with the
  budget on the axis (the e01 figure), and state the timer boundary.
- The honest headline at this corpus scale is that the exhaustive fused
  kernel beats every candidate pipeline because per-query candidate
  generation costs more than the whole scan (the RQ2 kernel already
  removed the fat the pipelines are designed to cut).
- Where PDX-IVF's *idea* genuinely helps is partition-granularity skipping
  with the fused kernel as the bucket scanner (the e03 arm) — the
  approximate frontier, at the price of probing recall, not the exact
  table.

## Build note (pdx_ivf arm)

`pdxearch` from `extern/PDX-sigmod` requires the nested Eigen submodule and
one documented setup.py change on Fedora — see `extern/patches/README.md`.
