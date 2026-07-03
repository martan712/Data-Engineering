# Stage 3b: Fused Panel MaxSim Kernel — Analysis, Theory, And Implementation Plan

Prepared 2026-07-02 for branch `final-research-implementation`.

Stage 3b is an unplanned but necessary interlude between the Stage 3 mechanism
experiments (e01–e02 done, e03 first run done) and the Stage 3 wall-clock
decision gate. The first e03 run on scifact exposed that our wall-clock
baseline comparison was not measuring what we thought it was measuring: the
PDX-layout brute-force kernel lost to the NumPy/BLAS baseline by 8.2x, which
superficially contradicts the PDX paper's claims. Section 1 decomposes that
gap with measurements; Section 2 explains why it does not contradict PDX;
Sections 3–4 develop the theory and prototype evidence for a kernel design
that beats out-of-the-box BLAS *by exploiting the PDX layout*; Sections 5–6
specify the implementation and how it integrates into the existing stages.

The headline: **the PDX vertical layout is (a mild variant of) the packed-panel
format that BLAS GEMM builds internally on every call.** Because our corpus is
packed once at index-build time while BLAS re-packs 584 MB per query, and
because MaxSim's segmented max-reduction can be fused into the accumulator
register tile while BLAS must materialize the full score matrix, a
register-tiled kernel on the PDX layout is structurally able to beat
`query @ corpus.T`. A prototype confirms this: **50.4 ms/query single-threaded
vs 147.7 ms for single-threaded OpenBLAS on the same machine — and within
1.4x of 12-thread OpenBLAS on one core.** Threading then pushes dense scan to
the DRAM bandwidth wall (~10 ms/query on this machine), below which *only
pruning can go* — which restores the BOND mechanism to its correct role in
the wall-clock story.

## 1. The Measurement That Forced This Stage

First e03 run (scifact: 5,183 docs, 1,141,434 tokens, D = 128, 50 queries,
mean m = 21.2 query tokens, k = 10, oracle policy, shrink = 1):

| arm | ms/query | notes |
|---|---|---|
| `bond` order, throughput kernel | 366.1 | recall 1.0, cells 83.7%, tok_prune 34.7% |
| `brute_pdx` (wide-block dense scan) | 297.6 | same columnar layout, no pruning |
| `brute_numpy` (`Q @ X.T` + `reduceat`) | 36.4 | OpenBLAS, **12 threads** |

Two alarming facts: the PDX-layout brute force is 8.2x slower than NumPy, and
the BOND throughput kernel is slower than its *own* dense baseline despite
pruning 34.7% of tokens.

The 8.2x decomposes into two independent, fully measured factors:

**Factor 1 — threading, 4.06x.** `brute_numpy` is one
`query @ flat_tokens.T` sgemm; OpenBLAS (scipy-openblas 0.3.31) uses all 12
cores. Re-running the identical NumPy baseline with `OPENBLAS_NUM_THREADS=1`
gives **147.7 ms/query** (42 GFLOPS). The kernels are single-threaded, so the
thread-fair gap is 2.0x, not 8.2x.

**Factor 2 — inner-loop codegen, 2.0x.** The kernel's innermost loop is
`for (i = 0; i < m; ++i) Pj[i] += qz[i] * dv` with `m` a *runtime* value
arriving through the ctypes ABI. A micro-benchmark isolating exactly this loop
at scifact shapes (G = 4096, D = 128, 279 groups):

| variant | ms (per query equiv.) | GFLOPS |
|---|---|---|
| `m` runtime (as in the kernel) | 259.6 | 23.7 |
| `M = 21` compile-time constant | 136.2 | 45.1 |
| `M = 24` compile-time, query zero-padded from m = 21 | 134.8 | 45.6 useful |

With runtime `m` the compiler emits a length-checked vector loop with a
5-float scalar tail (m = 21 = one 16-lane AVX-512 vector + remainder) executed
once per token per dimension — the tail handling halves throughput. 259.6 ms
matches the observed 297.6 ms/query (the remainder is finalization, TopK, and
DRAM streaming). Templating on M and zero-padding the query to a multiple of
8 recovers the full 45.6 GFLOPS. Zero-padded query tokens are exact:
`P_ij = sum_z 0 * d_jz = 0` and each padded token contributes
`max_j P_ij = max_j 0 = 0` to the score (see §5.3 for why the analogous
*document*-side padding must NOT use zeros).

A cache-tiling variant (P accumulator tiled to stay L1-resident) was also
tested and gained under 6% — the accumulator traffic theory is **rejected**;
at these sizes (P = 344 KB, L2 = 1 MB/core) the loop is codegen-bound, not
cache-bound.

The second alarming fact has the same root cause plus one more: at 83.7%
cells scanned, only 16.3% of multiply-adds are saved, while the per-boundary
bound evaluation (per-document `L_i` reduction + domination test + UB, all
scalar loops over m per live token) costs more than that. The mechanism's
bookkeeping granularity is wrong for wall-clock, which §5.5 addresses.

## 2. Why This Does Not Contradict The PDX Paper

The PDX/BOND results (Kuffo et al.; `extern/PDX-sigmod`) are obtained in a
regime that differs from MaxSim in every load-bearing dimension:

- **m = 1.** PDX kNN scans one query vector against a block. That workload is
  GEMV-shaped: every corpus value is loaded, used in exactly one multiply-add,
  and discarded — arithmetic intensity is fixed at ~0.25 FLOP/byte, so *every*
  implementation is memory-bound and BLAS register blocking buys nothing.
  Layout and pruning are the only levers, and PDX wins on both.
- **Cache-resident IVF buckets.** PDXearch scans inside an IVF partition that
  fits in cache, amortizing layout benefits across probed queries.
- **Baselines are row-major scalar/SIMD scans**, not a batched multithreaded
  sgemm — because for m = 1 a sgemm *is* just a GEMV and would not be faster.

MaxSim changes the regime: m ≈ 21 query tokens means each loaded corpus value
can be reused 21x. The scoring pass is GEMM-shaped, which is precisely the
workload BLAS is engineered for: register-blocked microkernels, packed
operands, near-peak FLOP utilization, and (by default) all cores. Comparing a
hand-written streaming loop against 12-thread OpenBLAS is therefore a
comparison the PDX paper never makes and never claims to win.

**Consequence for the research narrative (record this at the Stage 3 decision
gate):** in the multi-vector MaxSim setting, any pruning mechanism must beat a
much stronger dense baseline than in the m = 1 setting the PDX/BOND literature
reports on. That is a finding, not a bug. The remainder of this document shows
the baseline is beatable anyway — with the PDX layout as the enabling asset.

## 3. Theory: Why Out-Of-The-Box BLAS Is Beatable Here

`scores = Q @ X.T` followed by `np.maximum.reduceat` forces BLAS/NumPy to do
three things MaxSim does not need. Each is a structural inefficiency that a
custom kernel on the PDX layout removes.

### 3.1 BLAS re-packs the corpus on every call; our layout is pre-packed

High-performance GEMM does not operate on row-major inputs. Internally,
OpenBLAS copies ("packs") the B operand into contiguous dim-major
micro-panels sized to the register tile, because the microkernel needs one
cache-line-aligned column slice per FMA step. For a *skinny* GEMM
(A is m x D with m ≈ 21, B is D x 1.14M), this packing is the dominant cost:
584 MB is read, permuted, and written back **per query**, and the O(m)
arithmetic cannot amortize it. This is exactly why single-thread OpenBLAS
achieves only 42 GFLOPS here against a ~150+ GFLOPS machine peak.

The PDX vertical layout **is** that packed format, up to panel width. A
dim-major group (`group_data[z*G + t]`, Stage 1 §5.3) differs from OpenBLAS's
packed-B only in panel granularity (G = 4096 tokens vs 16). Narrowing the
panel to 16 tokens (§5.2) makes it *identical in kind*: each dimension slice
of a panel is one 64-byte cache line = one AVX-512 register load. We pay the
packing cost **once at index-build time**; BLAS pays it on every query. This
is the PDX layout earning its keep in the m ≈ 21 regime — through packing
amortization rather than the m = 1 memory-bound argument of the paper.

### 3.2 BLAS must materialize S; MaxSim's reduction fuses into the register tile

GEMM's contract is `C = A·B`: the full m x T score matrix S (21 x 1.14M x 4 B
= 96 MB) is written to DRAM, then `reduceat` reads all 96 MB back to compute
per-document maxima that occupy 21 x 5,183 floats. MaxSim only ever needs the
*running per-document max* of each score row. In a custom kernel the
accumulator tile (m query tokens x 16 corpus tokens) is register-resident
when a panel finishes; folding it into a per-document running-max vector is
one `vmaxps` per query token per panel, plus a horizontal reduction at each
document boundary. S never exists. This removes ~192 MB of DRAM round-trip
per query — BLAS structurally cannot do this because its epilogue is fixed.

### 3.3 GEMM cannot stop early; the vertical layout prunes below the memory wall

The fused dense kernel is embarrassingly parallel over groups. Threaded, it
saturates DRAM: any full-scan method must stream the 584 MB corpus, which at
this machine's realistic ~50–70 GB/s puts a hard floor of roughly
**8–12 ms/query** on dense MaxSim (NumPy 12T sits at 36 ms partly because of
the S round-trip of §3.2). At the wall, more FLOPS are free but more *bytes*
are not — the only way below the floor is to not read bytes. That is
dimension-incremental pruning, which requires partial scores by dimension
prefix — impossible inside a GEMM, natural in the vertical layout. This
inverts the e03 result's meaning: dense is bandwidth-capped, and BOND pruning
becomes the mechanism that breaks the cap — the PDX narrative, correctly
transplanted to MaxSim. The catch: bound evaluation must be near-free, which
sets the panel-granularity requirement of §5.5.

### 3.4 Roofline summary

Per query at scifact scale (T = 1.14M, D = 128, m = 21): 6.19 GFLOP useful
work, 584 MB corpus bytes.

| implementation | limiting resource | measured / predicted |
|---|---|---|
| current wide-block brute, 1T | inner-loop codegen (runtime m) | 297.6 ms measured |
| OpenBLAS 1T (`Q @ X.T`) | per-call packing + S round-trip | 147.7 ms measured |
| OpenBLAS 12T | packing + S, partially hidden | 36.4 ms measured |
| fused panel kernel, 1T | FMA throughput (122 GFLOPS) | **50.4 ms measured (prototype)** |
| fused panel kernel, ~6–12T | DRAM bandwidth (584 MB stream) | ~8–12 ms predicted |
| + panel-level BOND pruning | bytes actually read | below the floor; magnitude = Stage 3 question |

## 4. Prototype Evidence

`scratchpad` prototype (to be committed as a bench under
`cpp/fused_panel_maxsim/`), full 584 MB corpus streamed from DRAM (no
artificial cache residency), m = 21 padded to M = 24, best-of-5:

```text
current kernel loop : 143.6 ms   42.7 GFLOPS   (loop replica; in-situ kernel: 297.6)
fused microkernel   :  50.4 ms  121.8 GFLOPS   (2.9x, single thread)
```

Single-threaded, the fused microkernel is 2.9x faster than single-threaded
OpenBLAS and within 1.4x of **12-thread** OpenBLAS. The prototype omits
per-document epilogue boundary handling and TopK (both cheap: ~5,000 docs of
bookkeeping vs 6.19 GFLOP of scan); the production estimate stays ~55 ms/query
single-threaded.

## 5. Kernel Design Specification

New C++ target `cpp/fused_panel_maxsim/` (the existing
`cpp/wide_block_maxsim_bond/` accounting kernel is **kept unchanged** — it
remains the algorithmic-work instrument; Stage 3b only replaces the
wall-clock instruments).

### 5.1 Notation

- `PT = 16` — panel width in tokens (one AVX-512 register of floats).
- `M ∈ {8, 16, 24}` — compile-time query-tile heights. A query with m real
  tokens is split into ceil(m/24) tiles, each zero-padded to the smallest
  M ≥ its size (e.g. scifact has queries up to m = 34 → tiles of 24 + 16 →
  M = 24 and M = 16). Tiles iterate *inside* the panel loop, so the 8 KB
  panel stays L1-resident across tiles — multi-tile queries re-use cached
  panel data instead of re-streaming the corpus (§5.4).
- Groups, `doc_offsets`, `group_doc_starts`: unchanged from
  `pack_corpus_wide` semantics — documents stay whole within groups.

### 5.2 Panel-major packing (`pack_corpus_panels`, new in `data/packing.py`)

Within a group, tokens are stored in consecutive **panels of PT = 16 tokens**;
within a panel, storage is dim-major:

```text
panel_data[ panel_base + z*PT + j ],   j in [0,16), z in [0,D)
```

Each panel is 16 x 128 x 4 B = 8 KB contiguous; each dimension slice is
exactly one cache line. This is the OpenBLAS packed-B format with the packing
done once, offline (§3.1). It is also closer to the PDX paper's own small
vertical blocks than our current G = 4096 full-group columns.

### 5.3 Document padding — duplicate-last-token, NOT zeros

Panels must not straddle document boundaries (the epilogue's per-doc max
would need per-lane masks in the hot loop). Each document's token count is
padded up to a multiple of PT **by duplicating one of its own tokens** (the
last one). Correctness: `max_j` over a multiset is invariant under
duplicating an element, so every `max_j <q_i, d_j>` — hence every score — is
bit-identical. **Zero-padding would be wrong**: `<q_i, 0> = 0` would clamp
`max_j` at ≥ 0, corrupting scores for query tokens whose true max similarity
is negative. (Contrast with query-side padding, §1, where zero rows are exact
because padded *query* tokens only add `max_j 0 = 0` to the sum.)

Cost: mean doc length 220 tokens → ~8 padded tokens/doc ≈ **3.6% extra
corpus** (memory and dense-scan bytes). Recorded per dataset in the packing
metadata; all `cells_scanned` accounting stays on the *unpadded* kernel, so
no Stage 3 accounting numbers are affected.

`doc_offsets`/`group_offsets` gain padded variants (`*_padded`); original
unpadded offsets are kept alongside for id mapping and stats.

### 5.4 Microkernel and dispatch

Per panel, templated on M:

```text
acc[M] : M zmm registers (M x 16 fp32 accumulator tile)     // M=24 -> 24 regs
for z in 0..D-1:
    col = load_512(panel + z*PT)                            // one cache line
    for i in 0..M-1 (unrolled):
        acc[i] = fma(broadcast(qpack[z*M + i]), col, acc[i])
epilogue:
    for i: dmax[i] = max_ps(dmax[i], acc[i])                // running per-doc max
    at document end: score += Σ_i reduce_max(dmax[i]); offer to TopK; reset dmax
```

- `qpack[z*M + i]` is the query packed dim-major with the M padding applied —
  built once per query in Python (like `build_qcum`).
- M = 24 uses 24 of 32 zmm registers for the tile + col + broadcast scratch:
  fits exactly; this is why 24 (not 32) is the largest tile.
- Queries with m > 24 run multiple tiles per panel (§5.1); the panel is
  L1-resident so extra tiles cost FMAs, not memory traffic. Runtime dispatch:
  `switch` over the three instantiations per tile.
- Prototype-measured: 121.8 GFLOPS useful at M = 24 (m = 21), one core.

### 5.5 Panel-granularity BOND bounds (the pruning variant)

The fused kernel comes in two entry points:

- `fused_panel_maxsim_brute` — dense, no bounds. The dense layout baseline in
  e03+ (the old `brute_pdx` wide-block dense scan and the wide-block
  THROUGHPUT kernel were **removed** from the wall-clock surface entirely;
  the wide-block ACCOUNTING kernel remains as the algorithmic-work
  instrument).
- `fused_panel_maxsim_bond` — identical microkernel, but the z-loop is split
  at a small number of checkpoints (e.g. z = 32, 64; aligned to the existing
  fetch-schedule boundaries). At a checkpoint the partials are already in
  registers; per panel we compute the Cauchy–Schwarz document upper bound
  contribution and drop whole *panels* (and whole documents, once all their
  panels are dropped or bounded) when `UB_d < tau`. Token-level domination
  pruning (§2.4 of Stage 1) is **not** performed in the wall-clock kernel —
  e03's numbers show its bookkeeping costs more than it saves at wall-clock;
  it remains measured in the accounting kernel where it belongs.
- Threshold policies (self_bound / oracle / seed) and `shrink` semantics are
  unchanged; `tau` is read per checkpoint exactly as today.
- Exactness: same bound math as Stage 1 §2.3 at coarser granularity — a
  coarser (panel-level) max over resd is still an upper bound, so shrink = 1
  remains exact-safe; the exact-agreement gate re-verifies this empirically.

### 5.6 Threading

OpenMP over groups (`#pragma omp parallel for schedule(dynamic)`); groups are
independent. Per-thread TopK, merged at the end (K = 10; negligible). For the
BOND variant, `tau` is shared through a relaxed atomic float that threads
raise monotonically; a stale (lower) `tau` prunes *less*, never incorrectly,
so exact-safety is unaffected. Thread count from `OMP_NUM_THREADS` so
experiments can pin 1-thread and n-thread arms explicitly; every
`ResultRecord` for wall-clock arms must now record the thread count.

## 6. Implementation And Integration Plan

Work items, in order (each lands with tests green before the next):

- **K1 — packing.** `pack_corpus_panels` in `src/bondmaxsim/data/packing.py`
  (§5.2–5.3) + tests: layout round-trip, padding max-invariance (score
  equality vs unpadded oracle on random data), 16-alignment of every doc.
- **K2 — fused brute kernel.** `cpp/fused_panel_maxsim/` with
  `fused_panel_maxsim_brute` (§5.4), Makefile (`-O3 -march=native -fopenmp`),
  ctypes bindings in `src/bondmaxsim/kernels/fused_panel.py`; blocking check:
  exact agreement (recall == 1.0 and score equality within float tolerance)
  vs `exact_maxsim_topk` on all four debug datasets.
- **K3 — threading.** §5.6 on the brute kernel; verify 1T result invariance
  and measure scaling curve (expect DRAM saturation near 6 threads).
- **K4 — fused BOND kernel.** `fused_panel_maxsim_bond` (§5.5); blocking
  check: shrink = 1 exact-agreement gate across policies and orders (the
  Stage 2 gate re-run on the new kernel).
- **K5 — testbed + e03 refresh.** Runner gains `brute_fused` (1T and nT) and
  `bond_fused` modes; e03 adds them as arms and *keeps* `brute_numpy` in both
  12T and pinned-1T form so the threading factor stays visible in the
  figures. Re-run e03 (all four datasets); e04–e07 then run against the new
  wall-clock instruments.

Integration rules:

- The Stage 3 **decision gate** ("does any exact-safe arm beat brute force on
  wall-clock?") is evaluated against the *strongest* dense baseline:
  multithreaded fused brute (predicted ~10 ms/query), not NumPy. Record the
  §2 regime finding alongside the gate outcome either way.
- The accounting kernel and all e01/e02 results are unaffected (algorithmic
  work is layout- and codegen-independent); no re-runs needed there.
- Stage 4/5 inherit the fused kernels as the production wall-clock path.

### 6.1 K4 outcome (measured 2026-07-02, scifact, oracle policy, shrink = 1)

All arms recall 1.000. Wall-clock (50 queries, best-of-5):

| arm | 1 thread | all cores |
|---|---|---|
| BOND fused (natural / bond / pca order) | 81.0 / 89.1 / 81.6 ms/q | 16.7 / 17.1 / 16.4 ms/q |
| dense fused | 54.9 ms/q | 13.1 ms/q |
| dense NumPy/BLAS | 121.5 ms/q | 42.5 ms/q |

The instruments now agree with each other and the mechanism question gets a
clean answer on scifact: **exact-safe document-level pruning at the {32, 64}
checkpoints fires almost never** (0.0–2.0% of documents, vs the accounting
kernel's 99.8% "pruned before scan end") — the bounds are too loose mid-scan,
exactly as e01's bound-slack trajectories and e02's survival curves showed:
documents only become prunable late in the dimension scan, where most of
their bytes are already read. The BOND arms therefore pay the segment
spill/reload + bound-evaluation overhead (~1.3–1.6x over dense fused) and
prune nothing back. Note the accounting kernel's 84–92% cells-scanned had
already bounded the best case: even perfect pruning could save at most ~16%
of dense work on scifact at shrink = 1.

Decision-gate implication: on scifact, no exact-safe arm beats the dense
fused baseline; the exact-safe wall-clock win, if it exists, must come from
datasets/orders with earlier bound collapse or from the approximate regime
(shrink < 1, e05 — smaller residual scaling brings checkpoints forward).
This is now a *defensible* negative signal because both sides run the same
microkernel — the remaining e03 datasets and e05 decide the gate.

Risks / open points:

- The ~10 ms dense floor prediction assumes ~50–70 GB/s sustained; measure
  actual scaling in K3 before quoting it.
- Panel-level bounds are coarser than token-level: pruned-cell percentages
  will drop relative to e02's token-level curves. That is expected and must
  be reported as "wall-clock pruning" vs "accounting pruning", not mixed.
- If e03-refresh still shows no exact-safe wall-clock win at the DRAM wall,
  the negative-result path of the plan applies — but now against a defensible
  baseline, which makes the result publishable either way.
- Future headroom (out of scope for 3b, note for the paper's discussion):
  int8 + AVX512-VNNI (present on this Zen 5 machine) quadruples per-core
  FMA throughput and halves streamed bytes; ColBERT embeddings are unit-norm
  and quantize well.
