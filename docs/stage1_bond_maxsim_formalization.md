# Stage 1: BOND-MaxSim Formalization And Implementation Audit

Prepared 2026-06-29 for branch `final-research-implementation`.

Stage 1 formalizes the dimension-pruning method for ColBERT MaxSim and audits
the existing preliminary PDX-BOND-MaxSim kernels against that formalization. It
does **not** introduce a new kernel. The starting candidate is the preliminary
code identified in Stage 0:

- `archive/preliminaries/05_pruning_bound/pruning_bound_tightness.py`
  (document-level upper-bound formula and tightness probe);
- `archive/preliminaries/09_maxsim_pruning/maxsim_kernels.cpp`
  (reference pruning kernel, true-pruning cells counter);
- `archive/preliminaries/10_maxsim_pruning_opt/maxsim_kernels.cpp`
  (optimized kernel, same algorithm and bounds);
- `archive/preliminaries/11_pruning_scale_probe/probe.py`
  (candidate-set-size probe reusing the exp-09 kernel).

The goal is to state the objective and bounds precisely, **prove** the
exact-safe claim for `shrink = 1` or surface the exact gap, separate the
approximate variant, and produce an implementation audit checklist for Stage 2.

The headline result of this audit: the `shrink = 1` kernel is **provably
exact-safe** for top-k MaxSim under three explicit preconditions
(unit-normalized tokens, exact arithmetic, set-equality top-k semantics). The
preconditions, not the algorithm, are where the risk lives. The `shrink < 1`
variant is approximate with no recall guarantee, and the code already labels it
as such.

This audit also **redirects Stage 2**. The exact-safe proof (Sections 2-3) is a
statement about bounds and a pruning predicate, so it transfers unchanged to a
faithful PDX-BOND. But the preliminary kernel prunes at the wrong *granularity*
for PDX (one document's tokens per block, not a wide block of many tokens), which
is exactly where MaxSim pruning is weakest and where PDX's layout cannot pay off.
The preliminary kernel should therefore be **demoted to the exact-safe oracle**,
and Stage 2's first deliverable should be the **multi-vector MaxSim extension of
PDX's existing BOND/PDXearch path** over a wide token block (Section 5) — BOND in
PDX already exists for single-vector kNN, so this is an extension, not a rewrite.
The proof carries over; the threshold policy (Section 4.4), order semantics
(Section 4.5), and cost model (Section 6) do not, and are updated below.

## 1. Definitions And Assumptions

### 1.1 Objective

A query is a set of `m` token embeddings `q_1, ..., q_m`; a document `d` is a
set of `n_d` token embeddings `d_1, ..., d_{n_d}`. ColBERT late interaction
scores them with MaxSim:

```text
score(q, d) = sum_i max_j <q_i, d_j>
```

Top-k retrieval returns the `k` documents with the largest `score(q, d)`.

### 1.2 Assumptions

- **A1 (unit normalization).** Every token embedding is unit-`L2`:
  `||q_i|| = 1` and `||d_j|| = 1`. ColBERT/PyLate normalizes token embeddings,
  so this is expected to hold, but it is a *precondition for safety*, not a
  derived fact (see Section 4.1).
- **A2 (inner-product MaxSim).** The similarity per token pair is the dot
  product `<q_i, d_j>`. Under A1 this is the cosine similarity and is order- and
  rotation-invariant under any orthogonal transform.
- **A3 (fixed embedding dimension).** All tokens live in `R^D` with `D = 128`
  in the preliminary code.

### 1.3 Scanned set and partial score

Let `order` be a permutation of `{0, ..., D-1}` (the shared per-query scan
order). After scanning the first `cur` positions, the **scanned set** is

```text
S = { order[0], ..., order[cur-1] }
```

`S` is a *prefix of the scan order*, hence an arbitrary subset of the original
dimensions. The **partial score** for a query/doc token pair is

```text
P_ij(S) = <q_i[S], d_j[S]> = sum_{z in S} q_iz * d_jz
```

The exact inner product splits as

```text
<q_i, d_j> = P_ij(S) + <q_i[S^c], d_j[S^c]>
```

where `S^c` is the unscanned complement.

## 2. Exact-Safe PDX-BOND-MaxSim (`shrink = 1`)

### 2.1 Residual bound

By Cauchy-Schwarz, the unscanned remainder is bounded by the product of the
residual norms:

```text
| <q_i[S^c], d_j[S^c]> | <= ||q_i[S^c]|| * ||d_j[S^c]||  =:  R_ij(S)
```

Under A1, residual norms are computed from the scanned energy without touching
the unscanned coordinates (order-independent):

```text
||q_i[S^c]||^2 = 1 - sum_{z in S} q_iz^2
||d_j[S^c]||^2 = 1 - sum_{z in S} d_jz^2
```

This is exactly `1 - Qcum[i, cur]` for the query side and `1 - sumsq_d[j]` for
the document side in the kernels. **The use of the constant `1` here is the
algebraic form of assumption A1** (Section 4.1).

### 2.2 Pair interval

```text
L_ij(S) = P_ij(S) - R_ij(S)  <=  <q_i, d_j>  <=  P_ij(S) + R_ij(S) = U_ij(S)
```

`[L_ij, U_ij]` is a valid two-sided interval for every pair at every scanned
prefix.

### 2.3 Document upper bound and document pruning

For each query token `i`, `max_j <q_i, d_j> <= max_j U_ij(S)`. Summing over `i`:

```text
score(q, d) <= sum_i max_j U_ij(S)  =:  UB_d(S)
```

**Document pruning rule.** Let `tau_k` be the `k`-th largest *exact* score among
documents already finalized. If `UB_d(S) < tau_k`, then
`score(q, d) <= UB_d(S) < tau_k`, so `d` cannot displace any current top-k
member and cannot enter the final top-k. The document is safely skipped.

This is the classical branch-and-bound argument and is correct **provided the
scores inserted into the running top-k are exact**. That reduces document-level
correctness to the exactness of the finalized score, i.e. to token pruning
(Section 2.4).

### 2.4 Token pruning — proof of the survival invariant

The kernels also prune the inner `max_j` competition. At a scanned prefix `S`,
for each query token `i` define a lower bound on its contribution over the
**live** token set `Live`:

```text
L_i = max_{j in Live} L_ij(S) = max_{j in Live} ( P_ij(S) - R_ij(S) )
```

A live document token `j` is dropped iff it is **dominated for every query
token**:

```text
drop j   <=>   for all i:  U_ij(S) < L_i
```

**Claim (survival invariant).** For every query token `i`, its true argmax token
`j*(i) = argmax_j <q_i, d_j>` is never dropped, in any pruning round.

**Proof (induction over pruning rounds).**

*Base.* Initially `Live` contains all tokens, including each `j*(i)`.

*Step.* Assume `j*(i) in Live` at the start of a round. For that fixed `i`,
every live token `j'` satisfies

```text
L_ij'(S) = P_ij'(S) - R_ij'(S) <= <q_i, d_j'> <= max_j <q_i, d_j> = <q_i, d_{j*(i)}>
```

Taking the max over live `j'`, `L_i <= <q_i, d_{j*(i)}>`. By the upper interval,
`U_{i, j*(i)}(S) >= <q_i, d_{j*(i)}> >= L_i`. Therefore the domination test
`U_{i, j*(i)} < L_i` is **false** for query token `i`, so `j*(i)` is not
dominated everywhere and is not dropped. Hence `j*(i)` remains in `Live`. ∎

This holds independently of the dimension order, the fetch schedule, and which
other tokens are alive — the bound on `L_i` uses only that all its terms are
themselves lower bounds on `<q_i, ·> <= <q_i, d_{j*(i)}>`.

**The invariant is per-document.** MaxSim's `max_j` ranges over the tokens of a
single document, so `L_i` and the live set are *per-document* state, and `j*(i)`
is the argmax within that document. This is intrinsic to the objective, not to
the layout: under a wide token block that interleaves many documents (Section
5.3), `doc_offsets` still partitions tokens by document and `L_i` must be
maintained per document — never as a max across a document boundary. The proof
then transfers verbatim, per document.

### 2.5 Exact finalization

If a document is not pruned, the loop runs until `cur = D`, so for every live
token `P_ij(S) = <q_i, d_j>` exactly (the residual is zero). Finalization
computes

```text
score = sum_i max_{j in Live} P_ij
```

By the survival invariant, `j*(i) in Live`, so
`max_{j in Live} P_ij = <q_i, d_{j*(i)}> = max_j <q_i, d_j>`. The finalized score
equals the exact MaxSim score. Combined with Section 2.3, the returned top-k is
the exact top-k. **The `shrink = 1` kernel is exact-safe under A1-A3 and exact
arithmetic.**

A second elegant consequence: token pruning never invalidates the document
bound. Because `j*(i)` is always a survivor and
`max_{j in Live} U_ij >= U_{i, j*(i)} >= max_j <q_i, d_j>`, the survivor-only
`UB_d` in both kernels remains a valid upper bound on the full score.

## 3. Approximate Variant (`shrink < 1`)

The single recall knob `shrink in [0, 1]` does **not** scale the residual
uniformly. It sets a depth-dependent confidence ramp (ADSampling-style):

```text
beta(cur) = shrink + (1 - shrink) * (D - cur) / D
```

so `beta(0) = 1` (residual fully trusted when nothing is scanned) and
`beta(D) = shrink` (tightest when the partial score is most informative). Every
residual in the bounds is replaced by `beta(cur) * R_ij(S)`.

For any `shrink < 1` and `cur > 0`, `beta(cur) < 1`, so the scaled residual is
**smaller** than the true Cauchy-Schwarz bound. Consequently:

- `U_ij` may fall below the true `<q_i, d_j>`, so the survival invariant
  (Section 2.4) breaks: a true argmax token can be dropped.
- `UB_d` may fall below the true `score(q, d)`, so document pruning can discard a
  true top-k document.

**There is no recall guarantee for `shrink < 1`.** It is an approximation whose
recall must be *measured* against exact MaxSim and reported only on a
quality-work (or quality-latency) frontier. The benches sweep `shrink` and
report recall versus brute force precisely because it is not provable. This must
never be merged with the exact-safe arm in results.

## 4. Exactness Preconditions And Caveats

The proof in Section 2 is conditional. These are the conditions, ordered by how
much they can hurt.

### 4.1 Unit normalization is a *safety* precondition (highest risk)

The residual uses `1 - sum_{z in S} x_z^2` in place of
`||x||^2 - sum_{z in S} x_z^2`. If `||x|| != 1`:

- `||x|| < 1`: the residual is **over**-estimated. Bounds get looser; still
  safe, only less pruning.
- `||x|| > 1`: the residual is **under**-estimated. `R_ij` can be smaller than
  the true remainder, `U_ij` can drop below `<q_i, d_j>`, and the survival
  invariant can fail. **This silently breaks exactness.**

Action for Stage 2: verify that exported ColBERT token embeddings are unit-norm
to fp32 tolerance, or compute residuals from the actual per-token norm rather
than the constant `1`. Do not assume normalization holds just because ColBERT
"normalizes" — check the cached arrays.

### 4.2 Floating-point exactness is empirical, not proven

The proof assumes exact arithmetic. In fp32, `sqrt(max(0, 1 - sumsq))` and the
accumulated partials carry rounding error, so a true argmax token could in
principle be dropped by a hair, or a doc bound could round below `tau_k`. The
benches observe `recall = 1.000` on the sampled queries (exp-09 `validate()`),
which is strong evidence but not a proof of fp-exactness. Treat "exact" as
"exact up to fp32 rounding"; if Stage 2 ever sees recall `< 1.0` in `shrink = 1`
mode, suspect normalization (4.1) first, then fp accumulation order.

### 4.3 Top-k is set-equality, ties are arbitrary

`TopK.offer` replaces the current minimum slot only when `s` is strictly
greater, and the final sort is by score. With score ties at the threshold, which
document occupies the last slot depends on document processing order — exactly
as `np.argpartition` in the brute-force oracle also breaks ties arbitrarily.
"Exact" therefore means *a* valid exact top-k set, compared by set agreement,
not a canonical ordering. This matches how the benches measure recall.

### 4.4 Threshold policy: safe in both regimes, but effective only with a finite threshold

`tau_k = TopK.threshold()` is the minimum of the `K` slots, initialized to
`-inf`. The document-pruning *theorem* (Section 2.3, `UB_d < tau_k` is safe) holds
for any `tau_k` that is a valid lower bound on the final k-th best score — in
particular for `-inf`. So exactness is never at risk from the threshold. What the
threshold governs is whether pruning *fires*, and this differs sharply between
two execution regimes:

- **Sequential per-document scan (the preliminary kernel).** Documents are
  finalized one at a time, so early documents raise `tau_k` for later ones. Until
  `K` documents are finalized at least one slot is `-inf` and nothing prunes
  (a known efficiency floor, exp-11). Within a document's scan `T` is captured
  once and not refreshed; since `tau_k` only increases, a stale (smaller) `T` is
  conservative. Both choices preserve exactness.
- **Synchronized wide-block scan (the faithful PDX-BOND, Section 5.3).** All
  documents advance together, dimension by dimension, so **no document finalizes
  until the pass ends** and a self-bounded `tau_k` stays `-inf` for the whole
  scan. Self-bounded document pruning then *never fires*. This is not an
  exactness problem — it is an effectiveness problem — but it means the
  synchronized regime needs one of: (a) **staged finalization** (finalize
  documents in waves so a real `tau_k` emerges mid-pass), or (b) a **seeded
  threshold** from IVF, PLAID, or a prior pass. This promotes "seeded threshold"
  from a deferred ablation to a first-class design choice for any wide-block
  implementation. Token pruning (Section 2.4) is unaffected and still fires from
  the first block.

### 4.5 Order and schedule independence

Exactness does **not** depend on `order` or `fetch_schedule`. The residual
formula depends only on the *set* `S`, and the survival proof is order-agnostic.
`order` (natural, `bond` aggregate importance `sum_i (q_i - mu)^2`, or `ada`
rotation) and the fetch cadence affect *how early bounds tighten* — i.e.
efficiency — not correctness. The `ada` arm rotates all tokens by a fixed
orthogonal `R`; orthogonality preserves both `<q_i, d_j>` (A2) and unit norm
(A1), so it remains exact-safe at `shrink = 1`. This must stay true: any
non-orthogonal "rotation" would void A1/A2.

**Order is query-dependent and natively supported — it does not move to build
time.** PDX separates *physical layout* (fixed columnar storage, set at build)
from the *logical dimension access order* (a permutation `indices_dimensions`
recomputed per query). PDX-sigmod's BOND calls `GetDimensionsAccessOrder(query,
means)` once per query in `Search()`, and the distance kernels index columns
through it (`true_dimension_idx = indices_dimensions[dim]`), so the per-query
reorder is an `O(D log D)` sort reused across the entire scan — exactly the
preliminary kernel's "reorder once per query, every doc reuses it". BOND's
`DISTANCE_TO_MEANS_IMPROVED` (which the prototype's `bond` order mirrors) selects
the top 25% of dimensions by the query-dependent importance `|q - mu|`, then
sorts each partition by physical index for cache-friendly access. For MaxSim the
importance is aggregated over the `m` query tokens (`sum_i (q_i - mu)^2`), giving
one shared order per query. The only cost is the gather/cache penalty of
non-sequential column reads, softened by the partition-by-index sort — a constant
factor, not a barrier, and independent of exactness. (The *evolved* `PDX`
physically materializes its 25/75 split for ADSampling's build-time rotation, so
per-query reordering is less natural there — a reason to prefer extending
PDX-sigmod, Section 5.1.)

## 5. PDX Substrate And BOND Granularity

BOND (vertically decomposed data, partial-score bounds, dimension-incremental
pruning) is the *same mechanism* PDX's transposed layout is built to exploit;
ADSampling is the rival pruner. So "use BOND, not ADSampling" selects the pruner
the layout was practically designed for — it does not fight PDX. This section
separates what the preliminary kernels do from what a faithful PDX-BOND for
multi-vector search should do. The Section 2 proof applies to both; only the
granularity, substrate, and the policy items above differ.

### 5.1 What PDX actually provides (two local checkouts)

PDX is a **transposed (columnar/decomposed) layout**: dimensions of different
vectors are stored sequentially, *within a block — e.g. an IVF cluster*. PDX
reports large speedups for **exhaustive** pruned search as well as for IVF, so
the layout+pruning win is separable from the IVF win. Two checkouts are present,
and they differ in what they ship:

- **`extern/PDX-sigmod/` (SIGMOD snapshot, commit `fdc62f2` (public sigmod tip; see Stage 0))** ships a working **BOND**:
  `include/pdx/bond.hpp` defines `PDXBondSearcher : public PDXearch<L2>` — BOND
  "adds no relevant functionality to PDXearch", i.e. it *is* the PDXearch pruned
  columnar engine configured with BOND's dimension order, L2 distance, and **no
  vector transform**. It is exposed as `IndexPDXBONDFlat` (exact, no index,
  exhaustive pruned search) and has IVF variants, alongside `bsa.hpp` and
  `adsampling.hpp`, a full `bench_bond` harness, and existing results
  (including local `results/external_baselines/zen5-martan/IVF_PDX_BOND.csv` runs of BOND vs ADSampling vs
  BSA vs FAISS). **This is the BOND-in-PDX reference implementation** — for
  single-vector L2 kNN.
- **`extern/PDX/` (evolved layout, commit `93531b9`)** kept only
  `pruners/adsampling.hpp` (no BOND/BSA), in a **hybrid 25/75 float32 layout**:
  the first 25% of dimensions fully decomposed (the "vertical block", used for
  pruning), the remaining 75% in 64-dim sub-vectors (the "horizontal block"),
  touched only for survivors and re-checked every 64 dims. Flagship index is
  `IndexPDXIVFTreeSQ8` (two-level IVF + 8-bit quantization). This split is
  *physically materialized* and tuned for ADSampling, whose build-time rotation
  removes the need for a per-query dimension reorder — unlike BOND, which reorders
  per query (Section 4.5). PDX-sigmod instead reorders *logically* per query over
  plain columnar storage, which is what BOND needs.

Implication: BOND in PDX is **not** something to write from scratch — the
SIGMOD-2002 algorithm is already re-implemented for single-vector L2 kNN as a
PDXearch variant. The real gap, and the project's contribution, is the
**multi-vector MaxSim extension** of that BOND/PDXearch path (inner-product pair
bounds, two-level token -> document pruning). The Stage 2 decision is which base
to extend: PDX-sigmod (BOND already present, older layout) or the evolved PDX
(faster hybrid layout + quantization, but BOND would need forward-porting from
the sigmod snapshot). Note PDX BOND is `L2`; for unit-normalized tokens L2 kNN
and inner-product kNN are monotonically related, but MaxSim needs the per-pair
inner-product bound of Section 2 — so the extension changes both the distance
(L2 single-vector -> IP MaxSim) and the granularity (Section 5.3).

### 5.2 The preliminary kernels prune at the wrong granularity (Option B)

The preliminary kernels store each document **dim-major within the document**
(`docs[doc_offsets[d]*D + z*n_d + j]`) and prune per document. This is a
*per-document* analogue of a PDX vectorgroup, but the "block of vectors" is one
document's tokens — narrow (tens), variable, and not the wide block PDX's SIMD
scan is built for. It is also not PDX's 25/75 hybrid layout, uses none of PDX's
SIMD kernels, ADSampling, IVF, or quantization, and its `ada` arm is a *local*
random-rotation analogue, not PDX ADSampling.

Call this **Option B**: BOND with the MaxSim document bound at per-document
granularity. It is the configuration where MaxSim pruning is weakest — the
document bound `UB_d = sum_i max_j U_ij` is loose early (score compression), so
document pruning barely fires (exp-05/11), and the narrow block denies the layout
its SIMD win (exp-09/10: brute force beats the pruner in wall-clock). **A negative
wall-clock or pruning result from Option B is therefore not evidence against
BOND-in-PDX for multi-vector search.** Keep Option B as the **exact-safe
correctness oracle** — the Section 2 proof holds for it verbatim — and as one
ablation arm, not as the verdict.

### 5.3 Faithful PDX-BOND: wide token block, two-level bounds (recommended)

Put BOND where its block is wide: lay out document tokens dim-major **across many
documents at once** (a real PDX vectorgroup of tokens) and run BOND
dimension-incrementally over that wide column. The bounds are *exactly* Section 2:
per-pair intervals `[L_ij, U_ij]` aggregated to the per-document upper bound
`UB_d = sum_i max_j U_ij` for document pruning. This "two-level (token ->
document)" structure **is** the Section 2.3 bound — the wide layout introduces
**no new bound math**; it changes only storage, scan order, and bookkeeping.

Two consequences of the wide layout, both audited above and neither touching the
Section 2 theorems:

- **Per-document bound state** (Section 2.4 note): `L_i` and the live set are
  maintained per document even though the block interleaves documents;
  `doc_offsets` keeps the boundaries.
- **Threshold dynamics** (Section 4.4): a synchronized wide scan finalizes no
  document mid-pass, so `tau_k` must come from staged finalization or a seed
  (IVF/PLAID/prior pass). Token pruning still fires from the first block.

Run this **exhaustive-but-pruned (no IVF)** for the mechanism claim, so any
speedup is attributable to BOND rather than to candidate reduction; compose with
PDX-IVF only for the separate systems claim, reported apart (Stage 0 decision 2).
BOND-vs-ADSampling is already an apples-to-apples comparison inside PDX for
single-vector kNN (the existing `bench_bond` / `ZEN5-Martan` results); the
contribution is to make it so for **MaxSim**, which directly answers the
project's "BOND instead of ADSampling" framing for multi-vector search.

### 5.4 Candidate-generation alternative (Option A) — different math, separate arm

A distinct design treats *all corpus tokens* as the vector pool PDX indexes
(IVF + vertical layout + a pruner), runs per-query-token nearest-token search,
and aggregates documents with MaxSim — the PLAID architecture with PDX-BOND as
the token ANN. This is single-vector BOND over tokens: it ranks tokens globally,
so it uses **single-vector** distance bounds, **not** the Section 2 document
bound, and would need its own formalization. It is the regime PDX is designed for
and a legitimate baseline, but it answers a *candidate-generation* question, not
the *dimension-pruning* question, and must be kept separate so an Option-A win is
never read as a BOND-dimension-pruning win.

## 6. Cost Model

Per document, per scanned dim-block of width `b` over a live set of size
`n_live` with `m` query tokens:

| Component | Cost | Notes |
|---|---|---|
| Block scan (FMAs) | `O(b * n_live * m)` | dominant compute; updates `P` and `sumsq_d` |
| Residual `resq_i` | `O(m)` | one `beta`, `m` sqrt per block |
| Lower bounds `L_i` | `O(n_live * m)` | one pass over live tokens |
| Domination / compaction | `O(n_live * m)` | rebuild live/survivor positions |
| Document bound `UB_d` | `O(n_live * m)` | one pass over survivors |
| Top-k maintenance | `O(K)` per finalized doc | `offer` and `threshold` are linear scans of `K` slots |

The bound bookkeeping is `O(n_live * m)` per block on top of the
`O(b * n_live * m)` scan. When blocks are small (the early fetch schedule starts
at `b = 4`), bookkeeping is the same order as the scan it guards — this is why
the optimized kernel (exp-10) **skips bound evaluation before `D/4` dimensions**
(nothing prunes that early because of MaxSim score compression) and runs a dense
all-token scan during the warmup window, switching to the survivor-only
positional scan only once `>= 50%` of tokens are prunable. Both choices are pure
efficiency and preserve the exactness proof (they change *when* bounds are
checked and *which* tokens are scanned, never the bound values for survivors).

The exp-09 kernel's `cells` counter reflects *true* pruning (it scans only the
live set from the first dimension); the exp-10 kernel deliberately over-scans
during warmup for wall-clock speed, so its `cells` is not the algorithmic-work
signal. Stage 2 accounting mode should use exp-09-style counting; throughput
mode should use exp-10-style scanning. (This is exactly the split exp-11's
comment calls out.)

This cost model is for the per-document scalar prototype (Option B). A faithful
PDX-BOND (Section 5.3) must re-derive the *constants* against PDX's 25/75 hybrid
layout and wide-block SIMD scan: the vertical block (first 25% of dims) carries
pruning over many tokens at once, the horizontal block (64-dim sub-vectors) is
touched only for survivors, and the per-block bound bookkeeping is amortized over
a wide token block instead of one document. The asymptotics (`O(b*n_live*m)`
scan, `O(n_live*m)` bookkeeping) are unchanged; the constants and the SIMD
efficiency — which decide wall-clock — are not, and must be **measured on PDX's
kernels**, not extrapolated from the prototype.

## 7. Empirical Context From The Preliminaries

These are preliminary observations, not Stage 1 claims, but they frame Stage 2:

- Exp-05 tested document-bound tightness (`UB(k) / Score`) with a gate of median
  `< 1.2` at `k = 64` for irrelevant docs. The bound's usefulness depends on how
  fast it tightens.
- Exp-09/10 found that in safe mode (`shrink = 1`) inner-max **token** pruning
  removes most doc-token columns, while **document** pruning fires rarely at
  small candidate counts, and a dense brute-force scan can beat the scalar
  pruner in wall-clock (bookkeeping and gather overhead).
- Exp-11 probes whether document pruning improves with candidate-set size (more
  candidates -> more selective `tau_k`). A downward `cells%` slope would mean
  small samples understated pruning; a flat `~100%` would mean MaxSim score
  compression is the real ceiling.

Stage 1 conclusion on these: they are consistent with the formalization (token
pruning is the strong lever in safe mode; document pruning is threshold-limited)
but they are not decisive and must be rerun under the Stage 2 schema with
wall-clock separated from cells.

## 8. Implementation Audit Checklist For Stage 2

1. **Normalization guard (blocking).** Assert exported token norms are `1` to
   fp32 tolerance, or switch residuals to per-token actual norms. Without this,
   `shrink = 1` is not guaranteed safe (4.1).
2. **Exact-agreement test (blocking).** Standardized `shrink = 1` vs exact
   MaxSim top-k set agreement across all datasets and dimension orders, with a
   hard `recall == 1.0` gate (up to fp tolerance). Extend exp-09 `validate()`
   into the shared testbed.
3. **Two counting modes.** Accounting mode (exp-09 true-pruning cells) and
   throughput mode (exp-10 dense-warmup wall-clock), reported separately
   (Section 6, Stage 0 decision 5).
4. **Approximate arm isolation.** `shrink < 1` results live only on a recall-work
   / recall-latency frontier, never averaged with the exact arm (Section 3).
5. **Order/schedule as efficiency knobs only.** Confirm empirically that
   `order` and `fetch_schedule` never change the `shrink = 1` top-k set
   (regression test), consistent with Section 4.5. The order is **query-dependent
   and recomputed per query** (`GetDimensionsAccessOrder`, a logical permutation
   over fixed columnar storage), as in PDX-sigmod BOND — not a build-time choice;
   the MaxSim order aggregates token importance over the `m` query tokens.
6. **Seeded threshold is first-class (not deferred).** A synchronized wide-block
   scan (Section 5.3) leaves `tau_k = -inf` mid-pass, so self-bounded document
   pruning never fires. Implement staged finalization and/or an IVF/PLAID/prior-
   pass seed as an explicit method arm, kept separate from the self-bound arm so
   a seeded-threshold win is not read as a BOND win (Section 4.4).
7. **Extend PDX's existing BOND to MaxSim (the Stage 2 deliverable).** Start from
   PDX-sigmod's `bond.hpp` / `IndexPDXBONDFlat` (single-vector L2 PDXearch) and
   extend it to the multi-vector inner-product MaxSim bound over a wide token
   block (Section 5.1, 5.3); decide whether to extend PDX-sigmod or forward-port
   BOND into the evolved PDX hybrid layout. Demote the per-document kernel to the
   exact-safe oracle (Section 5.2). Benchmark BOND vs ADSampling for MaxSim,
   reusing the existing single-vector `bench_bond` harness as the template.
8. **Per-document bound state in a wide block.** Maintain `L_i` and the live set
   per document via `doc_offsets`; never take a max across a document boundary
   (Section 2.4 note).
9. **Mechanism vs candidate-generation separation.** Run the dimension-pruning
   claim exhaustive-but-pruned (no IVF); keep any Option-A token-retrieval arm
   (Section 5.4) separate with its own single-vector bounds.
10. **fp determinism.** If any `shrink = 1` run shows `recall < 1.0`, triage in
    order: normalization (4.1), then fp accumulation order (4.2), then a genuine
    bound bug.

## 9. Stage 1 Verdict

- The `shrink = 1` PDX-BOND-MaxSim kernel is **exact-safe by proof** (Sections
  2.3-2.5) under unit normalization, exact arithmetic, and set-equality top-k.
  The token-pruning survival invariant the Stage 0 handoff flagged as uncertain
  is **proven**, not merely claimed.
- The risk is concentrated in the **preconditions**, chiefly unit normalization
  (4.1), which must be verified rather than assumed before any exactness claim
  is published.
- The `shrink < 1` variant is **approximate with no recall guarantee** and must
  be reported only on a frontier.
- The proof is **granularity- and substrate-independent**: it transfers
  unchanged to a wide-block PDX-BOND. What does *not* transfer is the threshold
  policy (4.4), order semantics (4.5), and cost constants (6).
- The preliminary kernel is **Option B** (per-document granularity), the weakest
  configuration for MaxSim pruning and not PDX's layout. It should be the
  **exact-safe oracle**, not the verdict. BOND in PDX already exists for
  single-vector kNN (`extern/PDX-sigmod/include/pdx/bond.hpp`, `IndexPDXBONDFlat`); the
  Stage 2 deliverable is its **multi-vector MaxSim extension** over a wide token
  block, run exhaustive-but-pruned and benchmarked against ADSampling.

On the Stage 0 naming decision: the proof supports promoting the preliminary
method to the final name **BOND-MaxSim** for the `shrink = 1` exact-safe arm,
*conditional on the normalization guard (item 1) passing in Stage 2*. Until that
guard is in the testbed, keep the "preliminary" qualifier. Note that "BOND-MaxSim"
names the *bound and pruning rule*, which is what the wide-block PDX
implementation inherits; the per-document prototype is one (oracle) instantiation
of it, not the method itself.
