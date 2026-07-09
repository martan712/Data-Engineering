# Bound Cost Analysis: What BOND 2002 Actually Does, What We Do, And The Cheap-Bound Plan

Added 2026-07-03. Companion to `docs/stage3b_fused_panel_maxsim_kernel.md`
(§5.5, §5.8, §6.1.1) and the Stage 1 formalization. Instruments introduced
here: `fused_panel_maxsim_bond_cheap` (kernel arm) and e09
(`experiments/stage3_mechanism/e09_bound_tightness_ablation.py`).

## 1. Trigger

Two observations from the Stage 3 runs, plus a source reading, forced this
analysis:

1. **e07** shows the doc-level BOND kernel paying a real premium over the
   dense fused baseline even kernel-only (scifact: 16.7 ms/q natural-order
   BOND vs 15.7 ms/q dense), with pruning too rare to pay it back
   (e06: cells scanned still 84–96% under an ORACLE threshold).
2. **e06** shows the bound machinery is not "a single comparison": producing
   `UB_d` at each checkpoint costs a sumsq FMA in the hot loop, a sqrt per
   lane, and an m × panels reduction — bookkeeping of the kind the original
   BOND paper explicitly warns against.
3. Reading the **original BOND paper** — A. P. de Vries, N. Mamoulis,
   N. Nes, M. Kersten, *"Efficient k-NN Search on Vertically Decomposed
   Data"*, ACM SIGMOD 2002 (June 4–6, Madison, WI), pp. 322–333 — revealed
   that we implemented the analog of the bound family the paper measured and
   REJECTED, and never implemented the analog of the one it recommends.
   (This also settles the Stage 0 loose end on the BOND bibliographic
   details.)

## 2. What we currently do (the tight arm)

`fused_panel_maxsim_bond` (`cpp/fused_panel_maxsim/bond_doc.cpp`)
evaluates, at each non-final checkpoint `c` (dims scanned = `cur`), the
Cauchy–Schwarz document upper bound

    UB_d = Σ_i max_j ( P_ij + resq_i · resd_j )

with `resq_i = β·√(1 − Σ_{scanned} q_i²)` (query-side residual, from Qcum)
and `resd_j = √(1 − Σ_{scanned} d_j²)` (doc-side residual). The doc-side
term is what makes this bound EXPENSIVE:

| cost item | where it lands |
|---|---|
| per-lane sumsq accumulation (`col·col` FMA) | inside the hot loop, every column of query tile 0 (`panel_tile_seg_ss`) |
| `resd = √(1 − sumsq)` per lane | every checkpoint (`panel_resd`) |
| `resd` loads + FMA in the UB reduction | m rows × panels per checkpoint (`doc_row_ubmax`) |
| partial spill/reload between segments | shared infrastructure (any checkpointed MaxSim needs it) |

The §6.1.1 kernel revision already minimized these (fused sumsq, register-
folded final segment, prefetch), and its review verified the *individual*
sqrt/resd-reread costs are small — but the sumsq ride-along in the hot loop
and the residual-envelope UB reduction remain, and nobody had measured the
END-TO-END price of the doc-side residual as a package against a bound that
simply does not need it.

## 3. What BOND 2002 actually does

The 2002 paper's search is *phased candidate-set reduction*, not per-vector
early abandon: process the first m dimensions columnar for ALL candidates,
compute partial scores S⁻, derive the k-th-best bound κ_min, REMOVE every
candidate whose upper bound falls below it, then iterate on the survivors
with a larger m (their Algorithm 2). Three findings matter for us:

1. **Cheap query-only bounds beat tighter per-vector bounds at wall-clock.**
   For histogram intersection they define H_q — upper bound on the remaining
   contribution using ONLY query information (`1 − T(q⁻)`), one add and one
   compare per vector — and H_h, a stricter bound that also tracks a
   per-vector running sum. Table 3 (median ms: H_q 35, H_h 45): *"Although
   H_h prunes more effectively than H_q, the difference is not large enough
   for the additional bookkeeping to pay off."*
2. **Their sophisticated Euclidean bound lost outright.** E_v (Lemmas 1–2,
   per-vector `T(v⁺)` tracking): *"E_v is not that efficient in terms of
   CPU-cost, mainly due to the relatively complex bounds of S(v⁺,q⁺), which
   add computational overhead."* Our tight Cauchy–Schwarz arm is the direct
   E_v-analog.
3. **Per-vector early-abandon against the running top-k was SLOWER** than
   the phased scheme (their footnote 6): *"This version of sequential scan
   was slower on the average (due to the overhead of comparisons, and its
   incapability to consider first the most promising dimensions)."*

Also relevant: their §5.2 observes pruning only pays *within a range of
dimensions* — checking too early is futile (κ_min not yet meaningful),
checking too often adds overhead — which is our checkpoint-set question
(e08) stated 24 years earlier.

PDX-sigmod (`extern/PDX-sigmod/include/pdx/pdxearch.hpp`) inherits the
philosophy in its purest form for L2: the running partial distance IS the
bound (monotone, zero slack, zero extra state), the predicate is literally
`pruning_distances[v] < pruning_threshold`, and pruning happens block-wise
with branchless positions-array compaction. `bond.hpp` adds *nothing* to
that machinery — "BOND" there is purely the dimension ordering. The
source-level contrast with MaxSim is already documented in Stage 3b §5.8.

## 4. What is structural and what was a choice

Two things are FORCED by MaxSim and are not up for renegotiation:

- **We need an UPPER bound; L2 gets a lower bound for free.** Partial L2
  distance is monotone non-decreasing, so it prunes with zero slack and zero
  bookkeeping. A partial inner product bounds nothing without a residual
  envelope. This is the §5.8 finding and it stands.
- **Per-document state is m × tokens partials, not one scalar.** The spill/
  reload between segments and the m × panels reduction shape of any bound
  evaluation are MaxSim's price, common to every bound.

But the CHOICE of residual envelope was ours, and we only ever built the
expensive end:

- tight (E_v-analog): `P_ij + resq_i·resd_j` — needs the doc-side residual
  machinery of §2.
- cheap (H_q-analog): unit-norm document tokens give `resd_j ≤ 1` always, so

      UB_cheap = Σ_i max_j P_ij + Σ_i resq_i

  is a valid upper bound whose residual term is **query-only** —
  precomputable per checkpoint, shared by every document in the corpus.

`UB_cheap ≥ UB_tight` always, so the cheap bound never prunes a document the
tight bound keeps: shrink = 1 stays exact-safe (it can only prune LESS), and
`cells_cheap ≥ cells_tight` deterministically under a fixed (oracle) τ.

What the cheap bound deletes, per checkpointed segment: the sumsq FMA in the
hot loop (tile 0 reverts to the plain microkernel — same inner loop as the
brute arm, prefetch kept), the per-lane sqrt, and the `resd` loads/FMA in
the UB reduction (which becomes a plain running max over the spilled
partials plus one add of the precomputed query term).

What it costs: slack. Early in the scan `resd_j ≈ 1` and the two bounds
nearly coincide — but the energy-concentrating orders (bond, pca) exist
precisely to make `Σ_{scanned} d_j²` grow fast, i.e. to make `resd_j` drop
fast, i.e. to open a gap between the bounds where the tight one prunes more.
The bounds interact with the ordering mechanism in OPPOSITE directions:

| | natural order | bond/pca order |
|---|---|---|
| tight bound | little pruning advantage, full bookkeeping cost | max pruning advantage |
| cheap bound | nearly-tight bound, minimal cost | loosest relative to tight |

Which effect wins at wall-clock is an empirical question — the same question
the 2002 paper answered with "cheap wins" for their metric and data. Given
that exact-safe doc pruning currently fires on only 0.0–2.0% of documents
(§6.1.1), the pruning we can lose is small in absolute terms; the overhead
we can shed is not. But the honest prior is modest either way: the §6.1.1
review measured the sqrt/reread pieces as noise individually, so e09 is
run as a package measurement, not a foregone conclusion.

## 5. Implementation (landed 2026-07-03)

- **Kernel**: `fused_panel_maxsim_bond_cheap` — same ABI, same templated
  body as the tight arm (`fused_bond_doc_impl<CHEAP>` in
  `cpp/fused_panel_maxsim/bond_doc.cpp`), differing ONLY in the
  bound (the §5.8 one-mechanism-at-a-time discipline). As part of the same
  change, the query-side residuals `resq_i(c)` — query-only in BOTH arms —
  are hoisted out of the document loop and precomputed per checkpoint, so
  e09 compares bounds, not incidental hoisting.
- **Bindings/testbed**: `run_fused_panel_bond(..., bound="tight"|"cheap")`;
  Runner method string `fused_panel_maxsim_bond_cheap`; ResultRecord method
  `fused_panel_bond_doc_cheap`.
- **Gates** (`tests/test_fused_panel_gate.py`, all green): exact agreement
  at shrink = 1 across orders × policies × threads for the cheap arm;
  cheap-arm pruning > 0 on the energy-concentrated fixture; the invariant
  `cells_cheap ≥ cells_tight` under fixed oracle τ.
- **e09** (`e09_bound_tightness_ablation.py`, run 2026-07-03 on all four
  datasets): tight vs cheap × {natural, bond, pca}, oracle policy,
  shrink = 1, all queries, best-of-10, all cores, dense fused baseline.
  Reports ms/query, docs pruned %, recall gate at 1.0 (tie-aware).

## 6. Decision criteria and follow-ups

- **Cheap wins ms/query at equal recall (≥ 2 datasets)** → adopt cheap as
  the doc-level default; rerun e05 (approximate frontier) and e08
  (checkpoint sets) on the cheap arm — cheaper checkpoints may also move
  e08's optimal C toward more/earlier checkpoints. The e07 story is then
  partially "E_v's price", not "MaxSim's price".
- **Tight wins** → the residual envelope pays for itself under
  energy-concentrating orders; this is a genuine, quantified divergence from
  the 2002 lesson and belongs in the paper's mechanism section either way.
- **Both lose to dense** (the current exact-safe pattern) → the negative
  result gains a completeness argument: neither the cheap nor the tight end
  of the bound-cost spectrum rescues exact-safe BOND-MaxSim, with e01/e02
  slack instruments explaining why.

**Verdict (e09 run 2026-07-03, all four datasets, recall 1.0 everywhere)**:
at the default checkpoints C = {32, 64} the THIRD outcome holds — both
bounds lose to the dense baseline because neither prunes there: the cheap
bound prunes 0.00% of documents on every dataset × order; the tight bound
prunes at most 5.3% (scidocs, bond order; 4.6% nfcorpus, 2.6% arguana,
1.6% scifact — bond order only). Where nothing prunes, cheap is marginally
faster (nfcorpus natural 9.0 vs 10.2 ms/q — the bookkeeping gap this doc
predicted); where tight's pruning fires it wins narrowly (nfcorpus bond
9.6 vs 10.5). So at early checkpoints the overhead is MaxSim's price, not
E_v's. HOWEVER, e08 (same day) showed late checkpoints (C = {96}/{112})
prune 88–98% of documents and beat dense on 3 of 4 datasets — the regime
where the bound choice actually matters was not covered by e09's
checkpoint set. **RESOLVED — R12a (e09 rerun at the late sets, 2026-07-04, all 4 datasets):
adopt the TIGHT bound.** The cheap query-only bound prunes 0.00% at EVERY
late checkpoint ({112}, {64,112}, {32,64,96,112}) on EVERY dataset — its
`resd→1` relaxation is too loose to fire even at dim 112, so it only pays
checkpoint overhead for zero pruning. Tight prunes 87–98% and is faster in
10/12 natural-order late-checkpoint cases (exceptions: single C={112} on
nfcorpus/scidocs, marginally slower; at C={64,112} and the full set tight
wins on all four). **This REVERSES the 2002 lesson**: de Vries et al. found
the cheap H_q bound beat the tighter E_v/H_h bounds because H_q still pruned
enough; in MaxSim the cheap bound does not prune AT ALL at the checkpoints
that matter, so its cheaper bookkeeping buys nothing — the transfer fails in
the bound MATH (Cauchy–Schwarz residual, no monotonicity), not the
engineering. (The standalone-baseline wall-clock margins from e08/e09 are
inflated — see plan doc R12b/R12c — but the tight-vs-cheap verdict here is a
same-run RELATIVE comparison and is unaffected.)

Explicitly NOT pursued, with reasons: a token-only kernel (the domination
test needs the same shared state plus lane masks and an m-per-lane test —
strictly more bookkeeping than the doc bound it would replace); incremental
UB maintenance across checkpoints (impossible for the tight bound — `resd`
changes at every checkpoint, §6.1.1 — and pointless for the cheap bound,
whose per-checkpoint cost is already a plain max-reduction); replicating
PDX's positions-array compaction inside the fused kernel (its doc-at-a-time
schedule has no cross-document block to compact; the wide-block ACCOUNTING
kernel already implements the PDXearch pattern where it belongs).
