Ported onto branch from the gitignored research/Notes/ for self-containment (Stage 0/1 formula source).


# Methodology — BOND + PDX for Late Interaction (paper-ready)

This is a self-contained Methods section: the experiments in
[$bond_maxsim_research_plan.md$](bond_maxsim_research_plan.md) are reproducible from this document
alone. Every claim the research makes is defined and measured here — nothing is inherited from the
preliminary experiments (01–11), which served only as pilots and as a source of reusable code.

---

## M1. Problem setting

Late-interaction retrieval models (ColBERT and its successors) encode a query into $m$ token vectors
and a document into $n_d$ token vectors (here 128-dimensional, L2-normalised; model
$lightonai/GTE-ModernColBERT-v1$). Relevance is the **MaxSim** score

$$S(Q,D) = \sum_{i=1}^{m} \max_{j} \langle q_i, d_j\rangle .$$

Re-ranking a shortlist of candidate documents by exact MaxSim is the cost we target. We study
**pruning**: scanning only part of each document's dimensions and tokens while either provably
(lossless) or with controlled recall (approximate) preserving the true top-K documents.

## M2. Data and embeddings

We use the BEIR corpora **NFCorpus, SciFact, ArguAna, SCIDOCS** (ascending corpus size), with cached
token embeddings, plus their **qrels** for the Stage-3 end-to-end quality study. All token vectors are
unit-norm — a requirement for the Cauchy–Schwarz residual identity in M5. Results are reported
**per-dataset**: corpus statistics that drive pruning (per-dimension anisotropy, MaxSim score
dispersion, candidate-set size) vary across collections, and the conclusions depend on them.

## M3. Storage layout

Each document is stored **dimension-major within the document** — the per-document analogue of a PDX
*vectorgroup*. The value of dimension $z$, token $j$ of document $d$ lives at
$docs[offset_d · D + z · n_d + j]$. One column (a single dimension across all of a document's tokens)
is therefore contiguous, so a scan streams the corpus one dimension at a time. Because a document's
matrix ($n_d · 128 · 4$ bytes ≈ tens of KB) is L2-cache-resident, a per-query reordering of dimensions
is a near-free gather in this layout — in contrast to PDX's *blocked* IVF layout, where a reorder hops
between 64-vector tiles and pays a 1.4–2.7× access penalty. We **measure** this reorder cost
(Stage 1, precondition 5) rather than assume it.

## M4. The synchronized multi-token scan (base kernel)

A single pass over a document's columns serves all $m$ query tokens at once: load $d_j[z]$ once and
multiply it into every $q_i[z]$. This amortises the document's memory traffic across the whole query.
Both the brute-force baseline and the pruning kernel are built on this same synchronized scan, so any
measured difference between them isolates the effect of **pruning**, not of the scan layout. The speedup
of the synchronized scan over per-token matmuls is itself re-established as a measurement (Stage 1),
because the paper relies on it.

## M5. The Cauchy–Schwarz MaxSim pruning bound (full derivation)

This bound is the technical core; it is stated here in full and reproduced in every experiment artifact.

**Partial / residual split.** After scanning a dimension subset $S$ (with $|S| = k$), decompose each
token–token inner product into a known scanned part and an unknown residual:

$$\langle q_i, d_j\rangle = \underbrace{\langle q_i[S], d_j[S]\rangle}_{P_{ij}\text{ (known)}}
   + \underbrace{\langle q_i[S^c], d_j[S^c]\rangle}_{R_{ij}\text{ (unknown)}} .$$

**Cauchy–Schwarz on the residual.** $|R_{ij}| = |\langle q_i[S^c], d_j[S^c]\rangle| \le
\lVert q_i[S^c]\rVert \, \lVert d_j[S^c]\rVert$. For unit-norm tokens the residual norm is

$$\lVert q_i[S^c]\rVert = \sqrt{1 - \textstyle\sum_{d\in S} q_{i,d}^2},$$

which depends only on **which** dimensions are in $S$, not on the order they were scanned.

**Per-pair interval.** $U_{ij} = P_{ij} + \lVert q_i[S^c]\rVert\lVert d_j[S^c]\rVert$ (upper),
$Λ_{ij} = P_{ij} - \lVert q_i[S^c]\rVert\lVert d_j[S^c]\rVert$ (lower).

**Document upper bound → doc pruning.** $UB_d = \sum_i \max_j U_{ij} \ge S(Q,D)$. With $T$ the K-th best
full score seen so far, prune document $d$ as soon as $UB_d < T$. This is sound because $UB_d$ can never
underestimate $S$, so a pruned document cannot belong to the top-K.

**Token lower bound → inner-max (token) pruning.** $L_i = \max_j Λ_{ij} \le \max_j \langle q_i,d_j\rangle$.
A document-token $j$ is dropped when $U_{ij} < L_i$ for **every** query token $i$ — it cannot be the
argmax for any token. The true argmax token of each $i$ always satisfies $U_{i,\text{argmax}} \ge L_i$,
so it always survives; therefore the score finalised over the surviving live set is **exact**.

**Recall control (the single knob).** The residual is scaled by a depth-dependent ramp

$$\beta(k) = \text{shrink} + (1-\text{shrink})\,\frac{D-k}{D},$$

so $shrink = 1$ ⇒ $β ≡ 1$ ⇒ the exact safe bound (recall 1.000, asserted every run), while $shrink < 1$
trades recall for more pruning. The ramp is tightest late (large $k$), where the partial score is already
informative, which avoids an early recall cliff that a uniform residual scale would cause.

**Why dimension order matters.** The bound is a **product** of two residual norms. The query residual
$\sqrt{1-\sum_{d\in S} q^2}$ shrinks fastest when $S$ collects the largest-$q^2$ dimensions; the document
residual $\sqrt{1-\sum_{d\in S} d^2}$ shrinks fastest for the largest-$E[d^2]=\mu^2+\text{var}$ dimensions.
Hence the **order** in which dimensions enter $S$ controls how fast $UB_d$ falls toward $T$ — i.e. how
early a document is pruned. This is precisely what BOND exploits.

## M6. BOND and the three dimension-ordering signals

A single per-query dimension order is computed once and **shared across all $m$ query tokens** (BOND's
reorder is paid once per query and reused by every document). Per-dimension importance is aggregated over
the query's tokens. We compare three signals:

| signal | importance of dimension $d$ | residual factor it shrinks | rationale |
|---|---|---|---|
| $q²$ | $Σ_i q_{i,d}²$ | query | greedy-optimal for the query factor of the bound; native to inner products |
| $\|q−μ\|$ (DTM) | $Σ_i (q_{i,d}−μ_d)²$ | an L2-distance proxy | PDX's shipped single-vector order, ported to MaxSim |
| $q²·(μ²+var)$ | $(Σ_i q_{i,d}²)·(μ_d²+var_d)$ | **both** factors | **new**: query energy × expected document energy |

Here $μ_d$ and $var_d$ are per-dimension statistics over all document tokens in the candidate set. Each
signal then applies PDX's $DISTANCE_TO_MEANS_IMPROVED$ cache layout: take the top-25% of dimensions by
importance, then index-sort each partition so memory access is monotonic-with-gaps (prefetcher-friendly)
rather than a random scatter. Because the residual identity (M5) holds for **any** subset $S$, every
order yields a valid bound and is exact at $shrink = 1$; the orders differ only in **how fast the bound
tightens**.

The $|q−μ|$ signal is an L2-distance heuristic (it estimates $(q−μ)²$, the wrong quantity for an inner
product), inherited from PDX's single-vector setting. $q²$ directly minimises the query factor of the
MaxSim bound, and $q²·(μ²+var)$ additionally weights by expected document energy to shrink both factors —
the principled "BOND for MaxSim." Whether these signals actually diverge on real embeddings depends on
corpus anisotropy (if $μ_d ≈ 0$ then $|q−μ| ≈ |q|$), which we measure directly (Stage 1, precondition 2).

## M7. Kernels

Two C kernels implement the identical algorithm and bounds and are used for different questions:

- **Cells-accurate kernel.** Scans only the live (un-pruned) set; its $cells$ counter — the number of
  $(query-token, doc-token, dimension)$ multiply-adds actually performed — is the hardware-independent
  measure of algorithmic work.
- **Throughput-optimised kernel.** Token-major partial layout so the inner per-query-token loop is a
  contiguous SIMD axpy; a PDX-style **dense warmup** (scan all tokens until ≥50% are prunable) before
  switching to a **positional** survivor-only scan; and bound evaluation skipped over the first $D/4$
  dimensions where nothing prunes. Used for honest wall-clock.

Both are reported because cells and wall-clock answer different questions: cells is the algorithmic,
cross-implementation signal; wall-clock is implementation-specific (our pruning kernels are scalar /
auto-vectorised, whereas a production kernel would be hand-SIMD).

## M8. Baselines (and why each is the right comparator)

- **Brute — synchronized full scan.** Exact MaxSim with no bounds, using the same layout and the same
  synchronized scan (M4). This is the **honest floor**: it already contains the multi-token amortisation,
  so beating it means pruning genuinely pays for its overhead.
- **Natural-order pruning.** The pruning kernel with the identity dimension order. Isolates what
  **reordering** adds on top of pruning alone.
- **Ada-rotation.** The pruning kernel on orthogonally-rotated data, identity order. An orthogonal
  rotation preserves MaxSim exactly while spreading energy evenly across dimensions, giving a fixed,
  layout-friendly "order" that needs no per-query gather. This is **not** ADSampling: ADSampling's ratio
  bound is defined for a single distance and has no MaxSim analogue (the max-pool is nonlinear, and a
  union bound over the $m·n_d$ token pairs per document inflates the error until it prunes nothing); only
  its rotation transfers to MaxSim. We test the rotation and label it honestly.

## M9. Metrics

- **Approximation recall@K** — overlap of the kernel's returned top-K with the exact MaxSim top-K on the
  *same* scored set. This is the correctness/safety invariant of the *pruning* (asserted to be 1.000 at
  $shrink = 1$), and is distinct from end-to-end retrieval recall.
- **Cells scanned (% of full)** — algorithmic work from the cells-accurate kernel.
- **Work–recall frontier** — sweep $shrink$; plot cells (and wall-clock) against approximation recall.
- **Wall-clock** — min-of-repeats ms/query from the optimised kernel, reported relative to brute.
- **End-to-end retrieval quality (Stage 3)** — nDCG@10 / Recall@k against qrels at the chosen operating
  points, establishing that approximate pruning preserves *retrieval*, not merely approximation-recall.

## M10. Experimental design and statistics

Reporting is per-dataset. Queries are the longest available (to give the widest $m$ sweep); documents are
sampled with a fixed seed. **Candidate-set size $n_docs$ and query length $m$ are swept as first-class
axes**, because both govern whether pruning pays (the doc bound needs a selective threshold $T$, and the
shared reorder amortises across $m$). Every timed comparison is a min-of-repeats after warmup. We report
dispersion (IQR / confidence intervals) across queries, since the deciding comparisons are otherwise
within measurement noise. Stages 1–2 are single-thread and float32; threading and quantisation are
explicit Stage-3 axes, not silent omissions.