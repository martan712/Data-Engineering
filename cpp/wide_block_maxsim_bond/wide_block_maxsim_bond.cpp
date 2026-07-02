// wide_block_maxsim_bond.cpp — wide-block MaxSim BOND kernel (Stage 2 deliverable).
//
// Faithful PDX-BOND extension of the Section 2 bound math
// (docs/stage1_bond_maxsim_formalization.md) to a WIDE token block that
// interleaves many whole documents, instead of the narrow per-document block
// used by cpp/per_document_oracle/.  Same pair-interval bounds, same per-doc
// token-pruning survival invariant (§2.4), same document upper bound (§2.3);
// only the storage granularity, scan synchronization, and threshold policy
// (§4.4 staged finalization / seeded tau) differ.
//
// Pattern origin (cited, not copied): the Start -> Warmup -> Prune dimension-
// incremental scan, DIMENSIONS_FETCHING_SIZES cadence, and positions-array
// survivor compaction mirror
//   extern/PDX-sigmod/include/pdx/{bond,pdxearch}.hpp @ fdc62f2
// generalized here from single-vector L2 kNN to multi-vector inner-product
// MaxSim with a two-level (token -> document) bound.
//
// Layout (see src/bondmaxsim/data/packing.py::pack_corpus_wide):
//   The corpus is partitioned into "wide vectorgroups" of consecutive WHOLE
//   documents (~4096 tokens/group; an oversized document gets its own group).
//   Within a group of G tokens, storage is dim-major ACROSS ALL documents in
//   the group:
//       group_data[group_offsets[g]*D + z*G + t]
//   where t is the token's LOCAL index within the group (0..G-1, documents
//   concatenated in id order) and z is the physical dimension.  Groups are
//   concatenated into one flat buffer; group_offsets (len n_groups+1) gives
//   each group's cumulative GLOBAL token offset (mirrors doc_offsets).
//
//   doc_offsets (len n_docs+1): cumulative GLOBAL token counts per document.
//   A document's tokens live entirely inside exactly one group (never split),
//   so group_doc_starts (len n_groups+1) gives the first GLOBAL document id
//   of each group; group g owns documents [group_doc_starts[g],
//   group_doc_starts[g+1]).
//
// Algorithm (per query), Stage 1 §5.3:
//   Groups are processed sequentially, sharing ONE global TopK across all
//   groups (staged finalization, §4.4).  Within a group, ALL documents'
//   tokens are scanned in lockstep, dimension-block by dimension-block
//   (order + fetch_schedule, identical semantics to the oracle).  At every
//   fetch boundary:
//     - per-DOCUMENT token pruning: L_i(d) = max over doc d's live tokens of
//       (P_ij - resq_i*resd_j); a token j of doc d is dropped iff it is
//       dominated for every query token i.  L_i and the live set are NEVER
//       computed across a document boundary (§2.4 note) -- the live array
//       stays sorted by original token index (compaction preserves relative
//       order), so each document's live tokens form a contiguous run and the
//       reduction is done per document via a two-pointer scan over
//       doc_offsets intersected with the group's token range.
//     - per-DOCUMENT bound: UB_d = sum_i max_{j in live(d)} (P_ij +
//       resq_i*resd_j); document d is pruned (all its live tokens removed
//       from the scan) when UB_d < tau, tau = max(tau_seed, topk.threshold()).
//   When the group's scan reaches cur == D, residuals are exactly zero, so
//   every surviving document's exact score is computed
//   (sum_i max_{j in live} P_ij) and inserted into the global TopK -- this
//   is the "staged finalization" that lets tau become finite for LATER
//   groups even though tau_seed = -inf (self_bound policy).  With a
//   precomputed tau_seed (oracle/seed policy) pruning can also fire inside
//   the very first group.
//
// Two entry points (extern "C"), identical ABI:
//   uint64_t wide_block_maxsim_{accounting,throughput}(
//       const float* group_data, const uint64_t* group_offsets, size_t n_groups,
//       const uint64_t* doc_offsets, size_t n_docs,
//       const uint64_t* group_doc_starts,
//       const float* query, size_t m, size_t D,
//       const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
//       const float* Qcum, float shrink, float tau_seed, size_t K,
//       uint32_t* topk_id, float* topk_score, uint64_t* stats,
//       uint64_t* block_doc_live, uint64_t* block_token_live);
//
//   stats[0] = cells_scanned (multiply-adds: (dims in block) * (live tokens) * m,
//              exactly like the oracle's counting convention).
//   stats[1] = docs_pruned (incremented once per document, at the round its
//              UB_d < tau fires -- its already-token-pruned-this-round
//              survivors do NOT additionally count toward tokens_pruned,
//              mirroring the oracle's counting: a doc-pruned doc's remaining
//              live tokens are absorbed under docs_pruned only).
//   stats[2] = tokens_pruned (tokens dropped by the per-document domination
//              test; counted once per dropped token, exactly like the
//              oracle's accounting kernel).
//
//   block_doc_live / block_token_live (len n_fetch each, NULLABLE -- pass
//   NULL to skip): block_doc_live[b] / block_token_live[b] accumulate, over
//   ALL groups, the number of still-live documents / summed live-token count
//   observed at fetch-boundary index b AS EACH GROUP PASSES THAT BOUNDARY.
//   A group that finishes (fully pruned or D reached) before boundary b
//   simply contributes nothing further at indices >= its own boundary count
//   (not explicit zeros) -- documented approximation, Stage 2 spec §"e02".
//
// accounting variant: survivor-only scan from dimension 0 every round (true
//   algorithmic work; live token set is compacted via a positions array in
//   place, PDXearch pattern).
// throughput variant: dense (all G tokens of the group) warmup with no bound
//   checks until cur >= D/4, then bound checks every boundary; switches to
//   positions-array survivor-only scan once >= 50% of the group's tokens are
//   pruned (mirrors the oracle's WARMUP_MIN_DIM / SELECTIVITY constants);
//   partials are stored token-major (P[j*m+i]) for SIMD-friendly inner loops,
//   like the oracle's throughput variant.
//
// Build: see Makefile in this directory.  Self-contained -- no PDX include
// needed at compile time.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

// float32 guard on the UB < tau pruning test.  P_ij accumulates in float32
// (error ≈ D * eps_machine per term); tau can be inflated by earlier
// same-group finalizations.  Combined worst-case: m * D * eps_mach ≈ 5e-4
// (D=128, m=32).  1e-4 covers typical cases with a 5× margin.
static constexpr float UB_EPSILON = 1e-4f;

extern "C" {

// ---------------------------------------------------------------------------
// Shared helper: min-tracked top-K of the LARGEST doc scores (identical
// pattern to cpp/per_document_oracle/per_document_oracle.cpp's TopK; kept
// self-contained here to avoid a cross-target header dependency).
// ---------------------------------------------------------------------------
struct TopK {
    std::vector<float>    score;
    std::vector<uint32_t> id;
    size_t K;
    explicit TopK(size_t k) : score(k, -std::numeric_limits<float>::infinity()),
                              id(k, 0xffffffffu), K(k) {}
    float threshold() const {
        float t = score[0];
        for (size_t i = 1; i < K; ++i) t = std::min(t, score[i]);
        return t;
    }
    void offer(float s, uint32_t v) {
        size_t mn = 0;
        for (size_t i = 1; i < K; ++i) if (score[i] < score[mn]) mn = i;
        if (s > score[mn]) { score[mn] = s; id[mn] = v; }
    }
};

static void emit_topk(const TopK& topk, size_t K, uint32_t* topk_id, float* topk_score) {
    std::vector<size_t> idx(K);
    for (size_t t = 0; t < K; ++t) idx[t] = t;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return topk.score[a] > topk.score[b]; });
    for (size_t t = 0; t < K; ++t) { topk_score[t] = topk.score[idx[t]]; topk_id[t] = topk.id[idx[t]]; }
}

// ---------------------------------------------------------------------------
// wide_block_maxsim_accounting
// Survivor-only scan from dimension 0 (true algorithmic-work signal).
// P is stored token-major: P[j*m + i] — all m partial dot-products for one
// token are contiguous (one cache line for m<=16), so the inner i-loop is
// L1-friendly.  The old i-major layout (stride G=4096) caused a cache-line
// miss per query-token per live-token; this is 2-3x faster on large corpora.
// ---------------------------------------------------------------------------
uint64_t wide_block_maxsim_accounting(
        const float* group_data, const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets, size_t n_docs,
        const uint64_t* group_doc_starts,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
        const float* Qcum, float shrink, float tau_seed, size_t K,
        uint32_t* topk_id, float* topk_score, uint64_t* stats,
        uint64_t* block_doc_live, uint64_t* block_token_live) {

    (void)n_docs;

    size_t max_G = 0;
    for (size_t g = 0; g < n_groups; ++g)
        max_G = std::max(max_G, (size_t)(group_offsets[g + 1] - group_offsets[g]));

    std::vector<float>    P(max_G * m);       // token-major: P[j*m + i]
    std::vector<float>    sumsq_t(max_G);     // Sigma_scanned t_z^2 -> residual ||t[cur:]||
    std::vector<uint32_t> live(max_G);        // in-place-compacted positions array
    std::vector<float>    Li(m), resq(m);

    TopK topk(K);
    uint64_t cells = 0, docs_pruned = 0, tokens_pruned = 0;
    const float NEG = -std::numeric_limits<float>::infinity();

    if (block_doc_live)   std::memset(block_doc_live,   0, n_fetch * sizeof(uint64_t));
    if (block_token_live) std::memset(block_token_live, 0, n_fetch * sizeof(uint64_t));

    for (size_t g = 0; g < n_groups; ++g) {
        uint64_t g0 = group_offsets[g], g1 = group_offsets[g + 1];
        size_t G = (size_t)(g1 - g0);
        if (G == 0) continue;
        const float* base = group_data + (size_t)g0 * D;
        size_t d0 = (size_t)group_doc_starts[g], d1 = (size_t)group_doc_starts[g + 1];

        for (size_t x = 0; x < m * G; ++x) P[x] = 0.0f;
        for (size_t j = 0; j < G; ++j) { sumsq_t[j] = 0.0f; live[j] = (uint32_t)j; }
        size_t n_live = G;

        size_t cur = 0, fidx = 0;
        while (cur < D && n_live > 0) {
            size_t b_idx = fidx < n_fetch ? fidx : n_fetch - 1;
            size_t blk = fetch_schedule[b_idx];
            size_t end = cur + blk < D ? cur + blk : D;
            ++fidx;

            // (1) synchronized scan of this dim-block over the group's live set
            for (size_t t = cur; t < end; ++t) {
                uint32_t z = order[t];
                const float* col = base + (size_t)z * G;
                for (size_t a = 0; a < n_live; ++a) {
                    uint32_t j = live[a];
                    float dv = col[j];
                    sumsq_t[j] += dv * dv;
                    float* Pj = P.data() + j * m;             // contiguous m values
                    for (size_t i = 0; i < m; ++i) Pj[i] += query[i * D + z] * dv;
                }
            }
            cells += (uint64_t)(end - cur) * n_live * m;
            cur = end;

            // residual norms with the k-dependent confidence ramp beta(k)
            float beta = shrink + (1.0f - shrink) * ((float)(D - cur) / (float)D);
            for (size_t i = 0; i < m; ++i) {
                float s = 1.0f - Qcum[i * (D + 1) + cur];
                resq[i] = beta * std::sqrt(s > 0.0f ? s : 0.0f);
            }

            float tau = std::max(tau_seed, topk.threshold());

            // (2) per-document token pruning + document bound, two-pointer
            // scan over live[] (sorted ascending; documents form contiguous
            // runs) intersected with each document's [loc_start, loc_end).
            size_t w = 0, a = 0;
            uint64_t doc_live_this_round = 0, token_live_this_round = 0;
            for (size_t d = d0; d < d1; ++d) {
                uint64_t loc_end = doc_offsets[d + 1] - g0;
                size_t b = a;
                while (b < n_live && live[b] < loc_end) ++b;
                if (b > a) {
                    size_t doc_w_start = w;

                    // (2a) L_i(d) = max over doc d's live j of (P_ij - resq_i*resd_j)
                    for (size_t i = 0; i < m; ++i) Li[i] = NEG;
                    for (size_t k = a; k < b; ++k) {
                        uint32_t j = live[k];
                        float s = 1.0f - sumsq_t[j];
                        float resd = std::sqrt(s > 0.0f ? s : 0.0f);
                        const float* Pj = P.data() + j * m;
                        for (size_t i = 0; i < m; ++i) {
                            float lb = Pj[i] - resq[i] * resd;
                            if (lb > Li[i]) Li[i] = lb;
                        }
                    }

                    // (2b) token pruning: drop j if U_ij < L_i(d) for every i
                    for (size_t k = a; k < b; ++k) {
                        uint32_t j = live[k];
                        float s = 1.0f - sumsq_t[j];
                        float resd = std::sqrt(s > 0.0f ? s : 0.0f);
                        const float* Pj = P.data() + j * m;
                        bool dominated = true;
                        for (size_t i = 0; i < m; ++i) {
                            if (Pj[i] + resq[i] * resd >= Li[i]) { dominated = false; break; }
                        }
                        if (!dominated) live[w++] = j;
                    }
                    tokens_pruned += (b - a) - (w - doc_w_start);

                    // (3) document upper bound over this document's survivors
                    float UB = 0.0f;
                    for (size_t i = 0; i < m; ++i) {
                        float mx = NEG;
                        for (size_t k = doc_w_start; k < w; ++k) {
                            uint32_t j = live[k];
                            float s = 1.0f - sumsq_t[j];
                            float resd = std::sqrt(s > 0.0f ? s : 0.0f);
                            float ub = P[j * m + i] + resq[i] * resd;
                            if (ub > mx) mx = ub;
                        }
                        UB += mx;
                    }

                    if (UB + UB_EPSILON < tau) {
                        docs_pruned++;
                        w = doc_w_start;   // remove this document's tokens entirely
                    } else {
                        doc_live_this_round++;
                        token_live_this_round += (w - doc_w_start);
                        if (cur == D) {
                            // finalize: residual is zero, score = sum_i max_j P_ij (exact)
                            float score = 0.0f;
                            for (size_t i = 0; i < m; ++i) {
                                float mx = NEG;
                                for (size_t k = doc_w_start; k < w; ++k) {
                                    float v = P[live[k] * m + i];
                                    if (v > mx) mx = v;
                                }
                                score += mx;
                            }
                            topk.offer(score, (uint32_t)d);
                        }
                    }
                }
                a = b;
            }
            n_live = w;

            if (block_doc_live)   block_doc_live[b_idx]   += doc_live_this_round;
            if (block_token_live) block_token_live[b_idx] += token_live_this_round;
        }
    }

    emit_topk(topk, K, topk_id, topk_score);
    stats[0] = cells; stats[1] = docs_pruned; stats[2] = tokens_pruned;
    return cells;
}

// ---------------------------------------------------------------------------
// wide_block_maxsim_throughput
// Dense group-wide warmup (no bound checks) until cur >= D/4, then bound
// checks every fetch boundary; switches to positional (survivor-only) scan
// once >= 50% of the group's tokens have been pruned.  P is token-major
// (P[j*m + i]) for SIMD-friendly inner loops, like the oracle's throughput
// kernel.  doc_pruned[] gates the per-document bound loop so an already
// doc-pruned document is skipped even while the dense phase keeps rescanning
// its (unused) token positions -- mirrors the oracle's per-document `break`,
// generalized to many documents sharing one dense scan.
// ---------------------------------------------------------------------------
uint64_t wide_block_maxsim_throughput(
        const float* group_data, const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets, size_t n_docs,
        const uint64_t* group_doc_starts,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
        const float* Qcum, float shrink, float tau_seed, size_t K,
        uint32_t* topk_id, float* topk_score, uint64_t* stats,
        uint64_t* block_doc_live, uint64_t* block_token_live) {

    size_t max_G = 0;
    for (size_t g = 0; g < n_groups; ++g)
        max_G = std::max(max_G, (size_t)(group_offsets[g + 1] - group_offsets[g]));

    std::vector<float>    P(m * max_G);           // token-major: P[j*m + i]
    std::vector<float>    sumsq_t(max_G);
    std::vector<float>    resd(max_G);             // cached per-round (like the oracle)
    std::vector<uint32_t> pos(max_G);               // survivor positions array
    std::vector<uint8_t>  doc_pruned(std::max<size_t>(n_docs, 1), 0);
    std::vector<float>    Li(m), resq(m), qz(m);

    TopK topk(K);
    uint64_t cells = 0, docs_pruned = 0, tokens_pruned = 0;
    const float NEG = -std::numeric_limits<float>::infinity();
    const float  SELECTIVITY    = 0.5f;
    const size_t WARMUP_MIN_DIM = D / 4;

    if (block_doc_live)   std::memset(block_doc_live,   0, n_fetch * sizeof(uint64_t));
    if (block_token_live) std::memset(block_token_live, 0, n_fetch * sizeof(uint64_t));

    for (size_t g = 0; g < n_groups; ++g) {
        uint64_t g0 = group_offsets[g], g1 = group_offsets[g + 1];
        size_t G = (size_t)(g1 - g0);
        if (G == 0) continue;
        const float* base = group_data + (size_t)g0 * D;
        size_t d0 = (size_t)group_doc_starts[g], d1 = (size_t)group_doc_starts[g + 1];

        std::fill_n(P.data(), G * m, 0.0f);
        for (size_t j = 0; j < G; ++j) sumsq_t[j] = 0.0f;
        for (size_t d = d0; d < d1; ++d) doc_pruned[d] = 0;

        size_t n_live = G;      // meaningful once pos[] is built (post-warmup)
        bool   positional = false;
        size_t cur = 0, fidx = 0;

        while (cur < D) {
            size_t b_idx = fidx < n_fetch ? fidx : n_fetch - 1;
            size_t blk = fetch_schedule[b_idx];
            size_t end = cur + blk < D ? cur + blk : D;
            ++fidx;

            if (!positional) {
                for (size_t t = cur; t < end; ++t) {
                    uint32_t z = order[t];
                    const float* col = base + (size_t)z * G;
                    for (size_t i = 0; i < m; ++i) qz[i] = query[i * D + z];
                    for (size_t j = 0; j < G; ++j) {
                        float dv = col[j]; sumsq_t[j] += dv * dv;
                        float* Pj = P.data() + j * m;
                        for (size_t i = 0; i < m; ++i) Pj[i] += qz[i] * dv;
                    }
                }
                cells += (uint64_t)(end - cur) * G * m;
            } else {
                for (size_t t = cur; t < end; ++t) {
                    uint32_t z = order[t];
                    const float* col = base + (size_t)z * G;
                    for (size_t i = 0; i < m; ++i) qz[i] = query[i * D + z];
                    for (size_t a = 0; a < n_live; ++a) {
                        uint32_t j = pos[a];
                        float dv = col[j]; sumsq_t[j] += dv * dv;
                        float* Pj = P.data() + (size_t)j * m;
                        for (size_t i = 0; i < m; ++i) Pj[i] += qz[i] * dv;
                    }
                }
                cells += (uint64_t)(end - cur) * n_live * m;
            }
            cur = end;

            // skip bound eval in the early no-prune warmup window
            if (cur < WARMUP_MIN_DIM && !positional) continue;

            float beta = shrink + (1.0f - shrink) * ((float)(D - cur) / (float)D);
            for (size_t i = 0; i < m; ++i) {
                float s = 1.0f - Qcum[i * (D + 1) + cur];
                resq[i] = beta * std::sqrt(s > 0.0f ? s : 0.0f);
            }

            float tau = std::max(tau_seed, topk.threshold());
            uint64_t doc_live_this_round = 0, token_live_this_round = 0;

            auto eval_token = [&](uint32_t j) {
                float s = 1.0f - sumsq_t[j]; resd[j] = std::sqrt(s > 0.0f ? s : 0.0f);
                const float* Pj = P.data() + (size_t)j * m; float rd = resd[j];
                for (size_t i = 0; i < m; ++i) { float lb = Pj[i] - resq[i] * rd; if (lb > Li[i]) Li[i] = lb; }
            };
            auto survives = [&](uint32_t j) {
                const float* Pj = P.data() + (size_t)j * m; float rd = resd[j];
                for (size_t i = 0; i < m; ++i) if (Pj[i] + resq[i] * rd >= Li[i]) return true;
                return false;
            };

            if (!positional) {
                size_t nl = 0;
                for (size_t d = d0; d < d1; ++d) {
                    if (doc_pruned[d]) continue;
                    uint64_t ls = doc_offsets[d] - g0, le = doc_offsets[d + 1] - g0;
                    if (le <= ls) continue;
                    for (size_t i = 0; i < m; ++i) Li[i] = NEG;
                    for (size_t j = ls; j < le; ++j) eval_token((uint32_t)j);

                    size_t doc_w_start = nl;
                    for (size_t j = ls; j < le; ++j) if (survives((uint32_t)j)) pos[nl++] = (uint32_t)j;
                    tokens_pruned += (le - ls) - (nl - doc_w_start);

                    float UB = 0.0f;
                    for (size_t i = 0; i < m; ++i) {
                        float mx = NEG;
                        for (size_t k = doc_w_start; k < nl; ++k) {
                            uint32_t j = pos[k];
                            float ub = P[(size_t)j * m + i] + resq[i] * resd[j];
                            if (ub > mx) mx = ub;
                        }
                        UB += mx;
                    }
                    if (UB + UB_EPSILON < tau) {
                        docs_pruned++; doc_pruned[d] = 1; nl = doc_w_start;
                    } else {
                        doc_live_this_round++;
                        token_live_this_round += (nl - doc_w_start);
                        if (cur == D) {
                            float score = 0.0f;
                            for (size_t i = 0; i < m; ++i) {
                                float mx = NEG;
                                for (size_t k = doc_w_start; k < nl; ++k) {
                                    float v = P[(size_t)pos[k] * m + i];
                                    if (v > mx) mx = v;
                                }
                                score += mx;
                            }
                            topk.offer(score, (uint32_t)d);
                            doc_pruned[d] = 1;
                        }
                    }
                }
                n_live = nl;
                if ((G - n_live) >= (size_t)(SELECTIVITY * (float)G)) positional = true;
            } else {
                size_t w = 0, a = 0;
                for (size_t d = d0; d < d1; ++d) {
                    if (doc_pruned[d]) continue;
                    uint64_t le = doc_offsets[d + 1] - g0;
                    size_t b = a;
                    while (b < n_live && pos[b] < le) ++b;
                    if (b > a) {
                        size_t doc_w_start = w;
                        for (size_t i = 0; i < m; ++i) Li[i] = NEG;
                        for (size_t k = a; k < b; ++k) eval_token(pos[k]);
                        for (size_t k = a; k < b; ++k) if (survives(pos[k])) pos[w++] = pos[k];
                        tokens_pruned += (b - a) - (w - doc_w_start);

                        float UB = 0.0f;
                        for (size_t i = 0; i < m; ++i) {
                            float mx = NEG;
                            for (size_t k = doc_w_start; k < w; ++k) {
                                uint32_t j = pos[k];
                                float ub = P[(size_t)j * m + i] + resq[i] * resd[j];
                                if (ub > mx) mx = ub;
                            }
                            UB += mx;
                        }
                        if (UB + UB_EPSILON < tau) {
                            docs_pruned++; doc_pruned[d] = 1; w = doc_w_start;
                        } else {
                            doc_live_this_round++;
                            token_live_this_round += (w - doc_w_start);
                            if (cur == D) {
                                float score = 0.0f;
                                for (size_t i = 0; i < m; ++i) {
                                    float mx = NEG;
                                    for (size_t k = doc_w_start; k < w; ++k) {
                                        float v = P[(size_t)pos[k] * m + i];
                                        if (v > mx) mx = v;
                                    }
                                    score += mx;
                                }
                                topk.offer(score, (uint32_t)d);
                                doc_pruned[d] = 1;
                            }
                        }
                    }
                    a = b;
                }
                n_live = w;
            }

            if (block_doc_live)   block_doc_live[b_idx]   += doc_live_this_round;
            if (block_token_live) block_token_live[b_idx] += token_live_this_round;

            if (n_live == 0) break;
        }
    }

    emit_topk(topk, K, topk_id, topk_score);
    stats[0] = cells; stats[1] = docs_pruned; stats[2] = tokens_pruned;
    return cells;
}

}  // extern "C"
