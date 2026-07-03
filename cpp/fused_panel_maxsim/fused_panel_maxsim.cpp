// fused_panel_maxsim.cpp — fused panel MaxSim kernels (Stage 3b deliverable).
//
// Register-tiled dense MaxSim scan over the panel-major PDX layout, with the
// per-document max-reduction fused into the accumulator tile.  Design,
// theory, and measured motivation: docs/stage3b_fused_panel_maxsim_kernel.md.
// In short: the panel-major layout is the BLAS packed-B micro-panel format
// packed once at index-build time (§3.1), and fusing the segmented max into
// the epilogue means the m x T score matrix is never materialized (§3.2) —
// which is why this kernel beats `Q @ X.T` + reduceat despite BLAS.
//
// Layout (see src/bondmaxsim/data/packing.py::pack_corpus_panels):
//   Documents are padded to a multiple of PT=16 tokens by DUPLICATING their
//   last token (max-invariant; zeros would corrupt negative maxima, §5.3),
//   then stored in consecutive 16-token panels, dim-major within a panel:
//       panel_data[tok_base*D + z*PT + j],  tok_base = padded global offset.
//   Every document therefore covers WHOLE panels — the epilogue never needs
//   per-lane document masks.  doc_offsets are PADDED cumulative token counts.
//   Groups (consecutive whole documents, ~4096 padded tokens) exist as the
//   threading and (future BOND variant) pruning granularity.
//
// Query tiling (§5.1, §5.4): m query tokens are split into ceil(m/24) tiles,
// each zero-padded to the smallest M in {8,16,24} >= its size.  Zero-padded
// query rows accumulate exactly 0 and contribute max_j 0 = 0 to the score —
// exact.  Tiles iterate inside the panel loop, so the 8 KB panel stays
// L1-resident across tiles.  M=24 uses 24 of 32 zmm registers for the
// accumulator tile; hence 24, not 32, is the largest tile.
//
// Threading: OpenMP over groups (independent); per-thread TopK merged at the
// end.  n_threads <= 0 means the OpenMP default; the .so also builds and
// runs correctly without OpenMP (pragmas ignored, serial).
//
// ABI (extern "C"):
//   uint64_t fused_panel_maxsim_brute(
//       const float* panel_data,
//       const uint64_t* group_offsets, size_t n_groups,
//       const uint64_t* doc_offsets,          // PADDED, len n_docs+1
//       const uint64_t* group_doc_starts,     // len n_groups+1
//       const float* query, size_t m, size_t D,
//       size_t K, int n_threads,
//       uint32_t* topk_id, float* topk_score);
//
// Build: make -C cpp/fused_panel_maxsim  (adds -fopenmp; AVX-512 fast path
// under __AVX512F__, portable autovectorized fallback otherwise).

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static constexpr size_t PT = 16;          // panel width (tokens)
static constexpr size_t TILE_MAX = 24;    // largest query tile (register budget)

// float32 guard on the per-document UB < tau pruning test (same constant and
// rationale as cpp/wide_block_maxsim_bond).
static constexpr float UB_EPSILON = 1e-4f;

// ---------------------------------------------------------------------------
// TopK — identical pattern to cpp/wide_block_maxsim_bond (kept self-contained).
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

// Monotonically rising shared pruning threshold.  A thread's local top-K
// threshold is the k-th best over a SUBSET of documents, hence <= the final
// global k-th score, so publishing it can only make tau safer-or-equal —
// a stale read prunes less, never incorrectly (Stage 3b §5.6).
static inline void atomic_max_tau(std::atomic<float>& a, float v) {
    float cur = a.load(std::memory_order_relaxed);
    while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
}

static void emit_topk(const TopK& topk, size_t K, uint32_t* topk_id, float* topk_score) {
    std::vector<size_t> idx(K);
    for (size_t t = 0; t < K; ++t) idx[t] = t;
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return topk.score[a] > topk.score[b]; });
    for (size_t t = 0; t < K; ++t) { topk_score[t] = topk.score[idx[t]]; topk_id[t] = topk.id[idx[t]]; }
}

// ---------------------------------------------------------------------------
// Microkernel: one panel x one query tile, accumulated over all D dims, then
// folded into the per-document running max (fused epilogue, §5.4).
// ---------------------------------------------------------------------------

#ifdef __AVX512F__

template <size_t M>
static inline void panel_tile(const float* panel, const float* qpack, size_t D,
                              __m512* dmax) {
    __m512 acc[M];
    for (size_t i = 0; i < M; ++i) acc[i] = _mm512_setzero_ps();
    for (size_t z = 0; z < D; ++z) {
        __m512 col = _mm512_loadu_ps(panel + z * PT);
        const float* qz = qpack + z * M;
        for (size_t i = 0; i < M; ++i)
            acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(qz[i]), col, acc[i]);
    }
    for (size_t i = 0; i < M; ++i) dmax[i] = _mm512_max_ps(dmax[i], acc[i]);
}

struct DocMax {
    // Per query-tile running per-doc max vectors (16 lanes each).
    __m512 v[TILE_MAX];
    void reset(size_t M) {
        for (size_t i = 0; i < M; ++i) v[i] = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    }
    float reduce_sum(size_t m_real) const {
        float s = 0.0f;
        for (size_t i = 0; i < m_real; ++i) s += _mm512_reduce_max_ps(v[i]);
        return s;
    }
};

static inline void run_panel(const float* panel, const float* qpack, size_t M,
                             size_t D, DocMax& dm) {
    switch (M) {
        case 8:  panel_tile<8> (panel, qpack, D, dm.v); break;
        case 16: panel_tile<16>(panel, qpack, D, dm.v); break;
        default: panel_tile<24>(panel, qpack, D, dm.v); break;
    }
}

#else  // portable fallback: fixed-trip loops over a PT-wide lane array

template <size_t M>
static inline void panel_tile(const float* panel, const float* qpack, size_t D,
                              float (*dmax)[PT]) {
    float acc[M][PT];
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) acc[i][j] = 0.0f;
    for (size_t z = 0; z < D; ++z) {
        const float* col = panel + z * PT;
        const float* qz = qpack + z * M;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < PT; ++j) acc[i][j] += qz[i] * col[j];
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) dmax[i][j] = std::max(dmax[i][j], acc[i][j]);
}

struct DocMax {
    float v[TILE_MAX][PT];
    void reset(size_t M) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < PT; ++j) v[i][j] = -std::numeric_limits<float>::infinity();
    }
    float reduce_sum(size_t m_real) const {
        float s = 0.0f;
        for (size_t i = 0; i < m_real; ++i) {
            float mx = v[i][0];
            for (size_t j = 1; j < PT; ++j) mx = std::max(mx, v[i][j]);
            s += mx;
        }
        return s;
    }
};

static inline void run_panel(const float* panel, const float* qpack, size_t M,
                             size_t D, DocMax& dm) {
    switch (M) {
        case 8:  panel_tile<8> (panel, qpack, D, dm.v); break;
        case 16: panel_tile<16>(panel, qpack, D, dm.v); break;
        default: panel_tile<24>(panel, qpack, D, dm.v); break;
    }
}

#endif  // __AVX512F__

// Query tile descriptor built once per query.
struct QueryTiles {
    size_t n_tiles = 0;
    size_t M[8];        // padded tile height (8/16/24)
    size_t m_real[8];   // real query tokens in this tile
    std::vector<float> qpack;   // concatenated per-tile dim-major packs
    size_t pack_off[8];

    QueryTiles(const float* query, size_t m, size_t D) {
        size_t off = 0;
        for (size_t t0 = 0; t0 < m; t0 += TILE_MAX) {
            size_t t = n_tiles++;
            size_t mt = std::min(TILE_MAX, m - t0);
            size_t Mt = mt <= 8 ? 8 : (mt <= 16 ? 16 : 24);
            M[t] = Mt; m_real[t] = mt; pack_off[t] = off;
            off += Mt * D;
        }
        qpack.assign(off, 0.0f);
        for (size_t t = 0; t < n_tiles; ++t) {
            size_t t0 = t * TILE_MAX;
            float* dst = qpack.data() + pack_off[t];
            for (size_t z = 0; z < D; ++z)
                for (size_t i = 0; i < m_real[t]; ++i)
                    dst[z * M[t] + i] = query[(t0 + i) * D + z];
        }
    }
};

extern "C" {

uint64_t fused_panel_maxsim_brute(
        const float* panel_data,
        const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets,
        const uint64_t* group_doc_starts,
        const float* query, size_t m, size_t D,
        size_t K, int n_threads,
        uint32_t* topk_id, float* topk_score) {

    (void)group_offsets;
    QueryTiles qt(query, m, D);
    TopK global(K);

#ifdef _OPENMP
    int nt = n_threads > 0 ? n_threads : omp_get_max_threads();
#else
    int nt = 1; (void)n_threads;
#endif

    #pragma omp parallel num_threads(nt)
    {
        TopK local(K);
        DocMax dm[8];

        #pragma omp for schedule(dynamic)
        for (size_t g = 0; g < n_groups; ++g) {
            size_t d0 = (size_t)group_doc_starts[g], d1 = (size_t)group_doc_starts[g + 1];
            for (size_t d = d0; d < d1; ++d) {
                size_t tok0 = (size_t)doc_offsets[d], tok1 = (size_t)doc_offsets[d + 1];
                if (tok1 <= tok0) continue;                 // empty doc: never offered
                for (size_t t = 0; t < qt.n_tiles; ++t) dm[t].reset(qt.M[t]);
                for (size_t p = tok0; p < tok1; p += PT) {
                    const float* panel = panel_data + p * D;
                    for (size_t t = 0; t < qt.n_tiles; ++t)   // panel stays L1-resident
                        run_panel(panel, qt.qpack.data() + qt.pack_off[t], qt.M[t], D, dm[t]);
                }
                float score = 0.0f;
                for (size_t t = 0; t < qt.n_tiles; ++t) score += dm[t].reduce_sum(qt.m_real[t]);
                local.offer(score, (uint32_t)d);
            }
        }

        #pragma omp critical
        for (size_t i = 0; i < K; ++i)
            if (local.id[i] != 0xffffffffu) global.offer(local.score[i], local.id[i]);
    }

    emit_topk(global, K, topk_id, topk_score);
    return 0;
}

}  // extern "C" (brute)

// ---------------------------------------------------------------------------
// fused_panel_maxsim_bond — same microkernel, dimension-incremental with
// per-DOCUMENT Cauchy-Schwarz bound checkpoints (Stage 3b §5.5).
//
// The scan of one document is split into dimension segments ending at the
// checkpoints {32, 64, D} (clamped/deduped for small D; aligned with the
// DEFAULT_FETCH cumulative boundaries).  Within a segment the register-tiled
// microkernel runs unchanged over the document's panels, spilling partials
// to a small L1/L2-resident per-document buffer between segments.  At each
// non-final checkpoint the document upper bound
//     UB_d = sum_i max_j (P_ij + resq_i * resd_j)
// is evaluated lane-vectorized over the spilled partials; if UB_d < tau the
// document's remaining dimension segments are skipped entirely (bytes never
// read — this is the mechanism that goes below the dense DRAM floor).
// Token-level domination pruning is deliberately absent (e03 showed its
// bookkeeping costs more than it saves at wall-clock; it remains measured by
// the wide-block ACCOUNTING kernel).
//
// Exactness at shrink=1: identical bound math to Stage 1 §2.3 evaluated at
// document granularity; duplicate-token padding adds identical lanes (max-
// invariant); zero-padded query rows are excluded from UB and score.
//
// tau = max(tau_seed, shared_tau) where shared_tau is the atomic max over
// all threads' local top-K thresholds (safe: see atomic_max_tau).
//
// stats[0] = cells actually scanned (dims x PADDED tokens x m real query
//            tokens; wall-clock convention, NOT comparable to the accounting
//            kernel's unpadded live-set counter)
// stats[1] = docs pruned at a checkpoint (never finalized)
// ---------------------------------------------------------------------------

#ifdef __AVX512F__

// One panel x one tile over dims order[z0:z1); partials spilled to Pt[i*PT].
template <size_t M>
static inline void panel_tile_seg(const float* panel, const float* qpack,
                                  const uint32_t* order, size_t z0, size_t z1,
                                  float* Pt, bool first) {
    __m512 acc[M];
    if (first) for (size_t i = 0; i < M; ++i) acc[i] = _mm512_setzero_ps();
    else       for (size_t i = 0; i < M; ++i) acc[i] = _mm512_loadu_ps(Pt + i * PT);
    for (size_t t = z0; t < z1; ++t) {
        size_t z = order[t];
        __m512 col = _mm512_loadu_ps(panel + z * PT);
        const float* qz = qpack + z * M;
        for (size_t i = 0; i < M; ++i)
            acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(qz[i]), col, acc[i]);
    }
    for (size_t i = 0; i < M; ++i) _mm512_storeu_ps(Pt + i * PT, acc[i]);
}

static inline void panel_sumsq_seg(const float* panel, const uint32_t* order,
                                   size_t z0, size_t z1, float* ss, bool first) {
    __m512 s = first ? _mm512_setzero_ps() : _mm512_loadu_ps(ss);
    for (size_t t = z0; t < z1; ++t) {
        __m512 col = _mm512_loadu_ps(panel + (size_t)order[t] * PT);
        s = _mm512_fmadd_ps(col, col, s);
    }
    _mm512_storeu_ps(ss, s);
}

// resd = sqrt(max(0, 1 - sumsq)) per lane, one panel.
static inline void panel_resd(const float* ss, float* resd) {
    __m512 s = _mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_loadu_ps(ss));
    _mm512_storeu_ps(resd, _mm512_sqrt_ps(_mm512_max_ps(s, _mm512_setzero_ps())));
}

// max over the document's lanes of (P + resq*resd) for one query row.
static inline float doc_row_ubmax(const float* Pt_row, const float* resd,
                                  size_t n_panels, size_t row_stride, float resq) {
    __m512 vq = _mm512_set1_ps(resq);
    __m512 best = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t p = 0; p < n_panels; ++p) {
        __m512 P = _mm512_loadu_ps(Pt_row + p * row_stride);
        __m512 r = _mm512_loadu_ps(resd + p * PT);
        best = _mm512_max_ps(best, _mm512_fmadd_ps(vq, r, P));
    }
    return _mm512_reduce_max_ps(best);
}

#else  // portable fallback

template <size_t M>
static inline void panel_tile_seg(const float* panel, const float* qpack,
                                  const uint32_t* order, size_t z0, size_t z1,
                                  float* Pt, bool first) {
    float acc[M][PT];
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) acc[i][j] = first ? 0.0f : Pt[i * PT + j];
    for (size_t t = z0; t < z1; ++t) {
        const float* col = panel + (size_t)order[t] * PT;
        const float* qz = qpack + (size_t)order[t] * M;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < PT; ++j) acc[i][j] += qz[i] * col[j];
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) Pt[i * PT + j] = acc[i][j];
}

static inline void panel_sumsq_seg(const float* panel, const uint32_t* order,
                                   size_t z0, size_t z1, float* ss, bool first) {
    float s[PT];
    for (size_t j = 0; j < PT; ++j) s[j] = first ? 0.0f : ss[j];
    for (size_t t = z0; t < z1; ++t) {
        const float* col = panel + (size_t)order[t] * PT;
        for (size_t j = 0; j < PT; ++j) s[j] += col[j] * col[j];
    }
    for (size_t j = 0; j < PT; ++j) ss[j] = s[j];
}

static inline void panel_resd(const float* ss, float* resd) {
    for (size_t j = 0; j < PT; ++j) {
        float s = 1.0f - ss[j];
        resd[j] = std::sqrt(s > 0.0f ? s : 0.0f);
    }
}

static inline float doc_row_ubmax(const float* Pt_row, const float* resd,
                                  size_t n_panels, size_t row_stride, float resq) {
    float best = -std::numeric_limits<float>::infinity();
    for (size_t p = 0; p < n_panels; ++p)
        for (size_t j = 0; j < PT; ++j)
            best = std::max(best, Pt_row[p * row_stride + j] + resq * resd[p * PT + j]);
    return best;
}

#endif  // __AVX512F__

static inline void run_panel_seg(const float* panel, const float* qpack, size_t M,
                                 const uint32_t* order, size_t z0, size_t z1,
                                 float* Pt, bool first) {
    switch (M) {
        case 8:  panel_tile_seg<8> (panel, qpack, order, z0, z1, Pt, first); break;
        case 16: panel_tile_seg<16>(panel, qpack, order, z0, z1, Pt, first); break;
        default: panel_tile_seg<24>(panel, qpack, order, z0, z1, Pt, first); break;
    }
}

extern "C" {

uint64_t fused_panel_maxsim_bond(
        const float* panel_data,
        const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets,
        const uint64_t* group_doc_starts,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const float* Qcum,
        float shrink, float tau_seed, size_t K, int n_threads,
        uint32_t* topk_id, float* topk_score, uint64_t* stats) {

    (void)group_offsets;
    QueryTiles qt(query, m, D);
    TopK global(K);
    std::atomic<float> shared_tau{-std::numeric_limits<float>::infinity()};
    std::atomic<uint64_t> cells{0}, docs_pruned{0};

    // Checkpoints {32, 64, D}, clamped and deduped for small D.
    size_t cps[3]; size_t n_cps = 0;
    for (size_t c : {std::min<size_t>(32, D), std::min<size_t>(64, D), D})
        if (n_cps == 0 || c > cps[n_cps - 1]) cps[n_cps++] = c;

    // Max padded document length (per-thread spill buffer size).
    size_t max_doc_tok = 0;
    size_t n_docs_total = n_groups > 0 ? (size_t)group_doc_starts[n_groups] : 0;
    for (size_t d = 0; d < n_docs_total; ++d)
        max_doc_tok = std::max(max_doc_tok, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));
    size_t max_panels = max_doc_tok / PT;

#ifdef _OPENMP
    int nt = n_threads > 0 ? n_threads : omp_get_max_threads();
#else
    int nt = 1; (void)n_threads;
#endif

    #pragma omp parallel num_threads(nt)
    {
        TopK local(K);
        // Spilled partials: per tile, per panel, per query row: [i*PT] lanes.
        // Row stride across panels within a tile = M*PT.
        std::vector<std::vector<float>> Pt(qt.n_tiles);
        for (size_t t = 0; t < qt.n_tiles; ++t) Pt[t].assign(max_panels * qt.M[t] * PT, 0.0f);
        std::vector<float> ss(max_panels * PT), resd(max_panels * PT), resq(m);
        uint64_t l_cells = 0, l_pruned = 0;

        #pragma omp for schedule(dynamic)
        for (size_t g = 0; g < n_groups; ++g) {
            size_t d0 = (size_t)group_doc_starts[g], d1 = (size_t)group_doc_starts[g + 1];
            for (size_t d = d0; d < d1; ++d) {
                size_t tok0 = (size_t)doc_offsets[d], tok1 = (size_t)doc_offsets[d + 1];
                size_t n_tok = tok1 - tok0;
                if (n_tok == 0) continue;
                size_t n_panels = n_tok / PT;

                bool pruned = false;
                size_t cur = 0;
                for (size_t c = 0; c < n_cps && !pruned; ++c) {
                    size_t end = cps[c];
                    for (size_t p = 0; p < n_panels; ++p) {
                        const float* panel = panel_data + (tok0 + p * PT) * D;
                        panel_sumsq_seg(panel, order, cur, end, ss.data() + p * PT, cur == 0);
                        for (size_t t = 0; t < qt.n_tiles; ++t)
                            run_panel_seg(panel, qt.qpack.data() + qt.pack_off[t], qt.M[t],
                                          order, cur, end,
                                          Pt[t].data() + p * qt.M[t] * PT, cur == 0);
                    }
                    l_cells += (uint64_t)(end - cur) * n_tok * m;
                    cur = end;

                    if (cur == D) break;   // final segment: no bound needed, finalize below

                    // Document upper bound at this checkpoint.
                    float beta = shrink + (1.0f - shrink) * ((float)(D - cur) / (float)D);
                    for (size_t i = 0; i < m; ++i) {
                        float s = 1.0f - Qcum[i * (D + 1) + cur];
                        resq[i] = beta * std::sqrt(s > 0.0f ? s : 0.0f);
                    }
                    for (size_t p = 0; p < n_panels; ++p)
                        panel_resd(ss.data() + p * PT, resd.data() + p * PT);

                    float tau = std::max(tau_seed, shared_tau.load(std::memory_order_relaxed));
                    float UB = 0.0f;
                    for (size_t t = 0; t < qt.n_tiles; ++t) {
                        size_t row_stride = qt.M[t] * PT;
                        for (size_t i = 0; i < qt.m_real[t]; ++i)
                            UB += doc_row_ubmax(Pt[t].data() + i * PT, resd.data(),
                                                n_panels, row_stride,
                                                resq[t * TILE_MAX + i]);
                    }
                    if (UB + UB_EPSILON < tau) { pruned = true; ++l_pruned; }
                }

                if (!pruned) {
                    // Residual is zero at cur == D: exact score from partials.
                    float score = 0.0f;
                    for (size_t t = 0; t < qt.n_tiles; ++t) {
                        size_t row_stride = qt.M[t] * PT;
                        for (size_t i = 0; i < qt.m_real[t]; ++i)
                            score += doc_row_ubmax(Pt[t].data() + i * PT, resd.data(),
                                                   n_panels, row_stride, 0.0f);
                    }
                    local.offer(score, (uint32_t)d);
                    atomic_max_tau(shared_tau, local.threshold());
                }
            }
        }

        cells.fetch_add(l_cells, std::memory_order_relaxed);
        docs_pruned.fetch_add(l_pruned, std::memory_order_relaxed);

        #pragma omp critical
        for (size_t i = 0; i < K; ++i)
            if (local.id[i] != 0xffffffffu) global.offer(local.score[i], local.id[i]);
    }

    emit_topk(global, K, topk_id, topk_score);
    if (stats) { stats[0] = cells.load(); stats[1] = docs_pruned.load(); }
    return stats ? stats[0] : 0;
}

}  // extern "C" (bond, document-level)

// ---------------------------------------------------------------------------
// fused_panel_maxsim_bond_token — the TOKEN-level pruning arm.
//
// Identical to fused_panel_maxsim_bond in every scheduling respect (same
// microkernel, same panel layout, same checkpoints {32, 64}, same shared τ,
// same doc-at-a-time order) with EXACTLY ONE mechanism added: the Stage 1
// §2.4 token-level domination test at each checkpoint.  A token j of doc d
// is dropped iff  P_ij + resq_i·resd_j < L_i(d)  for EVERY query token i,
// where  L_i(d) = max over d's live tokens of (P_ij - resq_i·resd_j).
// Dropped tokens leave the Li/UB/score maxima (bound tightening) and, once
// ALL 16 lanes of a panel are dead, the panel's remaining dimension segments
// are skipped entirely (the compute saving — token pruning realizes FMA
// savings at panel granularity because lanes are SIMD rows).
//
// Purpose (three-arm isolation, requested 2026-07-03): dense / +doc-pruning
// / +token-pruning on the SAME kernel, so any difference is attributable to
// the pruning mechanism alone.  Contrast with PDX-sigmod BOND (pdxearch.hpp
// EvaluatePruningPredicate*): for L2 the partial distance is a monotone
// zero-slack lower bound and the predicate is ONE comparison per vector;
// MaxSim needs the Cauchy-Schwarz residual envelope, which both costs work
// and carries structural slack.
//
// stats[0] = cells (physical: 16 lanes per LIVE panel × dims × m real rows)
// stats[1] = docs pruned at a checkpoint
// stats[2] = token lanes pruned by domination (incl. padding lanes; padding
//            lanes duplicate a real token so their live/dead state always
//            matches their original's)
// ---------------------------------------------------------------------------

#ifdef __AVX512F__

// Masked variants of the per-doc reductions (live lanes only).
static inline float doc_row_ubmax_masked(const float* Pt_row, const float* resd,
                                         const uint16_t* live, size_t n_panels,
                                         size_t row_stride, float resq) {
    __m512 vq = _mm512_set1_ps(resq);
    __m512 best = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t p = 0; p < n_panels; ++p) {
        __mmask16 k = (__mmask16)live[p];
        if (!k) continue;
        __m512 P = _mm512_loadu_ps(Pt_row + p * row_stride);
        __m512 r = _mm512_loadu_ps(resd + p * PT);
        best = _mm512_mask_max_ps(best, k, best, _mm512_fmadd_ps(vq, r, P));
    }
    return _mm512_reduce_max_ps(best);
}

static inline float doc_row_lbmax_masked(const float* Pt_row, const float* resd,
                                         const uint16_t* live, size_t n_panels,
                                         size_t row_stride, float resq) {
    __m512 vq = _mm512_set1_ps(resq);
    __m512 best = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t p = 0; p < n_panels; ++p) {
        __mmask16 k = (__mmask16)live[p];
        if (!k) continue;
        __m512 P = _mm512_loadu_ps(Pt_row + p * row_stride);
        __m512 r = _mm512_loadu_ps(resd + p * PT);
        best = _mm512_mask_max_ps(best, k, best, _mm512_fnmadd_ps(vq, r, P));  // P - resq*resd
    }
    return _mm512_reduce_max_ps(best);
}

#else

static inline float doc_row_ubmax_masked(const float* Pt_row, const float* resd,
                                         const uint16_t* live, size_t n_panels,
                                         size_t row_stride, float resq) {
    float best = -std::numeric_limits<float>::infinity();
    for (size_t p = 0; p < n_panels; ++p)
        for (size_t j = 0; j < PT; ++j)
            if (live[p] & (1u << j))
                best = std::max(best, Pt_row[p * row_stride + j] + resq * resd[p * PT + j]);
    return best;
}

static inline float doc_row_lbmax_masked(const float* Pt_row, const float* resd,
                                         const uint16_t* live, size_t n_panels,
                                         size_t row_stride, float resq) {
    float best = -std::numeric_limits<float>::infinity();
    for (size_t p = 0; p < n_panels; ++p)
        for (size_t j = 0; j < PT; ++j)
            if (live[p] & (1u << j))
                best = std::max(best, Pt_row[p * row_stride + j] - resq * resd[p * PT + j]);
    return best;
}

#endif  // __AVX512F__

extern "C" {

uint64_t fused_panel_maxsim_bond_token(
        const float* panel_data,
        const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets,
        const uint64_t* group_doc_starts,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const float* Qcum,
        float shrink, float tau_seed, size_t K, int n_threads,
        uint32_t* topk_id, float* topk_score, uint64_t* stats) {

    (void)group_offsets;
    QueryTiles qt(query, m, D);
    TopK global(K);
    std::atomic<float> shared_tau{-std::numeric_limits<float>::infinity()};
    std::atomic<uint64_t> cells{0}, docs_pruned{0}, tokens_pruned{0};

    size_t cps[3]; size_t n_cps = 0;
    for (size_t c : {std::min<size_t>(32, D), std::min<size_t>(64, D), D})
        if (n_cps == 0 || c > cps[n_cps - 1]) cps[n_cps++] = c;

    size_t max_doc_tok = 0;
    size_t n_docs_total = n_groups > 0 ? (size_t)group_doc_starts[n_groups] : 0;
    for (size_t d = 0; d < n_docs_total; ++d)
        max_doc_tok = std::max(max_doc_tok, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));
    size_t max_panels = max_doc_tok / PT;

#ifdef _OPENMP
    int nt = n_threads > 0 ? n_threads : omp_get_max_threads();
#else
    int nt = 1; (void)n_threads;
#endif

    #pragma omp parallel num_threads(nt)
    {
        TopK local(K);
        std::vector<std::vector<float>> Pt(qt.n_tiles);
        for (size_t t = 0; t < qt.n_tiles; ++t) Pt[t].assign(max_panels * qt.M[t] * PT, 0.0f);
        std::vector<float> ss(max_panels * PT), resd(max_panels * PT), resq(m), Li(m);
        std::vector<uint16_t> live(max_panels);
        uint64_t l_cells = 0, l_docs = 0, l_tokens = 0;

        #pragma omp for schedule(dynamic)
        for (size_t g = 0; g < n_groups; ++g) {
            size_t d0 = (size_t)group_doc_starts[g], d1 = (size_t)group_doc_starts[g + 1];
            for (size_t d = d0; d < d1; ++d) {
                size_t tok0 = (size_t)doc_offsets[d], tok1 = (size_t)doc_offsets[d + 1];
                size_t n_tok = tok1 - tok0;
                if (n_tok == 0) continue;
                size_t n_panels = n_tok / PT;
                for (size_t p = 0; p < n_panels; ++p) live[p] = 0xFFFFu;
                size_t n_live_panels = n_panels;

                bool doc_pruned = false;
                size_t cur = 0;
                for (size_t c = 0; c < n_cps && !doc_pruned; ++c) {
                    size_t end = cps[c];
                    for (size_t p = 0; p < n_panels; ++p) {
                        if (!live[p]) continue;               // dead panel: skip its bytes
                        const float* panel = panel_data + (tok0 + p * PT) * D;
                        panel_sumsq_seg(panel, order, cur, end, ss.data() + p * PT, cur == 0);
                        for (size_t t = 0; t < qt.n_tiles; ++t)
                            run_panel_seg(panel, qt.qpack.data() + qt.pack_off[t], qt.M[t],
                                          order, cur, end,
                                          Pt[t].data() + p * qt.M[t] * PT, cur == 0);
                    }
                    l_cells += (uint64_t)(end - cur) * (n_live_panels * PT) * m;
                    cur = end;

                    if (cur == D) break;

                    float beta = shrink + (1.0f - shrink) * ((float)(D - cur) / (float)D);
                    for (size_t i = 0; i < m; ++i) {
                        float s = 1.0f - Qcum[i * (D + 1) + cur];
                        resq[i] = beta * std::sqrt(s > 0.0f ? s : 0.0f);
                    }
                    for (size_t p = 0; p < n_panels; ++p)
                        if (live[p]) panel_resd(ss.data() + p * PT, resd.data() + p * PT);

                    // (a) L_i(d) over live lanes.
                    for (size_t t = 0; t < qt.n_tiles; ++t) {
                        size_t row_stride = qt.M[t] * PT;
                        for (size_t i = 0; i < qt.m_real[t]; ++i)
                            Li[t * TILE_MAX + i] = doc_row_lbmax_masked(
                                Pt[t].data() + i * PT, resd.data(), live.data(),
                                n_panels, row_stride, resq[t * TILE_MAX + i]);
                    }

                    // (b) domination test per live panel: lane survives iff
                    //     ∃ i: P + resq_i·resd >= L_i.
                    for (size_t p = 0; p < n_panels; ++p) {
                        if (!live[p]) continue;
#ifdef __AVX512F__
                        __mmask16 surv = 0;
                        __m512 r = _mm512_loadu_ps(resd.data() + p * PT);
                        for (size_t t = 0; t < qt.n_tiles && surv != live[p]; ++t) {
                            size_t row_stride = qt.M[t] * PT;
                            for (size_t i = 0; i < qt.m_real[t]; ++i) {
                                __m512 P = _mm512_loadu_ps(Pt[t].data() + i * PT + p * row_stride);
                                __m512 ub = _mm512_fmadd_ps(
                                    _mm512_set1_ps(resq[t * TILE_MAX + i]), r, P);
                                surv |= _mm512_cmp_ps_mask(
                                    ub, _mm512_set1_ps(Li[t * TILE_MAX + i]), _CMP_GE_OQ);
                                if ((surv & live[p]) == live[p]) break;
                            }
                        }
                        uint16_t new_live = live[p] & (uint16_t)surv;
#else
                        uint16_t new_live = 0;
                        for (size_t j = 0; j < PT; ++j) {
                            if (!(live[p] & (1u << j))) continue;
                            bool s = false;
                            for (size_t t = 0; t < qt.n_tiles && !s; ++t) {
                                size_t row_stride = qt.M[t] * PT;
                                for (size_t i = 0; i < qt.m_real[t]; ++i) {
                                    float ub = Pt[t][i * PT + p * row_stride + j]
                                             + resq[t * TILE_MAX + i] * resd[p * PT + j];
                                    if (ub >= Li[t * TILE_MAX + i]) { s = true; break; }
                                }
                            }
                            if (s) new_live |= (1u << j);
                        }
#endif
                        l_tokens += (uint64_t)__builtin_popcount((unsigned)(live[p] ^ new_live));
                        if (live[p] && !new_live) --n_live_panels;
                        live[p] = new_live;
                    }

                    // (c) document upper bound over surviving lanes.
                    float tau = std::max(tau_seed, shared_tau.load(std::memory_order_relaxed));
                    float UB = 0.0f;
                    for (size_t t = 0; t < qt.n_tiles; ++t) {
                        size_t row_stride = qt.M[t] * PT;
                        for (size_t i = 0; i < qt.m_real[t]; ++i)
                            UB += doc_row_ubmax_masked(Pt[t].data() + i * PT, resd.data(),
                                                       live.data(), n_panels, row_stride,
                                                       resq[t * TILE_MAX + i]);
                    }
                    if (UB + UB_EPSILON < tau) { doc_pruned = true; ++l_docs; }
                    if (n_live_panels == 0) break;   // fully token-pruned (finalize below)
                }

                if (!doc_pruned) {
                    // Exact score over surviving lanes (survival invariant §2.4
                    // guarantees each query token's argmax lane is live).
                    float score = 0.0f;
                    for (size_t t = 0; t < qt.n_tiles; ++t) {
                        size_t row_stride = qt.M[t] * PT;
                        for (size_t i = 0; i < qt.m_real[t]; ++i)
                            score += doc_row_ubmax_masked(Pt[t].data() + i * PT, resd.data(),
                                                          live.data(), n_panels, row_stride,
                                                          0.0f);
                    }
                    local.offer(score, (uint32_t)d);
                    atomic_max_tau(shared_tau, local.threshold());
                }
            }
        }

        cells.fetch_add(l_cells, std::memory_order_relaxed);
        docs_pruned.fetch_add(l_docs, std::memory_order_relaxed);
        tokens_pruned.fetch_add(l_tokens, std::memory_order_relaxed);

        #pragma omp critical
        for (size_t i = 0; i < K; ++i)
            if (local.id[i] != 0xffffffffu) global.offer(local.score[i], local.id[i]);
    }

    emit_topk(global, K, topk_id, topk_score);
    if (stats) { stats[0] = cells.load(); stats[1] = docs_pruned.load(); stats[2] = tokens_pruned.load(); }
    return stats ? stats[0] : 0;
}

}  // extern "C"
