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

#include <cstdint>
#include <cstddef>
#include <cstring>
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

// ---------------------------------------------------------------------------
// TopK — identical pattern to cpp/wide_block_maxsim_bond (kept self-contained).
// ---------------------------------------------------------------------------
struct TopK {
    std::vector<float>    score;
    std::vector<uint32_t> id;
    size_t K;
    explicit TopK(size_t k) : score(k, -std::numeric_limits<float>::infinity()),
                              id(k, 0xffffffffu), K(k) {}
    void offer(float s, uint32_t v) {
        size_t mn = 0;
        for (size_t i = 1; i < K; ++i) if (score[i] < score[mn]) mn = i;
        if (s > score[mn]) { score[mn] = s; id[mn] = v; }
    }
};

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

}  // extern "C"
