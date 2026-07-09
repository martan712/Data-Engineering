// common.hpp — shared infrastructure for the fused panel MaxSim kernels
// (Stage 3b deliverable).
//
// Register-tiled dense MaxSim scan over the panel-major PDX layout, with the
// per-document max-reduction fused into the accumulator tile.  Design,
// theory, and measured motivation: docs/stage3b_fused_panel_maxsim_kernel.md.
// In short: the panel-major layout is the BLAS packed-B micro-panel format
// packed once at index-build time (§3.1), and fusing the segmented max into
// the epilogue means the m x T score matrix is never materialized (§3.2) —
// which is why these kernels beat `Q @ X.T` + reduceat despite BLAS.
//
// Package map (one concern per file, all linked into fused_panel_maxsim.so):
//   common.hpp               — layout contract, constants, TopK, QueryTiles,
//                              checkpoint-set handling (this file)
//   microkernel.hpp          — arch selection + tile-size dispatch wrappers
//   microkernel_avx512.hpp   — AVX-512 panel microkernels + bound reductions
//   microkernel_portable.hpp — autovectorized fallbacks (same symbol set)
//   brute.cpp                — fused_panel_maxsim_brute (dense baseline)
//   bond_doc.cpp             — fused_panel_maxsim_bond / _bond_cheap
//   bond_token.cpp           — fused_panel_maxsim_bond_token
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
// Threading: OpenMP over groups (independent); per-thread TopK merged at the
// end.  n_threads <= 0 means the OpenMP default; the .so also builds and
// runs correctly without OpenMP (pragmas ignored, serial).
//
// Build: make -C cpp/fused_panel_maxsim  (adds -fopenmp; AVX-512 fast path
// under __AVX512F__, portable autovectorized fallback otherwise).

#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

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

static inline void emit_topk(const TopK& topk, size_t K, uint32_t* topk_id, float* topk_score) {
    std::vector<size_t> idx(K);
    for (size_t t = 0; t < K; ++t) idx[t] = t;
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return topk.score[a] > topk.score[b]; });
    for (size_t t = 0; t < K; ++t) { topk_score[t] = topk.score[idx[t]]; topk_id[t] = topk.id[idx[t]]; }
}

// ---------------------------------------------------------------------------
// Query tiling (§5.1, §5.4): m query tokens are split into ceil(m/24) tiles,
// each zero-padded to the smallest M in {8,16,24} >= its size.  Zero-padded
// query rows accumulate exactly 0 and contribute max_j 0 = 0 to the score —
// exact.  Tiles iterate inside the panel loop, so the 8 KB panel stays
// L1-resident across tiles.  M=24 uses 24 of 32 zmm registers for the
// accumulator tile; hence 24, not 32, is the largest tile.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Checkpoint-set handling (BOND kernels only).
// ---------------------------------------------------------------------------

// Build the segment-end sequence from the caller's checkpoint set (R3).
// NULL/0 selects the default {32, 64}; entries are clamped to (0, D), sorted
// ascending, deduped, and D is always appended as the final segment end.
// At most MAX_CPS-1 bound checkpoints are honored (more buys nothing: e02
// shows survival changes on a coarser grid than 8 boundaries).
static constexpr size_t MAX_CPS = 9;
static inline size_t build_checkpoints(const uint32_t* checkpoints, size_t n_checkpoints,
                                       size_t D, size_t* cps) {
    static const uint32_t kDefault[2] = {32, 64};
    if (checkpoints == nullptr || n_checkpoints == 0) {
        checkpoints = kDefault;
        n_checkpoints = 2;
    }
    size_t tmp[MAX_CPS - 1];
    size_t n = 0;
    for (size_t i = 0; i < n_checkpoints && n < MAX_CPS - 1; ++i) {
        size_t c = checkpoints[i];
        if (c > 0 && c < D) tmp[n++] = c;
    }
    std::sort(tmp, tmp + n);
    size_t n_cps = 0;
    for (size_t i = 0; i < n; ++i)
        if (n_cps == 0 || tmp[i] > cps[n_cps - 1]) cps[n_cps++] = tmp[i];
    cps[n_cps++] = D;
    return n_cps;
}
