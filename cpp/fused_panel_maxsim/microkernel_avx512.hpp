// microkernel_avx512.hpp — AVX-512 panel microkernels and bound reductions.
//
// One of the two interchangeable arch backends (the other is
// microkernel_portable.hpp — same symbol set, autovectorized).  Selected by
// microkernel.hpp under __AVX512F__; include that header, not this one.
//
// Naming: panel_tile* computes one 16-lane panel x one M-row query tile;
// doc_row_* reduces spilled partials into per-query-row bound terms.
// The _seg variants accumulate a dimension segment [z0, z1) of the permuted
// order and spill/reload partials in Pt between segments (BOND kernels);
// suffixes: _ss fuses the per-lane sumsq, _pf keeps only the prefetch,
// _final folds into the per-document max without spilling, _masked applies a
// live-lane mask (token-level kernel).

#pragma once

#include <immintrin.h>

#include "common.hpp"

// ---------------------------------------------------------------------------
// Dense microkernel (brute): one panel x one query tile, accumulated over all
// D dims, then folded into the per-document running max (fused epilogue, §5.4).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Segmented microkernels (BOND kernels): one panel x one tile over dims
// order[z0:z1); partials spilled to Pt[i*PT].
// ---------------------------------------------------------------------------

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

// A permuted dimension order defeats the hardware prefetcher (each column is
// one scattered 64 B line), so the first tile to touch a panel prefetches the
// SAME columns of the NEXT panel (pf = panel + PT*D): one panel of compute of
// lead time hides the miss.  Prefetch is a hint — a pf past the end of
// panel_data (last panel) is architecturally harmless.
#define BOND_PF(pf, z) _mm_prefetch((const char*)((pf) + (z) * PT), _MM_HINT_T0)

// panel_tile_seg fused with the per-lane sumsq accumulation (query tile 0
// only): each panel column is loaded once and feeds both the M dot-product
// rows and the col*col sum needed for resd at the checkpoint.
template <size_t M>
static inline void panel_tile_seg_ss(const float* panel, const float* qpack,
                                     const uint32_t* order, size_t z0, size_t z1,
                                     float* Pt, float* ss, bool first,
                                     const float* pf) {
    __m512 acc[M], s;
    if (first) {
        for (size_t i = 0; i < M; ++i) acc[i] = _mm512_setzero_ps();
        s = _mm512_setzero_ps();
    } else {
        for (size_t i = 0; i < M; ++i) acc[i] = _mm512_loadu_ps(Pt + i * PT);
        s = _mm512_loadu_ps(ss);
    }
    for (size_t t = z0; t < z1; ++t) {
        size_t z = order[t];
        BOND_PF(pf, z);
        __m512 col = _mm512_loadu_ps(panel + z * PT);
        s = _mm512_fmadd_ps(col, col, s);
        const float* qz = qpack + z * M;
        for (size_t i = 0; i < M; ++i)
            acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(qz[i]), col, acc[i]);
    }
    for (size_t i = 0; i < M; ++i) _mm512_storeu_ps(Pt + i * PT, acc[i]);
    _mm512_storeu_ps(ss, s);
}

// panel_tile_seg with the next-panel prefetch but WITHOUT the sumsq
// accumulation — query tile 0 of the cheap-bound arm (no doc-side residual).
template <size_t M>
static inline void panel_tile_seg_pf(const float* panel, const float* qpack,
                                     const uint32_t* order, size_t z0, size_t z1,
                                     float* Pt, bool first, const float* pf) {
    __m512 acc[M];
    if (first) for (size_t i = 0; i < M; ++i) acc[i] = _mm512_setzero_ps();
    else       for (size_t i = 0; i < M; ++i) acc[i] = _mm512_loadu_ps(Pt + i * PT);
    for (size_t t = z0; t < z1; ++t) {
        size_t z = order[t];
        BOND_PF(pf, z);
        __m512 col = _mm512_loadu_ps(panel + z * PT);
        const float* qz = qpack + z * M;
        for (size_t i = 0; i < M; ++i)
            acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(qz[i]), col, acc[i]);
    }
    for (size_t i = 0; i < M; ++i) _mm512_storeu_ps(Pt + i * PT, acc[i]);
}

// Final-segment variant: reloads the spilled partials, accumulates the
// remaining dims, and folds straight into the per-document running max —
// no store back to Pt, no sumsq (the residual is 0 at D).
template <size_t M>
static inline void panel_tile_seg_final(const float* panel, const float* qpack,
                                        const uint32_t* order, size_t z0, size_t z1,
                                        const float* Pt, bool first, __m512* dmax,
                                        const float* pf) {
    __m512 acc[M];
    if (first) for (size_t i = 0; i < M; ++i) acc[i] = _mm512_setzero_ps();
    else       for (size_t i = 0; i < M; ++i) acc[i] = _mm512_loadu_ps(Pt + i * PT);
    for (size_t t = z0; t < z1; ++t) {
        size_t z = order[t];
        if (pf) BOND_PF(pf, z);
        __m512 col = _mm512_loadu_ps(panel + z * PT);
        const float* qz = qpack + z * M;
        for (size_t i = 0; i < M; ++i)
            acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(qz[i]), col, acc[i]);
    }
    for (size_t i = 0; i < M; ++i) dmax[i] = _mm512_max_ps(dmax[i], acc[i]);
}

// Masked final-segment fold (token-level kernel): only live lanes enter the
// per-document max; dead lanes stay at -inf from DocMax::reset.
template <size_t M>
static inline void panel_tile_seg_final_masked(const float* panel, const float* qpack,
                                               const uint32_t* order, size_t z0, size_t z1,
                                               const float* Pt, bool first,
                                               uint16_t live, __m512* dmax,
                                               const float* pf) {
    __m512 acc[M];
    if (first) for (size_t i = 0; i < M; ++i) acc[i] = _mm512_setzero_ps();
    else       for (size_t i = 0; i < M; ++i) acc[i] = _mm512_loadu_ps(Pt + i * PT);
    for (size_t t = z0; t < z1; ++t) {
        size_t z = order[t];
        if (pf) BOND_PF(pf, z);
        __m512 col = _mm512_loadu_ps(panel + z * PT);
        const float* qz = qpack + z * M;
        for (size_t i = 0; i < M; ++i)
            acc[i] = _mm512_fmadd_ps(_mm512_set1_ps(qz[i]), col, acc[i]);
    }
    __mmask16 k = (__mmask16)live;
    for (size_t i = 0; i < M; ++i)
        dmax[i] = _mm512_mask_max_ps(dmax[i], k, dmax[i], acc[i]);
}

// ---------------------------------------------------------------------------
// Bound reductions over the spilled partials.
// ---------------------------------------------------------------------------

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

// Cheap-bound variant: max over the document's lanes of P only (the doc-side
// residual is bounded by 1, so resq moves outside the max as a query-only term).
static inline float doc_row_pmax(const float* Pt_row, size_t n_panels,
                                 size_t row_stride) {
    __m512 best = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t p = 0; p < n_panels; ++p)
        best = _mm512_max_ps(best, _mm512_loadu_ps(Pt_row + p * row_stride));
    return _mm512_reduce_max_ps(best);
}

// Masked variants of the per-doc reductions (token-level kernel, live lanes only).
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

// Token domination test, one query row x one panel: mask of lanes j where
// P_ij + resq_i*resd_j >= Li (the lane survives row i's threshold).  The
// caller ORs these over query rows and ANDs with the live mask.
static inline uint16_t row_survivor_mask(const float* Pt_row, const float* resd_p,
                                         float resq, float Li) {
    __m512 P = _mm512_loadu_ps(Pt_row);
    __m512 r = _mm512_loadu_ps(resd_p);
    __m512 ub = _mm512_fmadd_ps(_mm512_set1_ps(resq), r, P);
    return (uint16_t)_mm512_cmp_ps_mask(ub, _mm512_set1_ps(Li), _CMP_GE_OQ);
}
