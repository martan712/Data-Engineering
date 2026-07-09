// microkernel_portable.hpp — autovectorized fallback microkernels.
//
// One of the two interchangeable arch backends (the other is
// microkernel_avx512.hpp — see its header comment for the naming scheme).
// Selected by microkernel.hpp when __AVX512F__ is absent; include that
// header, not this one.  Fixed-trip loops over a PT-wide lane array so the
// compiler can autovectorize; semantics are identical to the AVX-512 backend.

#pragma once

#include "common.hpp"

// ---------------------------------------------------------------------------
// Dense microkernel (brute).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Segmented microkernels (BOND kernels).
// ---------------------------------------------------------------------------

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

#define BOND_PF(pf, z) __builtin_prefetch((pf) + (z) * PT)

template <size_t M>
static inline void panel_tile_seg_ss(const float* panel, const float* qpack,
                                     const uint32_t* order, size_t z0, size_t z1,
                                     float* Pt, float* ss, bool first,
                                     const float* pf) {
    float acc[M][PT], s[PT];
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) acc[i][j] = first ? 0.0f : Pt[i * PT + j];
    for (size_t j = 0; j < PT; ++j) s[j] = first ? 0.0f : ss[j];
    for (size_t t = z0; t < z1; ++t) {
        BOND_PF(pf, (size_t)order[t]);
        const float* col = panel + (size_t)order[t] * PT;
        const float* qz = qpack + (size_t)order[t] * M;
        for (size_t j = 0; j < PT; ++j) s[j] += col[j] * col[j];
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < PT; ++j) acc[i][j] += qz[i] * col[j];
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) Pt[i * PT + j] = acc[i][j];
    for (size_t j = 0; j < PT; ++j) ss[j] = s[j];
}

template <size_t M>
static inline void panel_tile_seg_pf(const float* panel, const float* qpack,
                                     const uint32_t* order, size_t z0, size_t z1,
                                     float* Pt, bool first, const float* pf) {
    float acc[M][PT];
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) acc[i][j] = first ? 0.0f : Pt[i * PT + j];
    for (size_t t = z0; t < z1; ++t) {
        BOND_PF(pf, (size_t)order[t]);
        const float* col = panel + (size_t)order[t] * PT;
        const float* qz = qpack + (size_t)order[t] * M;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < PT; ++j) acc[i][j] += qz[i] * col[j];
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) Pt[i * PT + j] = acc[i][j];
}

template <size_t M>
static inline void panel_tile_seg_final(const float* panel, const float* qpack,
                                        const uint32_t* order, size_t z0, size_t z1,
                                        const float* Pt, bool first, float (*dmax)[PT],
                                        const float* pf) {
    float acc[M][PT];
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) acc[i][j] = first ? 0.0f : Pt[i * PT + j];
    for (size_t t = z0; t < z1; ++t) {
        if (pf) BOND_PF(pf, (size_t)order[t]);
        const float* col = panel + (size_t)order[t] * PT;
        const float* qz = qpack + (size_t)order[t] * M;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < PT; ++j) acc[i][j] += qz[i] * col[j];
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) dmax[i][j] = std::max(dmax[i][j], acc[i][j]);
}

template <size_t M>
static inline void panel_tile_seg_final_masked(const float* panel, const float* qpack,
                                               const uint32_t* order, size_t z0, size_t z1,
                                               const float* Pt, bool first,
                                               uint16_t live, float (*dmax)[PT],
                                               const float* pf) {
    float acc[M][PT];
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j) acc[i][j] = first ? 0.0f : Pt[i * PT + j];
    for (size_t t = z0; t < z1; ++t) {
        if (pf) BOND_PF(pf, (size_t)order[t]);
        const float* col = panel + (size_t)order[t] * PT;
        const float* qz = qpack + (size_t)order[t] * M;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < PT; ++j) acc[i][j] += qz[i] * col[j];
    }
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < PT; ++j)
            if (live & (1u << j)) dmax[i][j] = std::max(dmax[i][j], acc[i][j]);
}

// ---------------------------------------------------------------------------
// Bound reductions over the spilled partials.
// ---------------------------------------------------------------------------

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

static inline float doc_row_pmax(const float* Pt_row, size_t n_panels,
                                 size_t row_stride) {
    float best = -std::numeric_limits<float>::infinity();
    for (size_t p = 0; p < n_panels; ++p)
        for (size_t j = 0; j < PT; ++j)
            best = std::max(best, Pt_row[p * row_stride + j]);
    return best;
}

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

// Token domination test, one query row x one panel: mask of lanes j where
// P_ij + resq_i*resd_j >= Li.  Computed over all 16 lanes; the caller ANDs
// with the live mask, so dead-lane bits are harmless.
static inline uint16_t row_survivor_mask(const float* Pt_row, const float* resd_p,
                                         float resq, float Li) {
    uint16_t mask = 0;
    for (size_t j = 0; j < PT; ++j)
        if (Pt_row[j] + resq * resd_p[j] >= Li) mask |= (uint16_t)(1u << j);
    return mask;
}
