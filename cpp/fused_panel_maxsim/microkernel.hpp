// microkernel.hpp — arch backend selection + tile-size dispatch.
//
// Include this from the kernel .cpp files; it pulls in exactly one of the
// two arch backends (identical symbol sets) and provides the run_panel*
// wrappers that dispatch the runtime tile height M in {8, 16, 24} to the
// compile-time template instantiations.

#pragma once

#include "common.hpp"

#ifdef __AVX512F__
#include "microkernel_avx512.hpp"
#else
#include "microkernel_portable.hpp"
#endif

static inline void run_panel(const float* panel, const float* qpack, size_t M,
                             size_t D, DocMax& dm) {
    switch (M) {
        case 8:  panel_tile<8> (panel, qpack, D, dm.v); break;
        case 16: panel_tile<16>(panel, qpack, D, dm.v); break;
        default: panel_tile<24>(panel, qpack, D, dm.v); break;
    }
}

static inline void run_panel_seg(const float* panel, const float* qpack, size_t M,
                                 const uint32_t* order, size_t z0, size_t z1,
                                 float* Pt, bool first) {
    switch (M) {
        case 8:  panel_tile_seg<8> (panel, qpack, order, z0, z1, Pt, first); break;
        case 16: panel_tile_seg<16>(panel, qpack, order, z0, z1, Pt, first); break;
        default: panel_tile_seg<24>(panel, qpack, order, z0, z1, Pt, first); break;
    }
}

static inline void run_panel_seg_ss(const float* panel, const float* qpack, size_t M,
                                    const uint32_t* order, size_t z0, size_t z1,
                                    float* Pt, float* ss, bool first, const float* pf) {
    switch (M) {
        case 8:  panel_tile_seg_ss<8> (panel, qpack, order, z0, z1, Pt, ss, first, pf); break;
        case 16: panel_tile_seg_ss<16>(panel, qpack, order, z0, z1, Pt, ss, first, pf); break;
        default: panel_tile_seg_ss<24>(panel, qpack, order, z0, z1, Pt, ss, first, pf); break;
    }
}

static inline void run_panel_seg_pf(const float* panel, const float* qpack, size_t M,
                                    const uint32_t* order, size_t z0, size_t z1,
                                    float* Pt, bool first, const float* pf) {
    switch (M) {
        case 8:  panel_tile_seg_pf<8> (panel, qpack, order, z0, z1, Pt, first, pf); break;
        case 16: panel_tile_seg_pf<16>(panel, qpack, order, z0, z1, Pt, first, pf); break;
        default: panel_tile_seg_pf<24>(panel, qpack, order, z0, z1, Pt, first, pf); break;
    }
}

static inline void run_panel_seg_final(const float* panel, const float* qpack, size_t M,
                                       const uint32_t* order, size_t z0, size_t z1,
                                       const float* Pt, bool first, DocMax& dm,
                                       const float* pf) {
    switch (M) {
        case 8:  panel_tile_seg_final<8> (panel, qpack, order, z0, z1, Pt, first, dm.v, pf); break;
        case 16: panel_tile_seg_final<16>(panel, qpack, order, z0, z1, Pt, first, dm.v, pf); break;
        default: panel_tile_seg_final<24>(panel, qpack, order, z0, z1, Pt, first, dm.v, pf); break;
    }
}

static inline void run_panel_seg_final_masked(const float* panel, const float* qpack, size_t M,
                                              const uint32_t* order, size_t z0, size_t z1,
                                              const float* Pt, bool first,
                                              uint16_t live, DocMax& dm, const float* pf) {
    switch (M) {
        case 8:  panel_tile_seg_final_masked<8> (panel, qpack, order, z0, z1, Pt, first, live, dm.v, pf); break;
        case 16: panel_tile_seg_final_masked<16>(panel, qpack, order, z0, z1, Pt, first, live, dm.v, pf); break;
        default: panel_tile_seg_final_masked<24>(panel, qpack, order, z0, z1, Pt, first, live, dm.v, pf); break;
    }
}
