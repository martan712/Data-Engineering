// brute.cpp — fused_panel_maxsim_brute: the dense (no-pruning) baseline arm.
//
// Register-tiled dense MaxSim scan over the panel-major PDX layout with the
// per-document max-reduction fused into the accumulator tile.  Layout,
// tiling, and threading contract: common.hpp.
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

#include "common.hpp"
#include "microkernel.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// Dense scan of one document: every panel x every query tile, with the
// per-doc max folded in registers; the 8 KB panel stays L1-resident across
// tiles because tiles iterate inside the panel loop.
static inline float score_doc(const float* panel_data, size_t D, const QueryTiles& qt,
                              DocMax* dm, size_t tok0, size_t tok1) {
    for (size_t t = 0; t < qt.n_tiles; ++t) dm[t].reset(qt.M[t]);
    for (size_t p = tok0; p < tok1; p += PT) {
        const float* panel = panel_data + p * D;
        for (size_t t = 0; t < qt.n_tiles; ++t)
            run_panel(panel, qt.qpack.data() + qt.pack_off[t], qt.M[t], D, dm[t]);
    }
    float score = 0.0f;
    for (size_t t = 0; t < qt.n_tiles; ++t) score += dm[t].reduce_sum(qt.m_real[t]);
    return score;
}

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
                local.offer(score_doc(panel_data, D, qt, dm, tok0, tok1), (uint32_t)d);
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
