// bond_doc.cpp — fused_panel_maxsim_bond / _bond_cheap: the document-level
// pruning arms — same microkernel as brute.cpp, dimension-incremental with
// per-DOCUMENT Cauchy-Schwarz bound checkpoints (Stage 3b §5.5).
//
// The scan of one document is split into dimension segments ending at the
// caller-supplied checkpoint set C plus D (R3: `checkpoints`/`n_checkpoints`
// ABI parameters; NULL/0 selects the default {32, 64} used by Stage 3b —
// values are clamped to (0, D), sorted, and deduped).  Within a NON-final segment the
// register-tiled microkernel runs unchanged over the document's panels,
// spilling partials to a small L1/L2-resident per-document buffer between
// segments; the per-lane sum of squares (for resd) is accumulated in the
// same pass as query tile 0, so each panel column is traversed once per
// segment.  At each non-final checkpoint the document upper bound
//     UB_d = sum_i max_j (P_ij + resq_i * resd_j)
// is evaluated lane-vectorized over the spilled partials; if UB_d < tau the
// document's remaining dimension segments are skipped entirely (bytes never
// read — this is the mechanism that goes below the dense DRAM floor).
// The FINAL segment needs neither sumsq (the residual is 0 at D — no bound
// is evaluated) nor a spill: it reloads the partials once and folds the
// per-document max in registers, exactly like the brute epilogue, so Pt is
// never stored or rescanned after the last checkpoint.
// Token-level domination pruning is deliberately absent (e03 showed its
// bookkeeping costs more than it saves at wall-clock; it remains measured by
// the wide-block ACCOUNTING kernel and by bond_token.cpp).
//
// Both document-level arms share one templated scanner (CHEAP below):
//
//   fused_panel_maxsim_bond       (CHEAP = false) — the TIGHT bound above:
//       UB_d = sum_i max_j (P_ij + resq_i * resd_j).
//       Doc-side residual machinery: sumsq fused into tile 0, one sqrt per
//       lane per checkpoint (panel_resd), resd loads in the UB reduction.
//   fused_panel_maxsim_bond_cheap (CHEAP = true)  — the QUERY-ONLY bound
//       (BOND SIGMOD-2002 H_q lesson: prefer the cheapest bound; see
//       docs/bond2002_bound_cost_analysis.md).  Unit-norm doc tokens give
//       resd_j <= 1, so
//           UB_d = sum_i max_j P_ij + sum_i resq_i
//       where the second term is query-only and hoisted per checkpoint.
//       Tile 0 runs the plain microkernel (prefetch kept, NO sumsq), no
//       sqrt, no resd — the bound costs one plain max-reduction over the
//       spilled partials plus one add.  Looser: UB_cheap >= UB_tight, so it
//       never prunes a document the tight bound would keep (still exact-safe
//       at shrink=1), but it may prune later or not at all (e09 measures
//       the trade).  shrink applies to resq exactly as in the tight arm.
//
// The query-side residuals resq_i(c) = beta_c * sqrt(1 - Qcum[i, cps[c]])
// depend only on the query and checkpoint, so BOTH arms precompute them once
// per call (hoisted out of the document loop; identical bound math).
//
// Exactness at shrink=1: identical bound math to Stage 1 §2.3 evaluated at
// document granularity (the cheap arm additionally relaxes resd_j to 1,
// which only raises UB); duplicate-token padding adds identical lanes (max-
// invariant); zero-padded query rows are excluded from UB and score.
//
// tau = max(tau_seed, shared_tau) where shared_tau is the atomic max over
// all threads' local top-K thresholds (safe: see atomic_max_tau).
//
// stats[0] = cells actually scanned (dims x PADDED tokens x m real query
//            tokens; wall-clock convention, NOT comparable to the accounting
//            kernel's unpadded live-set counter)
// stats[1] = docs pruned at a checkpoint (never finalized)

#include "common.hpp"
#include "microkernel.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// Per-thread scanner for the document-level arms: owns the spill scratch and
// walks one document at a time through the checkpoint sequence.  CHEAP is a
// compile-time flag, so each arm gets its own fully specialized copy.
template <bool CHEAP>
struct DocScanner {
    // Per-call constants (shared across threads, read-only).
    const float* panel_data;
    size_t D, m;
    const QueryTiles& qt;
    const uint32_t* order;
    const size_t* cps;
    size_t n_cps;
    const float* resq_cp;   // [n_cps x m] query-side residuals per checkpoint
    const float* rq_sum;    // [n_cps] hoisted query-only bound term (cheap arm)
    const std::atomic<float>* shared_tau;
    float tau_seed;

    // Per-thread scratch.  Spilled partials Pt: per tile, per panel, per
    // query row: [i*PT] lanes; row stride across panels within a tile = M*PT.
    std::vector<DocMax> dm;
    std::vector<std::vector<float>> Pt;
    std::vector<float> ss, resd;
    uint64_t cells = 0, docs_pruned = 0;

    DocScanner(const float* panel_data_, size_t D_, size_t m_, const QueryTiles& qt_,
               const uint32_t* order_, const size_t* cps_, size_t n_cps_,
               const float* resq_cp_, const float* rq_sum_,
               const std::atomic<float>* shared_tau_, float tau_seed_,
               size_t max_panels)
        : panel_data(panel_data_), D(D_), m(m_), qt(qt_), order(order_),
          cps(cps_), n_cps(n_cps_), resq_cp(resq_cp_), rq_sum(rq_sum_),
          shared_tau(shared_tau_), tau_seed(tau_seed_),
          dm(qt_.n_tiles), Pt(qt_.n_tiles),
          ss(CHEAP ? 0 : max_panels * PT), resd(CHEAP ? 0 : max_panels * PT) {
        for (size_t t = 0; t < qt.n_tiles; ++t)
            Pt[t].assign(max_panels * qt.M[t] * PT, 0.0f);
    }

    // One NON-final dimension segment [cur, end) over all panels, spilling
    // the partials.  Tile 0 fuses the per-lane sumsq (tight arm) or runs the
    // plain microkernel with only the prefetch (cheap arm — no doc-side
    // residual); remaining tiles reuse the L1-resident panel.
    void scan_segment(size_t tok0, size_t n_panels, size_t cur, size_t end) {
        for (size_t p = 0; p < n_panels; ++p) {
            const float* panel = panel_data + (tok0 + p * PT) * D;
            if constexpr (CHEAP) {
                run_panel_seg_pf(panel, qt.qpack.data() + qt.pack_off[0], qt.M[0],
                                 order, cur, end, Pt[0].data() + p * qt.M[0] * PT,
                                 cur == 0, panel + PT * D);
            } else {
                run_panel_seg_ss(panel, qt.qpack.data() + qt.pack_off[0], qt.M[0],
                                 order, cur, end, Pt[0].data() + p * qt.M[0] * PT,
                                 ss.data() + p * PT, cur == 0, panel + PT * D);
            }
            for (size_t t = 1; t < qt.n_tiles; ++t)
                run_panel_seg(panel, qt.qpack.data() + qt.pack_off[t], qt.M[t],
                              order, cur, end, Pt[t].data() + p * qt.M[t] * PT, cur == 0);
        }
    }

    // FINAL segment [cur, D): no bound follows, so skip sumsq and fold the
    // per-doc max in registers instead of spilling and rescanning Pt.
    void scan_final(size_t tok0, size_t n_panels, size_t cur, size_t end) {
        for (size_t p = 0; p < n_panels; ++p) {
            const float* panel = panel_data + (tok0 + p * PT) * D;
            for (size_t t = 0; t < qt.n_tiles; ++t)
                run_panel_seg_final(panel, qt.qpack.data() + qt.pack_off[t], qt.M[t],
                                    order, cur, end, Pt[t].data() + p * qt.M[t] * PT,
                                    cur == 0, dm[t], t == 0 ? panel + PT * D : nullptr);
        }
    }

    // Document upper bound at checkpoint c, lane-vectorized over the spilled
    // partials (the tight arm first turns the fused sumsq into resd).
    float upper_bound(size_t c, size_t n_panels) {
        const float* resq_c = resq_cp + c * m;
        if constexpr (!CHEAP)
            for (size_t p = 0; p < n_panels; ++p)
                panel_resd(ss.data() + p * PT, resd.data() + p * PT);
        float UB = CHEAP ? rq_sum[c] : 0.0f;
        for (size_t t = 0; t < qt.n_tiles; ++t) {
            size_t row_stride = qt.M[t] * PT;
            for (size_t i = 0; i < qt.m_real[t]; ++i) {
                if constexpr (CHEAP)
                    UB += doc_row_pmax(Pt[t].data() + i * PT, n_panels, row_stride);
                else
                    UB += doc_row_ubmax(Pt[t].data() + i * PT, resd.data(),
                                        n_panels, row_stride, resq_c[t * TILE_MAX + i]);
            }
        }
        return UB;
    }

    // Walk one document through the checkpoint sequence.  Returns true if it
    // was pruned at a checkpoint; otherwise writes the exact score (from the
    // register-folded per-doc max).
    bool scan_doc(size_t tok0, size_t n_tok, float& score) {
        size_t n_panels = n_tok / PT;
        for (size_t t = 0; t < qt.n_tiles; ++t) dm[t].reset(qt.M[t]);

        size_t cur = 0;
        for (size_t c = 0; c < n_cps; ++c) {
            size_t end = cps[c];
            if (end == D) {
                scan_final(tok0, n_panels, cur, end);
                cells += (uint64_t)(end - cur) * n_tok * m;
                break;
            }
            scan_segment(tok0, n_panels, cur, end);
            cells += (uint64_t)(end - cur) * n_tok * m;
            cur = end;

            float tau = std::max(tau_seed, shared_tau->load(std::memory_order_relaxed));
            if (upper_bound(c, n_panels) + UB_EPSILON < tau) { ++docs_pruned; return true; }
        }

        score = 0.0f;
        for (size_t t = 0; t < qt.n_tiles; ++t) score += dm[t].reduce_sum(qt.m_real[t]);
        return false;
    }
};

template <bool CHEAP>
static uint64_t fused_bond_doc_impl(
        const float* panel_data,
        const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets,
        const uint64_t* group_doc_starts,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const float* Qcum,
        const uint32_t* checkpoints, size_t n_checkpoints,
        float shrink, float tau_seed, size_t K, int n_threads,
        uint32_t* topk_id, float* topk_score, uint64_t* stats) {

    (void)group_offsets;
    QueryTiles qt(query, m, D);
    if (!qt.valid || panel_data == nullptr || group_offsets == nullptr ||
        doc_offsets == nullptr || group_doc_starts == nullptr || order == nullptr ||
        Qcum == nullptr || topk_id == nullptr || topk_score == nullptr ||
        n_groups == 0 || K == 0 || K > (size_t)group_doc_starts[n_groups]) {
        return NATIVE_ERROR;
    }
    TopK global(K);
    std::atomic<float> shared_tau{-std::numeric_limits<float>::infinity()};
    std::atomic<uint64_t> cells{0}, docs_pruned{0};

    size_t cps[MAX_CPS];
    size_t n_cps = build_checkpoints(checkpoints, n_checkpoints, D, cps);

    // Query-side residuals per (checkpoint, query row) — query-only, so
    // computed once per call for both arms; the cheap arm additionally sums
    // them into its query-only bound term.
    std::vector<float> resq_cp(n_cps * m, 0.0f);
    std::vector<float> rq_sum(n_cps, 0.0f);
    for (size_t c = 0; c < n_cps; ++c) {
        size_t end = cps[c];
        if (end == D) continue;                       // final segment: no bound
        float beta = shrink + (1.0f - shrink) * ((float)(D - end) / (float)D);
        for (size_t i = 0; i < m; ++i) {
            float s = 1.0f - Qcum[i * (D + 1) + end];
            float r = beta * std::sqrt(s > 0.0f ? s : 0.0f);
            resq_cp[c * m + i] = r;
            rq_sum[c] += r;
        }
    }

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
        DocScanner<CHEAP> scan(panel_data, D, m, qt, order, cps, n_cps,
                               resq_cp.data(), rq_sum.data(),
                               &shared_tau, tau_seed, max_panels);

        #pragma omp for schedule(dynamic)
        for (size_t g = 0; g < n_groups; ++g) {
            size_t d0 = (size_t)group_doc_starts[g], d1 = (size_t)group_doc_starts[g + 1];
            for (size_t d = d0; d < d1; ++d) {
                size_t tok0 = (size_t)doc_offsets[d];
                size_t n_tok = (size_t)(doc_offsets[d + 1] - doc_offsets[d]);
                if (n_tok == 0) continue;
                float score;
                if (scan.scan_doc(tok0, n_tok, score)) continue;   // pruned
                local.offer(score, (uint32_t)d);
                atomic_max_tau(shared_tau, local.threshold());
            }
        }

        cells.fetch_add(scan.cells, std::memory_order_relaxed);
        docs_pruned.fetch_add(scan.docs_pruned, std::memory_order_relaxed);

        #pragma omp critical
        for (size_t i = 0; i < K; ++i)
            if (local.id[i] != 0xffffffffu) global.offer(local.score[i], local.id[i]);
    }

    emit_topk(global, K, topk_id, topk_score);
    if (stats) { stats[0] = cells.load(); stats[1] = docs_pruned.load(); }
    return stats ? stats[0] : 0;
}

extern "C" {

uint64_t fused_panel_maxsim_bond(
        const float* panel_data,
        const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets,
        const uint64_t* group_doc_starts,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const float* Qcum,
        const uint32_t* checkpoints, size_t n_checkpoints,
        float shrink, float tau_seed, size_t K, int n_threads,
        uint32_t* topk_id, float* topk_score, uint64_t* stats) {
    return fused_bond_doc_impl<false>(
        panel_data, group_offsets, n_groups, doc_offsets, group_doc_starts,
        query, m, D, order, Qcum, checkpoints, n_checkpoints,
        shrink, tau_seed, K, n_threads, topk_id, topk_score, stats);
}

uint64_t fused_panel_maxsim_bond_cheap(
        const float* panel_data,
        const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets,
        const uint64_t* group_doc_starts,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const float* Qcum,
        const uint32_t* checkpoints, size_t n_checkpoints,
        float shrink, float tau_seed, size_t K, int n_threads,
        uint32_t* topk_id, float* topk_score, uint64_t* stats) {
    return fused_bond_doc_impl<true>(
        panel_data, group_offsets, n_groups, doc_offsets, group_doc_starts,
        query, m, D, order, Qcum, checkpoints, n_checkpoints,
        shrink, tau_seed, K, n_threads, topk_id, topk_score, stats);
}

}  // extern "C"
