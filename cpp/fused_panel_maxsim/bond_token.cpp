// bond_token.cpp — fused_panel_maxsim_bond_token: the TOKEN-level pruning arm.
//
// Identical to fused_panel_maxsim_bond (bond_doc.cpp) in every scheduling
// respect (same microkernel, same panel layout, same checkpoint-set parameter,
// same shared τ, same doc-at-a-time order) with EXACTLY ONE mechanism added:
// the Stage 1 §2.4 token-level domination test at each checkpoint.  A token j
// of doc d is dropped iff  P_ij + resq_i·resd_j < L_i(d)  for EVERY query
// token i, where  L_i(d) = max over d's live tokens of (P_ij - resq_i·resd_j).
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

#include "common.hpp"
#include "microkernel.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// Per-thread scanner: owns the spill scratch plus the per-panel live-lane
// masks and walks one document at a time through the checkpoint sequence.
struct TokenScanner {
    // Per-call constants (shared across threads, read-only).
    const float* panel_data;
    size_t D, m;
    const QueryTiles& qt;
    const uint32_t* order;
    const size_t* cps;
    size_t n_cps;
    const float* Qcum;
    float shrink;
    const std::atomic<float>* shared_tau;
    float tau_seed;

    // Per-thread scratch.  Pt layout as in bond_doc.cpp; live[p] is the
    // 16-lane survival mask of panel p for the current document.
    DocMax dm[8];
    std::vector<std::vector<float>> Pt;
    std::vector<float> ss, resd, resq, Li;
    std::vector<uint16_t> live;
    size_t n_live_panels = 0;
    uint64_t cells = 0, docs_pruned = 0, tokens_pruned = 0;

    TokenScanner(const float* panel_data_, size_t D_, size_t m_, const QueryTiles& qt_,
                 const uint32_t* order_, const size_t* cps_, size_t n_cps_,
                 const float* Qcum_, float shrink_,
                 const std::atomic<float>* shared_tau_, float tau_seed_,
                 size_t max_panels)
        : panel_data(panel_data_), D(D_), m(m_), qt(qt_), order(order_),
          cps(cps_), n_cps(n_cps_), Qcum(Qcum_), shrink(shrink_),
          shared_tau(shared_tau_), tau_seed(tau_seed_),
          Pt(qt_.n_tiles), ss(max_panels * PT), resd(max_panels * PT),
          resq(m_), Li(m_), live(max_panels) {
        for (size_t t = 0; t < qt.n_tiles; ++t)
            Pt[t].assign(max_panels * qt.M[t] * PT, 0.0f);
    }

    // One NON-final dimension segment [cur, end) over the LIVE panels,
    // spilling the partials; tile 0 fuses the per-lane sumsq.
    void scan_segment(size_t tok0, size_t n_panels, size_t cur, size_t end) {
        for (size_t p = 0; p < n_panels; ++p) {
            if (!live[p]) continue;               // dead panel: skip its bytes
            const float* panel = panel_data + (tok0 + p * PT) * D;
            run_panel_seg_ss(panel, qt.qpack.data() + qt.pack_off[0], qt.M[0],
                             order, cur, end, Pt[0].data() + p * qt.M[0] * PT,
                             ss.data() + p * PT, cur == 0, panel + PT * D);
            for (size_t t = 1; t < qt.n_tiles; ++t)
                run_panel_seg(panel, qt.qpack.data() + qt.pack_off[t], qt.M[t],
                              order, cur, end, Pt[t].data() + p * qt.M[t] * PT, cur == 0);
        }
    }

    // FINAL segment [cur, D): register-folded masked max, no sumsq, no spill
    // (mirrors the doc-level kernel's final segment).
    void scan_final(size_t tok0, size_t n_panels, size_t cur, size_t end) {
        for (size_t p = 0; p < n_panels; ++p) {
            if (!live[p]) continue;               // dead panel: skip its bytes
            const float* panel = panel_data + (tok0 + p * PT) * D;
            for (size_t t = 0; t < qt.n_tiles; ++t)
                run_panel_seg_final_masked(panel, qt.qpack.data() + qt.pack_off[t], qt.M[t],
                                           order, cur, end, Pt[t].data() + p * qt.M[t] * PT,
                                           cur == 0, live[p], dm[t],
                                           t == 0 ? panel + PT * D : nullptr);
        }
    }

    // Refresh both residual sides at dimension cut `cur`: query-side resq
    // (with the shrink schedule) and doc-side resd from the fused sumsq.
    void update_residuals(size_t cur, size_t n_panels) {
        float beta = shrink + (1.0f - shrink) * ((float)(D - cur) / (float)D);
        for (size_t i = 0; i < m; ++i) {
            float s = 1.0f - Qcum[i * (D + 1) + cur];
            resq[i] = beta * std::sqrt(s > 0.0f ? s : 0.0f);
        }
        for (size_t p = 0; p < n_panels; ++p)
            if (live[p]) panel_resd(ss.data() + p * PT, resd.data() + p * PT);
    }

    // (a) L_i(d): per query row, the max lower bound over live lanes.
    void compute_Li(size_t n_panels) {
        for (size_t t = 0; t < qt.n_tiles; ++t) {
            size_t row_stride = qt.M[t] * PT;
            for (size_t i = 0; i < qt.m_real[t]; ++i)
                Li[t * TILE_MAX + i] = doc_row_lbmax_masked(
                    Pt[t].data() + i * PT, resd.data(), live.data(),
                    n_panels, row_stride, resq[t * TILE_MAX + i]);
        }
    }

    // (b) domination test per live panel: lane survives iff ∃ i:
    //     P + resq_i·resd >= L_i.  Early-exits once every live lane has
    //     found a witnessing query row.
    void prune_lanes(size_t n_panels) {
        for (size_t p = 0; p < n_panels; ++p) {
            if (!live[p]) continue;
            uint16_t surv = 0;
            for (size_t t = 0; t < qt.n_tiles && (surv & live[p]) != live[p]; ++t) {
                size_t row_stride = qt.M[t] * PT;
                for (size_t i = 0; i < qt.m_real[t]; ++i) {
                    surv |= row_survivor_mask(Pt[t].data() + i * PT + p * row_stride,
                                              resd.data() + p * PT,
                                              resq[t * TILE_MAX + i], Li[t * TILE_MAX + i]);
                    if ((surv & live[p]) == live[p]) break;
                }
            }
            uint16_t new_live = live[p] & surv;
            tokens_pruned += (uint64_t)__builtin_popcount((unsigned)(live[p] ^ new_live));
            if (live[p] && !new_live) --n_live_panels;
            live[p] = new_live;
        }
    }

    // (c) document upper bound over surviving lanes.
    float upper_bound(size_t n_panels) {
        float UB = 0.0f;
        for (size_t t = 0; t < qt.n_tiles; ++t) {
            size_t row_stride = qt.M[t] * PT;
            for (size_t i = 0; i < qt.m_real[t]; ++i)
                UB += doc_row_ubmax_masked(Pt[t].data() + i * PT, resd.data(),
                                           live.data(), n_panels, row_stride,
                                           resq[t * TILE_MAX + i]);
        }
        return UB;
    }

    // Walk one document through the checkpoint sequence.  Returns true if the
    // whole DOCUMENT was pruned at a checkpoint; otherwise writes the exact
    // score over surviving lanes (survival invariant §2.4 guarantees each
    // query token's argmax lane is live).
    bool scan_doc(size_t tok0, size_t n_tok, float& score) {
        size_t n_panels = n_tok / PT;
        for (size_t p = 0; p < n_panels; ++p) live[p] = 0xFFFFu;
        n_live_panels = n_panels;
        for (size_t t = 0; t < qt.n_tiles; ++t) dm[t].reset(qt.M[t]);

        size_t cur = 0;
        for (size_t c = 0; c < n_cps; ++c) {
            size_t end = cps[c];
            if (end == D) {
                scan_final(tok0, n_panels, cur, end);
                cells += (uint64_t)(end - cur) * (n_live_panels * PT) * m;
                break;
            }
            scan_segment(tok0, n_panels, cur, end);
            cells += (uint64_t)(end - cur) * (n_live_panels * PT) * m;
            cur = end;

            update_residuals(cur, n_panels);
            compute_Li(n_panels);
            prune_lanes(n_panels);

            float tau = std::max(tau_seed, shared_tau->load(std::memory_order_relaxed));
            if (upper_bound(n_panels) + UB_EPSILON < tau) { ++docs_pruned; return true; }
            if (n_live_panels == 0) break;   // fully token-pruned (finalize below)
        }

        score = 0.0f;
        for (size_t t = 0; t < qt.n_tiles; ++t) score += dm[t].reduce_sum(qt.m_real[t]);
        return false;
    }
};

extern "C" {

uint64_t fused_panel_maxsim_bond_token(
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
    TopK global(K);
    std::atomic<float> shared_tau{-std::numeric_limits<float>::infinity()};
    std::atomic<uint64_t> cells{0}, docs_pruned{0}, tokens_pruned{0};

    size_t cps[MAX_CPS];
    size_t n_cps = build_checkpoints(checkpoints, n_checkpoints, D, cps);

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
        TokenScanner scan(panel_data, D, m, qt, order, cps, n_cps,
                          Qcum, shrink, &shared_tau, tau_seed, max_panels);

        #pragma omp for schedule(dynamic)
        for (size_t g = 0; g < n_groups; ++g) {
            size_t d0 = (size_t)group_doc_starts[g], d1 = (size_t)group_doc_starts[g + 1];
            for (size_t d = d0; d < d1; ++d) {
                size_t tok0 = (size_t)doc_offsets[d];
                size_t n_tok = (size_t)(doc_offsets[d + 1] - doc_offsets[d]);
                if (n_tok == 0) continue;
                float score;
                if (scan.scan_doc(tok0, n_tok, score)) continue;   // doc pruned
                local.offer(score, (uint32_t)d);
                atomic_max_tau(shared_tau, local.threshold());
            }
        }

        cells.fetch_add(scan.cells, std::memory_order_relaxed);
        docs_pruned.fetch_add(scan.docs_pruned, std::memory_order_relaxed);
        tokens_pruned.fetch_add(scan.tokens_pruned, std::memory_order_relaxed);

        #pragma omp critical
        for (size_t i = 0; i < K; ++i)
            if (local.id[i] != 0xffffffffu) global.offer(local.score[i], local.id[i]);
    }

    emit_topk(global, K, topk_id, topk_score);
    if (stats) { stats[0] = cells.load(); stats[1] = docs_pruned.load(); stats[2] = tokens_pruned.load(); }
    return stats ? stats[0] : 0;
}

}  // extern "C"
