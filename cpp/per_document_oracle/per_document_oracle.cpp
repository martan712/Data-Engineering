// per_document_oracle.cpp — unified per-document MaxSim pruning kernels.
//
// Exports three entry points:
//   maxsim_knn_accounting  — exp-09 algorithm (live-set-only cells counter).
//                            Stage 1 §6 accounting mode.
//   maxsim_knn_throughput  — exp-10 algorithm (dense warmup + survivor scan).
//                            Stage 1 §6 throughput mode.
//   maxsim_full            — shared brute-force baseline (token-major, vectorizable).
//
// Source mapping:
//   maxsim_knn_accounting  ← archive/preliminaries/09_maxsim_pruning/maxsim_kernels.cpp
//   maxsim_knn_throughput  ← archive/preliminaries/10_maxsim_pruning_opt/maxsim_kernels.cpp
//   maxsim_full            ← archive/preliminaries/10_maxsim_pruning_opt/maxsim_kernels.cpp
//                            (token-major; identical result to exp-09 brute-force)
//
// Layout: each doc is stored dim-major WITHIN the doc:
//   docs[doc_offsets[d]*D + z*n_d + j]  = dim z, token j of doc d.
//
// Build: see Makefile in this directory.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#include "../native_validation.hpp"

extern "C" {

// ---------------------------------------------------------------------------
// Shared helper: min-tracked top-K of the LARGEST doc scores.
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

// ---------------------------------------------------------------------------
// maxsim_knn_accounting
// Source: archive/preliminaries/09_maxsim_pruning/maxsim_kernels.cpp (maxsim_knn).
// Stage 1 §6 accounting mode: counts only live-set FMAs from dimension 0.
//
// docs          : flat per-doc dim-major store
// doc_offsets   : length n_docs+1, cumulative token counts
// n_docs        : number of documents
// query         : m x D row-major
// m             : number of query tokens
// D             : embedding dimension
// order         : D dim indices — per-query scan order
// fetch_schedule: PDX adaptive block cadence
// n_fetch       : length of fetch_schedule
// Qcum          : m x (D+1) prefix of q_i^2 along order
// shrink        : Cauchy-Schwarz residual scale (1.0 = exact-safe)
// K             : top-K
// topk_id/score : outputs [K]
// stats         : [cells_scanned, docs_pruned, tokens_pruned]
// Returns: cells_scanned.
// ---------------------------------------------------------------------------
uint64_t maxsim_knn_accounting(
        const float* docs, const uint64_t* doc_offsets, size_t n_docs,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
        const float* Qcum, float shrink, size_t K,
        uint32_t* topk_id, float* topk_score, uint64_t* stats) {

    if (!native_validation::document_corpus(docs, doc_offsets, n_docs, D) ||
        !native_validation::query(query, m, D) ||
        !native_validation::scratch_extents(doc_offsets, n_docs + 1, m) ||
        !native_validation::order(order, D) ||
        !native_validation::qcum_matches(Qcum, query, order, m, D) ||
        !native_validation::scalar_parameters(
            shrink, -std::numeric_limits<float>::infinity()) ||
        fetch_schedule == nullptr || n_fetch == 0 || K == 0 || K > n_docs ||
        topk_id == nullptr || topk_score == nullptr || stats == nullptr) {
        return native_validation::ERROR;
    }
    for (size_t i = 0; i < n_fetch; ++i)
        if (fetch_schedule[i] == 0) return native_validation::ERROR;

    // scratch sized to the largest doc
    size_t max_nd = 0;
    for (size_t d = 0; d < n_docs; ++d)
        max_nd = std::max(max_nd, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));

    std::vector<float>    P(m * max_nd);      // partial inner products P[i*nd + j]
    std::vector<float>    sumsq_d(max_nd);    // Σ_scanned d_j^2  → residual ||d_j[k:]||
    std::vector<uint32_t> live(max_nd);
    std::vector<float>    Li(m), resq(m);

    TopK topk(K);
    uint64_t cells = 0, docs_pruned = 0, tokens_pruned = 0;
    const float NEG = -std::numeric_limits<float>::infinity();

    for (size_t d = 0; d < n_docs; ++d) {
        size_t nd = (size_t)(doc_offsets[d + 1] - doc_offsets[d]);
        const float* base = docs + (size_t)doc_offsets[d] * D;
        if (nd == 0) continue;

        for (size_t x = 0; x < m * nd; ++x) P[x] = 0.0f;
        for (size_t j = 0; j < nd; ++j) { sumsq_d[j] = 0.0f; live[j] = (uint32_t)j; }
        size_t n_live = nd;

        float T = topk.threshold();
        size_t cur = 0, fidx = 0;
        bool pruned = false;

        while (cur < D && n_live > 0) {
            size_t blk = fetch_schedule[fidx < n_fetch ? fidx : n_fetch - 1];
            size_t end = cur + blk < D ? cur + blk : D;
            ++fidx;

            // (1) synchronized scan of this dim-block over the live token set
            for (size_t t = cur; t < end; ++t) {
                uint32_t z = order[t];
                const float* col = base + (size_t)z * nd;
                for (size_t a = 0; a < n_live; ++a) {
                    uint32_t j = live[a];
                    float dv = col[j];
                    sumsq_d[j] += dv * dv;
                    float* Pj = P.data() + j;             // stride nd over i
                    for (size_t i = 0; i < m; ++i) Pj[i * nd] += query[i * D + z] * dv;
                }
            }
            cells += (uint64_t)(end - cur) * n_live * m;
            cur = end;

            // residual norms ||q_i[cur:]|| with k-dependent confidence ramp beta(k)
            float beta = shrink + (1.0f - shrink) * ((float)(D - cur) / (float)D);
            for (size_t i = 0; i < m; ++i) {
                float s = 1.0f - Qcum[i * (D + 1) + cur];
                resq[i] = beta * std::sqrt(s > 0.0f ? s : 0.0f);
            }

            // (2a) L_i = max over live j of ( P_ij - resq_i*resd_j )
            for (size_t i = 0; i < m; ++i) Li[i] = NEG;
            for (size_t a = 0; a < n_live; ++a) {
                uint32_t j = live[a];
                float s = 1.0f - sumsq_d[j];
                float resd = std::sqrt(s > 0.0f ? s : 0.0f);
                const float* Pj = P.data() + j;
                for (size_t i = 0; i < m; ++i) {
                    float lb = Pj[i * nd] - resq[i] * resd;
                    if (lb > Li[i]) Li[i] = lb;
                }
            }

            // (2b) token pruning: drop j if U_ij < L_i for EVERY i; compact live set
            size_t w = 0;
            for (size_t a = 0; a < n_live; ++a) {
                uint32_t j = live[a];
                float s = 1.0f - sumsq_d[j];
                float resd = std::sqrt(s > 0.0f ? s : 0.0f);
                const float* Pj = P.data() + j;
                bool dominated = true;
                for (size_t i = 0; i < m; ++i) {
                    if (Pj[i * nd] + resq[i] * resd >= Li[i]) { dominated = false; break; }
                }
                if (!dominated) live[w++] = j;
            }
            tokens_pruned += n_live - w;
            n_live = w;

            // (3) doc upper bound UB = Σ_i max_j U_ij over the surviving live set
            float UB = 0.0f;
            for (size_t i = 0; i < m; ++i) {
                float mx = NEG;
                for (size_t a = 0; a < n_live; ++a) {
                    uint32_t j = live[a];
                    float s = 1.0f - sumsq_d[j];
                    float resd = std::sqrt(s > 0.0f ? s : 0.0f);
                    float ub = P[i * nd + j] + resq[i] * resd;
                    if (ub > mx) mx = ub;
                }
                UB += mx;
            }
            if (UB < T) { pruned = true; docs_pruned++; break; }
        }

        if (pruned || n_live == 0) continue;

        // finalize: score = Σ_i max_j P_ij over the live set (exact when shrink=1)
        float score = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            float mx = NEG;
            const float* Pi = P.data() + i * nd;
            for (size_t a = 0; a < n_live; ++a) { float v = Pi[live[a]]; if (v > mx) mx = v; }
            score += mx;
        }
        topk.offer(score, (uint32_t)d);
    }

    // emit sorted-descending top-K
    std::vector<size_t> idx(K);
    for (size_t t = 0; t < K; ++t) idx[t] = t;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return topk.score[a] > topk.score[b]; });
    for (size_t t = 0; t < K; ++t) { topk_score[t] = topk.score[idx[t]]; topk_id[t] = topk.id[idx[t]]; }

    stats[0] = cells; stats[1] = docs_pruned; stats[2] = tokens_pruned;
    return cells;
}

// ---------------------------------------------------------------------------
// maxsim_knn_throughput
// Source: archive/preliminaries/10_maxsim_pruning_opt/maxsim_kernels.cpp (maxsim_knn).
// Stage 1 §6 throughput mode: token-major P[j*m+i] for SIMD FMA; dense warmup
// over first D/4 dims, then positional (survivor-only) scan once ≥50% prunable.
//
// Identical signature and semantics as maxsim_knn_accounting; only the internal
// memory layout and warmup logic differ.  shrink=1.0 → recall 1.000 (exact-safe).
// ---------------------------------------------------------------------------
uint64_t maxsim_knn_throughput(
        const float* docs, const uint64_t* doc_offsets, size_t n_docs,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
        const float* Qcum, float shrink, size_t K,
        uint32_t* topk_id, float* topk_score, uint64_t* stats) {

    if (!native_validation::document_corpus(docs, doc_offsets, n_docs, D) ||
        !native_validation::query(query, m, D) ||
        !native_validation::scratch_extents(doc_offsets, n_docs + 1, m) ||
        !native_validation::order(order, D) ||
        !native_validation::qcum_matches(Qcum, query, order, m, D) ||
        !native_validation::scalar_parameters(
            shrink, -std::numeric_limits<float>::infinity()) ||
        fetch_schedule == nullptr || n_fetch == 0 || K == 0 || K > n_docs ||
        topk_id == nullptr || topk_score == nullptr || stats == nullptr) {
        return native_validation::ERROR;
    }
    for (size_t i = 0; i < n_fetch; ++i)
        if (fetch_schedule[i] == 0) return native_validation::ERROR;

    size_t max_nd = 0;
    for (size_t d = 0; d < n_docs; ++d)
        max_nd = std::max(max_nd, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));

    std::vector<float>    P(m * max_nd);          // token-major: P[j*m + i]
    std::vector<float>    sumsq_d(max_nd);
    std::vector<float>    resd(max_nd);
    std::vector<uint32_t> pos(max_nd);
    std::vector<float>    Li(m), resq(m), qz(m);

    TopK topk(K);
    uint64_t cells = 0, docs_pruned = 0, tokens_pruned = 0;
    const float NEG = -std::numeric_limits<float>::infinity();
    const float  SELECTIVITY    = 0.5f;
    const size_t WARMUP_MIN_DIM = D / 4;

    for (size_t d = 0; d < n_docs; ++d) {
        size_t nd = (size_t)(doc_offsets[d + 1] - doc_offsets[d]);
        const float* base = docs + (size_t)doc_offsets[d] * D;
        if (nd == 0) continue;
        for (size_t x = 0; x < m * nd; ++x) P[x] = 0.0f;
        for (size_t j = 0; j < nd; ++j) sumsq_d[j] = 0.0f;

        size_t n_live = nd;
        bool   positional = false, pruned = false;
        float  T = topk.threshold();
        size_t cur = 0, fidx = 0;

        while (cur < D) {
            size_t blk = fetch_schedule[fidx < n_fetch ? fidx : n_fetch - 1];
            size_t end = cur + blk < D ? cur + blk : D;
            ++fidx;

            // scan this block (dense during warmup; survivors once positional)
            if (!positional) {
                for (size_t t = cur; t < end; ++t) {
                    uint32_t z = order[t];
                    const float* col = base + (size_t)z * nd;
                    for (size_t i = 0; i < m; ++i) qz[i] = query[i * D + z];
                    for (size_t j = 0; j < nd; ++j) {
                        float dv = col[j]; sumsq_d[j] += dv * dv;
                        float* Pj = P.data() + j * m;            // contiguous m-vector
                        for (size_t i = 0; i < m; ++i) Pj[i] += qz[i] * dv;
                    }
                }
                cells += (uint64_t)(end - cur) * nd * m;
            } else {
                for (size_t t = cur; t < end; ++t) {
                    uint32_t z = order[t];
                    const float* col = base + (size_t)z * nd;
                    for (size_t i = 0; i < m; ++i) qz[i] = query[i * D + z];
                    for (size_t a = 0; a < n_live; ++a) {
                        uint32_t j = pos[a];
                        float dv = col[j]; sumsq_d[j] += dv * dv;
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

            // L_i = max over the active token set of (P_ij − resq_i·resd_j)
            for (size_t i = 0; i < m; ++i) Li[i] = NEG;
            auto eval_token = [&](uint32_t j) {
                float s = 1.0f - sumsq_d[j]; resd[j] = std::sqrt(s > 0.0f ? s : 0.0f);
                const float* Pj = P.data() + (size_t)j * m; float rd = resd[j];
                for (size_t i = 0; i < m; ++i) { float lb = Pj[i] - resq[i] * rd; if (lb > Li[i]) Li[i] = lb; }
            };
            if (!positional) { for (size_t j = 0; j < nd; ++j) eval_token((uint32_t)j); }
            else             { for (size_t a = 0; a < n_live; ++a) eval_token(pos[a]); }

            // token pruning → (re)build the survivor positions array
            auto survives = [&](uint32_t j) {
                const float* Pj = P.data() + (size_t)j * m; float rd = resd[j];
                for (size_t i = 0; i < m; ++i) if (Pj[i] + resq[i] * rd >= Li[i]) return true;
                return false;
            };
            size_t nl = 0;
            if (!positional) {
                for (size_t j = 0; j < nd; ++j) if (survives((uint32_t)j)) pos[nl++] = (uint32_t)j;
                n_live = nl;
                if ((nd - n_live) >= (size_t)(SELECTIVITY * nd)) positional = true;
            } else {
                for (size_t a = 0; a < n_live; ++a) if (survives(pos[a])) pos[nl++] = pos[a];
                n_live = nl;
            }

            // doc upper bound over survivors
            float UB = 0.0f;
            for (size_t i = 0; i < m; ++i) {
                float mx = NEG;
                for (size_t a = 0; a < n_live; ++a) {
                    uint32_t j = pos[a];
                    float ub = P[(size_t)j * m + i] + resq[i] * resd[j];
                    if (ub > mx) mx = ub;
                }
                UB += mx;
            }
            if (UB < T) { pruned = true; docs_pruned++; break; }
            if (n_live == 0) break;
        }

        if (pruned || n_live == 0) continue;

        // finalize: score = Σ_i max_j P_ij over survivors (exact when shrink=1)
        const uint32_t* set = pos.data(); size_t ns = n_live;
        std::vector<uint32_t> all;
        if (ns == 0) {   // never built positions (e.g. D <= WARMUP_MIN): use all
            all.resize(nd); for (size_t j = 0; j < nd; ++j) all[j] = (uint32_t)j;
            set = all.data(); ns = nd;
        }
        float score = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            float mx = NEG;
            for (size_t a = 0; a < ns; ++a) { float v = P[(size_t)set[a] * m + i]; if (v > mx) mx = v; }
            score += mx;
        }
        tokens_pruned += nd - ns;
        topk.offer(score, (uint32_t)d);
    }

    std::vector<size_t> idx(K);
    for (size_t t = 0; t < K; ++t) idx[t] = t;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return topk.score[a] > topk.score[b]; });
    for (size_t t = 0; t < K; ++t) { topk_score[t] = topk.score[idx[t]]; topk_id[t] = topk.id[idx[t]]; }

    stats[0] = cells; stats[1] = docs_pruned; stats[2] = tokens_pruned;
    return cells;
}

// ---------------------------------------------------------------------------
// maxsim_full — brute-force baseline (token-major, vectorizable).
// Source: archive/preliminaries/10_maxsim_pruning_opt/maxsim_kernels.cpp.
// No pruning, no bounds.  Full scan over all dims and all doc tokens.
// Same per-doc dim-major layout as the pruning kernels (honest "pruning off"
// baseline).  Returns total cells scanned (= n_d·D·m summed over docs).
// ---------------------------------------------------------------------------
uint64_t maxsim_full(
        const float* docs, const uint64_t* doc_offsets, size_t n_docs,
        const float* query, size_t m, size_t D, size_t K,
        uint32_t* topk_id, float* topk_score) {

    if (!native_validation::document_corpus(docs, doc_offsets, n_docs, D) ||
        !native_validation::query(query, m, D) ||
        !native_validation::scratch_extents(doc_offsets, n_docs + 1, m) ||
        K == 0 || K > n_docs ||
        topk_id == nullptr || topk_score == nullptr) {
        return native_validation::ERROR;
    }

    size_t max_nd = 0;
    for (size_t d = 0; d < n_docs; ++d)
        max_nd = std::max(max_nd, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));

    std::vector<float> P(m * max_nd);   // token-major P[j*m + i]
    std::vector<float> qz(m);
    TopK topk(K);
    uint64_t cells = 0;
    const float NEG = -std::numeric_limits<float>::infinity();

    for (size_t d = 0; d < n_docs; ++d) {
        size_t nd = (size_t)(doc_offsets[d + 1] - doc_offsets[d]);
        const float* base = docs + (size_t)doc_offsets[d] * D;
        if (nd == 0) continue;
        for (size_t x = 0; x < m * nd; ++x) P[x] = 0.0f;

        for (size_t z = 0; z < D; ++z) {
            const float* col = base + (size_t)z * nd;
            for (size_t i = 0; i < m; ++i) qz[i] = query[i * D + z];
            for (size_t j = 0; j < nd; ++j) {
                float dv = col[j];
                float* Pj = P.data() + j * m;
                for (size_t i = 0; i < m; ++i) Pj[i] += qz[i] * dv;
            }
        }
        cells += (uint64_t)D * nd * m;

        float score = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            float mx = NEG;
            for (size_t j = 0; j < nd; ++j) { float v = P[j * m + i]; if (v > mx) mx = v; }
            score += mx;
        }
        topk.offer(score, (uint32_t)d);
    }

    std::vector<size_t> idx(K);
    for (size_t t = 0; t < K; ++t) idx[t] = t;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return topk.score[a] > topk.score[b]; });
    for (size_t t = 0; t < K; ++t) { topk_score[t] = topk.score[idx[t]]; topk_id[t] = topk.id[idx[t]]; }
    return cells;
}

}  // extern "C"
