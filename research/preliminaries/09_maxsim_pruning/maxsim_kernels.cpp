// Experiment 9 — MaxSim (ColBERT) pruning kernel, PDX-faithful.
//
// The BOND↔MaxSim resolution: single-vector BOND picks one per-query dim order, but
// MaxSim has m query tokens. We prune on the DOC-TOKEN axis and share ONE block-aware
// dim order across the whole query — so the per-query reorder is paid once per query,
// then every doc reuses it. Three nested pruning levels in one synchronized columnar pass:
//
//   (1) Synchronized multi-token scan  — one pass over a doc's token-columns serves all
//       m query tokens (load d_j[z] once, multiply into all q_i[z]).
//   (2) Inner-max / token pruning      — per query token i keep a valid lower bound
//          L_i = max_j ( P_ij - shrink*||q_i[k:]||*||d_j[k:]|| )      (Cauchy-Schwarz)
//       and per (i,j) an upper bound
//          U_ij = P_ij + shrink*||q_i[k:]||*||d_j[k:]||
//       Drop doc-token j when U_ij < L_i for EVERY i (dominated everywhere) and compact
//       the live-token set. The true argmax token of each i always survives, so the
//       finalized score over the live set is exact when shrink=1.
//   (3) Outer / doc pruning            — threshold T = K-th best full doc score so far.
//       Doc upper bound UB_d = Σ_i max_j U_ij. Prune the doc when UB_d < T.
//
// `shrink` ∈ [0,1] is the SINGLE recall knob. It does NOT scale the residual uniformly
// (that over-prunes good docs in the very first block, where residuals≈1 dominate the
// bound — a recall cliff). Instead it sets a k-DEPENDENT confidence ramp, ADSampling-style:
//        beta(k) = shrink + (1 - shrink) * (D - k)/D
// so beta(0)=1 (safe early, residual fully trusted) and beta(D)=shrink (tightest late, when
// the partial score is already informative). The Cauchy-Schwarz residual used in every bound
// is beta(k)·||q_i[k:]||·||d_j[k:]||.  shrink = 1.0 → beta≡1 → the SAFE bound (recall 1.000,
// exact top-K);  shrink < 1.0 → approximate pruning that trades recall smoothly.
// Residuals come from unit-norm tokens: ||x[k:]||^2 = 1 - Σ_{scanned} x^2 (order-independent).
//
// Layout: each doc is stored dim-major WITHIN the doc (the per-doc analogue of a PDX
// vectorgroup): value of dim z, token j of doc d is  docs[doc_offsets[d]*D + z*n_d + j].
//
// Build: clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -o maxsim_kernels.so maxsim_kernels.cpp
#include <cstdint>
#include <cstddef>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

extern "C" {

// Min-tracked top-K of the LARGEST doc scores. T (=K-th best) is the smallest kept score
// once K are present, else -inf. Prune a doc when its upper bound < T.
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
        size_t mn = 0;                              // slot holding the current K-th best
        for (size_t i = 1; i < K; ++i) if (score[i] < score[mn]) mn = i;
        if (s > score[mn]) { score[mn] = s; id[mn] = v; }
    }
};

// maxsim_knn — score n_docs docs against an m-token query, return top-K by MaxSim.
//   docs          : flat per-doc dim-major store (see layout note above)
//   doc_offsets   : length n_docs+1, token offsets (cumulative n_d)
//   query         : m x D row-major
//   order         : D dim indices — the shared per-query scan order (block-aware BOND / ident)
//   fetch_schedule: PDX adaptive cadence; predicate evaluated once per dim-block
//   Qcum          : m x (D+1) prefix of q_i^2 ALONG `order` (Qcum[i,t]=Σ_{u<t} q_i[order[u]]^2)
//   shrink        : Cauchy-Schwarz residual scale / recall knob (1.0 = safe)
//   topk_id/score : outputs (length K)
//   stats         : [cells_scanned (i,j,dim), docs_pruned, tokens_pruned]
// Returns cells_scanned.
unsigned long long maxsim_knn(
        const float* docs, const uint64_t* doc_offsets, size_t n_docs,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
        const float* Qcum, float shrink, size_t K,
        uint32_t* topk_id, float* topk_score, unsigned long long* stats) {

    // scratch sized to the largest doc
    size_t max_nd = 0;
    for (size_t d = 0; d < n_docs; ++d)
        max_nd = std::max(max_nd, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));

    std::vector<float>    P(m * max_nd);      // partial inner products P[i*nd + j]
    std::vector<float>    sumsq_d(max_nd);    // Σ_scanned d_j^2  → residual ||d_j[k:]||
    std::vector<uint32_t> live(max_nd);
    std::vector<float>    Li(m), resq(m);

    TopK topk(K);
    unsigned long long cells = 0, docs_pruned = 0, tokens_pruned = 0;
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
            cells += (unsigned long long)(end - cur) * n_live * m;
            cur = end;

            // residual norms ||q_i[cur:]|| = sqrt(1 - Qcum[i,cur]), scaled by the
            // k-dependent confidence ramp beta(cur) (= 1 at cur=0, = shrink at cur=D)
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

            // (3) doc upper bound UB = Σ_i max_j U_ij over the (surviving) live set
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

// maxsim_full — brute-force baseline: full synchronized scan over ALL dims and ALL doc tokens,
// no pruning, no bounds, no live-set. Same per-doc dim-major columnar layout as maxsim_knn, so it
// is the honest "pruning off" comparison (isolates what the bound machinery buys). Returns total
// cells scanned (always n_d·D·m summed over docs). Recall is 1.0 by construction.
unsigned long long maxsim_full(
        const float* docs, const uint64_t* doc_offsets, size_t n_docs,
        const float* query, size_t m, size_t D, size_t K,
        uint32_t* topk_id, float* topk_score) {

    size_t max_nd = 0;
    for (size_t d = 0; d < n_docs; ++d)
        max_nd = std::max(max_nd, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));

    std::vector<float> P(m * max_nd);
    TopK topk(K);
    unsigned long long cells = 0;
    const float NEG = -std::numeric_limits<float>::infinity();

    for (size_t d = 0; d < n_docs; ++d) {
        size_t nd = (size_t)(doc_offsets[d + 1] - doc_offsets[d]);
        const float* base = docs + (size_t)doc_offsets[d] * D;
        if (nd == 0) continue;
        for (size_t x = 0; x < m * nd; ++x) P[x] = 0.0f;

        for (size_t z = 0; z < D; ++z) {              // natural order, every dimension
            const float* col = base + z * nd;
            for (size_t j = 0; j < nd; ++j) {
                float dv = col[j];
                float* Pj = P.data() + j;
                for (size_t i = 0; i < m; ++i) Pj[i * nd] += query[i * D + z] * dv;
            }
        }
        cells += (unsigned long long)D * nd * m;

        float score = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            float mx = NEG;
            const float* Pi = P.data() + i * nd;
            for (size_t j = 0; j < nd; ++j) if (Pi[j] > mx) mx = Pi[j];
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
