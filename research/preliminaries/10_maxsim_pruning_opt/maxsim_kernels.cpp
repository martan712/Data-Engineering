// Experiment 10 — OPTIMIZED MaxSim (ColBERT) pruning kernel (exp 09 + perf fixes).
//
// Exp 09 found a true dense brute force beats the (scalar, live-set-from-dim-0) pruning kernel
// in wall-clock by ~1.4–1.75×, because pruning paid the gather indirection from the very first
// dimension even though MaxSim prunes LATE, plus per-block bound overhead, on a non-vectorizable
// loop. Exp 10 keeps the same algorithm/bounds (and the same recall-1.000 guarantee) but rewrites
// the kernels for speed:
//
//   (A) Token-major partials  P[j*m + i]  → the inner per-query-token loop is a CONTIGUOUS axpy
//       of length m, so both kernels auto-vectorize (SIMD FMA). Brute and pruning share this layout
//       so the only difference timed is the pruning logic, not vectorization tricks.
//   (B) PDX-style Warmup → Prune split. While the live token set is still ~full we scan ALL tokens
//       DENSELY (brute speed, no indirection). Only once ≥ SELECTIVITY of tokens are prunable do we
//       switch to the positional (survivor-only) scan. Since MaxSim prunes late, most dims run at
//       brute speed and the cheap positional tail handles the rest.
//   (C) Bound evaluation is skipped for the first WARMUP_MIN_FRAC of dimensions (nothing prunes that
//       early — score compression), removing the early per-block O(nd·m) bound overhead.
//
// Same recall knob as exp 09: `shrink`∈[0,1] sets the k-dependent Cauchy-Schwarz residual ramp
// beta(k)=shrink+(1-shrink)(D-k)/D  (shrink=1.0 → exact safe bound, recall 1.000).
// Layout: each doc dim-major within the doc — docs[doc_offsets[d]*D + z*n_d + j].
//
// Build: clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -o maxsim_kernels.so maxsim_kernels.cpp
#include <cstdint>
#include <cstddef>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

extern "C" {

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

// maxsim_knn — optimized pruner. Same signature/semantics as exp 09 (P is token-major internally).
//   stats = [cells_scanned (i,j,dim), docs_pruned, tokens_pruned]
unsigned long long maxsim_knn(
        const float* docs, const uint64_t* doc_offsets, size_t n_docs,
        const float* query, size_t m, size_t D,
        const uint32_t* order, const uint32_t* fetch_schedule, size_t n_fetch,
        const float* Qcum, float shrink, size_t K,
        uint32_t* topk_id, float* topk_score, unsigned long long* stats) {

    size_t max_nd = 0;
    for (size_t d = 0; d < n_docs; ++d)
        max_nd = std::max(max_nd, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));

    std::vector<float>    P(m * max_nd);          // token-major: P[j*m + i]
    std::vector<float>    sumsq_d(max_nd);
    std::vector<float>    resd(max_nd);
    std::vector<uint32_t> pos(max_nd);
    std::vector<float>    Li(m), resq(m), qz(m);

    TopK topk(K);
    unsigned long long cells = 0, docs_pruned = 0, tokens_pruned = 0;
    const float NEG = -std::numeric_limits<float>::infinity();
    const float  SELECTIVITY    = 0.5f;   // switch to positional once ≥50% of tokens are prunable
    const size_t WARMUP_MIN_DIM = D / 4;  // skip bound eval before this many dims (nothing prunes early)

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

            // ---- scan this block (dense over all tokens during warmup; survivors once positional) ----
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
                cells += (unsigned long long)(end - cur) * nd * m;
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
                cells += (unsigned long long)(end - cur) * n_live * m;
            }
            cur = end;

            // ---- bound eval (skipped while still in the early no-prune warmup window) ----
            if (cur < WARMUP_MIN_DIM && !positional) continue;

            float beta = shrink + (1.0f - shrink) * ((float)(D - cur) / (float)D);
            for (size_t i = 0; i < m; ++i) {
                float s = 1.0f - Qcum[i * (D + 1) + cur];
                resq[i] = beta * std::sqrt(s > 0.0f ? s : 0.0f);
            }

            // L_i = max over the active token set of (P_ij − resq_i·resd_j); also refresh resd.
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
                if ((nd - n_live) >= (size_t)(SELECTIVITY * nd)) positional = true;  // enough pruned → switch
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

        // finalize: score = Σ_i max_j P_ij over survivors (exact when shrink=1).
        // If we never left warmup, pos[] holds the last-built survivor set (argmax tokens always survive).
        const uint32_t* set = pos.data(); size_t ns = n_live;
        std::vector<uint32_t> all;
        if (ns == 0) {                          // never built positions (e.g. D ≤ WARMUP_MIN): use all
            all.resize(nd); for (size_t j = 0; j < nd; ++j) all[j] = (uint32_t)j; set = all.data(); ns = nd;
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

// maxsim_full — brute-force baseline, token-major + vectorizable (same layout as the pruner, so the
// only timed difference is the pruning logic). Full scan, no bounds. Returns total cells.
unsigned long long maxsim_full(
        const float* docs, const uint64_t* doc_offsets, size_t n_docs,
        const float* query, size_t m, size_t D, size_t K,
        uint32_t* topk_id, float* topk_score) {

    size_t max_nd = 0;
    for (size_t d = 0; d < n_docs; ++d)
        max_nd = std::max(max_nd, (size_t)(doc_offsets[d + 1] - doc_offsets[d]));

    std::vector<float> P(m * max_nd);   // token-major P[j*m + i]
    std::vector<float> qz(m);
    TopK topk(K);
    unsigned long long cells = 0;
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
        cells += (unsigned long long)D * nd * m;

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
