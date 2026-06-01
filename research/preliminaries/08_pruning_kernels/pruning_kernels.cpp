// Experiment 8 — pruning kernels for BOND vs ADSampling, global vs blocked layout.
//
// Single-vector L2 k-NN with incremental dimension scanning + early termination.
// No materializing copy: the dimension scan order is passed as an index array and
// the kernels read the selected dimension-rows directly from the layout.
//
// Layouts:
//   global  : G[d*N + v]              — dim-major over all N vectors (each dim contiguous, N floats)
//   blocked : Bk[blk*BS*D + d*BS + j] — blocks of BS vectors, dim-major within a block (BS floats/dim)
//
// Build:  clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -o pruning_kernels.so pruning_kernels.cpp
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>

extern "C" {

// ---------- Part A: pure access cost (no pruning) ----------
// Partial inner product over the first k dims of `order`, for all N vectors.
// Isolates the memory-access cost of scanning dims in a given (possibly scattered) order.

void scan_partial_global(const float* G, const float* q, const uint32_t* order,
                         size_t k, size_t N, size_t D, float* out) {
    (void)D;
    std::memset(out, 0, N * sizeof(float));
    for (size_t i = 0; i < k; ++i) {
        uint32_t d = order[i];
        float qd = q[d];
        const float* col = G + (size_t)d * N;        // contiguous run of N floats
        for (size_t v = 0; v < N; ++v) out[v] += qd * col[v];
    }
}

void scan_partial_blocked(const float* Bk, const float* q, const uint32_t* order,
                          size_t k, size_t N, size_t D, size_t BS, float* out) {
    std::memset(out, 0, N * sizeof(float));
    size_t nb = N / BS;
    for (size_t b = 0; b < nb; ++b) {
        const float* bp = Bk + b * BS * D;
        float* o = out + b * BS;
        for (size_t i = 0; i < k; ++i) {
            uint32_t d = order[i];
            float qd = q[d];
            const float* col = bp + (size_t)d * BS;   // BS floats (256B for BS=64)
            for (size_t j = 0; j < BS; ++j) o[j] += qd * col[j];
        }
    }
}

// ---------- Part B: k-NN L2 with early termination (vertical block scan) ----------
// Generic over the prune bound via `ratios[visited]`:
//   prune candidate when  partial_sqdist(visited) > kth_best_dist * ratios[visited]
//   - exact / monotone bound (recall=1):  ratios[v] = 1 for all v
//   - ADSampling (approximate):           ratios[v] = (v/D)*(1+alpha/sqrt(v))^2  (data+query pre-rotated)
// `order` is the dimension scan order (length D). Returns total dims scanned over all vectors.

unsigned long long knn_l2_blocked(const float* Bk, const float* q, const uint32_t* order,
                                  const float* ratios, size_t N, size_t D, size_t BS,
                                  size_t knn, uint32_t* topk_id, float* topk_dist) {
    std::vector<float>    best_d(knn, 1e30f);
    std::vector<uint32_t> best_i(knn, 0xffffffffu);
    float threshold = 1e30f;
    auto update_topk = [&](uint32_t id, float dist) {
        if (dist >= threshold) return;
        size_t mx = 0; for (size_t t = 1; t < knn; ++t) if (best_d[t] > best_d[mx]) mx = t;
        best_d[mx] = dist; best_i[mx] = id;
        float th = 0; for (size_t t = 0; t < knn; ++t) if (best_d[t] > th) th = best_d[t];
        threshold = th;
    };

    unsigned long long total_dims = 0;
    size_t nb = N / BS;
    std::vector<float>    partial(BS);
    std::vector<uint32_t> live(BS);       // dense list of still-alive lanes (compacted)
    std::vector<uint32_t> scanned(BS);

    for (size_t b = 0; b < nb; ++b) {
        const float* bp = Bk + b * BS * D;
        for (size_t j = 0; j < BS; ++j) { partial[j] = 0.0f; live[j] = (uint32_t)j; scanned[j] = 0; }
        size_t n_live = BS;

        for (size_t i = 0; i < D && n_live > 0; ++i) {
            uint32_t d = order[i];
            float qd = q[d];
            const float* col = bp + (size_t)d * BS;
            // accumulate only over the dense live set (pruned lanes skipped entirely)
            for (size_t a = 0; a < n_live; ++a) {
                uint32_t j = live[a];
                float diff = qd - col[j];
                partial[j] += diff * diff;
                scanned[j] = (uint32_t)(i + 1);
            }
            // compact: keep only lanes still within the bound
            float bound = threshold * ratios[i + 1];
            size_t w = 0;
            for (size_t a = 0; a < n_live; ++a) {
                uint32_t j = live[a];
                if (partial[j] <= bound) live[w++] = j;
            }
            n_live = w;
        }
        for (size_t a = 0; a < n_live; ++a) {
            uint32_t j = live[a];
            update_topk((uint32_t)(b * BS + j), partial[j]);
        }
        for (size_t j = 0; j < BS; ++j) total_dims += scanned[j];
    }

    std::vector<size_t> idx(knn);
    for (size_t t = 0; t < knn; ++t) idx[t] = t;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t c) { return best_d[a] < best_d[c]; });
    for (size_t t = 0; t < knn; ++t) { topk_dist[t] = best_d[idx[t]]; topk_id[t] = best_i[idx[t]]; }
    return total_dims;
}

}  // extern "C"
