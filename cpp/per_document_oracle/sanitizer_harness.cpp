#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" {
uint64_t maxsim_knn_accounting(
    const float*, const uint64_t*, size_t, const float*, size_t, size_t,
    const uint32_t*, const uint32_t*, size_t, const float*, float, size_t,
    uint32_t*, float*, uint64_t*);
uint64_t maxsim_knn_throughput(
    const float*, const uint64_t*, size_t, const float*, size_t, size_t,
    const uint32_t*, const uint32_t*, size_t, const float*, float, size_t,
    uint32_t*, float*, uint64_t*);
uint64_t maxsim_full(
    const float*, const uint64_t*, size_t, const float*, size_t, size_t,
    size_t, uint32_t*, float*);
}

int main() {
    constexpr size_t D = 4, m = 3, K = 2;
    const float docs[16] = {
        1, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 1,
    };
    const uint64_t offsets[3] = {0, 2, 4};
    const float query[m * D] = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
    };
    const uint32_t order[D] = {2, 0, 3, 1};
    const uint32_t fetch[2] = {1, 2};
    float qcum[m * (D + 1)] = {};
    for (size_t i = 0; i < m; ++i)
        for (size_t z = 0; z < D; ++z) {
            float q = query[i * D + order[z]];
            qcum[i * (D + 1) + z + 1] = qcum[i * (D + 1) + z] + q * q;
        }
    uint32_t ids[K]; float scores[K]; uint64_t stats[3];
    const uint64_t error = std::numeric_limits<uint64_t>::max();
    if (maxsim_knn_accounting(docs, offsets, 2, query, m, D, order, fetch, 2,
            qcum, 1.0f, K, ids, scores, stats) == error) return 1;
    if (maxsim_knn_throughput(docs, offsets, 2, query, m, D, order, fetch, 2,
            qcum, 1.0f, K, ids, scores, stats) == error) return 2;
    if (maxsim_full(docs, offsets, 2, query, m, D, K, ids, scores) == error) return 3;
    const uint64_t bad_offsets[3] = {0, 4, 2};
    if (maxsim_full(docs, bad_offsets, 2, query, m, D, K, ids, scores) != error)
        return 4;
    return 0;
}
