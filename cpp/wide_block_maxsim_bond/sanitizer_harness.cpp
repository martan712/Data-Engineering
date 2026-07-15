#include <cstddef>
#include <cstdint>
#include <limits>

extern "C" uint64_t wide_block_maxsim_accounting(
    const float*, const uint64_t*, size_t, const uint64_t*, size_t,
    const uint64_t*, const float*, size_t, size_t, const uint32_t*,
    const uint32_t*, size_t, const float*, float, float, size_t,
    uint32_t*, float*, uint64_t*, uint64_t*, uint64_t*);
extern "C" uint64_t wide_block_maxsim_throughput(
    const float*, const uint64_t*, size_t, const uint64_t*, size_t,
    const uint64_t*, const float*, size_t, size_t, const uint32_t*,
    const uint32_t*, size_t, const float*, float, float, size_t,
    uint32_t*, float*, uint64_t*, uint64_t*, uint64_t*);

int main() {
    constexpr size_t D = 4, m = 3, K = 2;
    const float data[16] = {
        1, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 1,
    };
    const uint64_t group_offsets[2] = {0, 4};
    const uint64_t doc_offsets[3] = {0, 2, 4};
    const uint64_t group_docs[2] = {0, 2};
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
    auto call = [&](auto fn, const uint64_t* gd) {
        return fn(data, group_offsets, 1, doc_offsets, 2, gd, query, m, D,
                  order, fetch, 2, qcum, 1.0f, -std::numeric_limits<float>::infinity(),
                  K, ids, scores, stats, nullptr, nullptr);
    };
    if (call(wide_block_maxsim_accounting, group_docs) == error) return 1;
    if (call(wide_block_maxsim_throughput, group_docs) == error) return 2;
    const uint64_t bad_group_docs[2] = {0, 1};
    if (call(wide_block_maxsim_accounting, bad_group_docs) != error) return 3;
    return 0;
}
