#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" uint64_t fused_panel_maxsim_brute(
    const float*, const uint64_t*, size_t, const uint64_t*, const uint64_t*,
    const float*, size_t, size_t, size_t, int, uint32_t*, float*);
using BondFn = uint64_t (*)(
    const float*, const uint64_t*, size_t, const uint64_t*, const uint64_t*,
    const float*, size_t, size_t, const uint32_t*, const float*, const uint32_t*,
    size_t, float, float, size_t, int, uint32_t*, float*, uint64_t*);
extern "C" uint64_t fused_panel_maxsim_bond(
    const float*, const uint64_t*, size_t, const uint64_t*, const uint64_t*,
    const float*, size_t, size_t, const uint32_t*, const float*, const uint32_t*,
    size_t, float, float, size_t, int, uint32_t*, float*, uint64_t*);
extern "C" uint64_t fused_panel_maxsim_bond_cheap(
    const float*, const uint64_t*, size_t, const uint64_t*, const uint64_t*,
    const float*, size_t, size_t, const uint32_t*, const float*, const uint32_t*,
    size_t, float, float, size_t, int, uint32_t*, float*, uint64_t*);
extern "C" uint64_t fused_panel_maxsim_bond_token(
    const float*, const uint64_t*, size_t, const uint64_t*, const uint64_t*,
    const float*, size_t, size_t, const uint32_t*, const float*, const uint32_t*,
    size_t, float, float, size_t, int, uint32_t*, float*, uint64_t*);

int main() {
    constexpr size_t D = 4, K = 2;
    std::vector<float> data(32 * D, 0.25f);
    const uint64_t groups[2] = {0, 32};
    const uint64_t docs[3] = {0, 16, 32};
    const uint64_t group_docs[2] = {0, 2};
    const uint32_t order[D] = {2, 0, 3, 1};
    const uint32_t checkpoints[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t ids[K]; float scores[K]; uint64_t stats[3];
    const uint64_t error = std::numeric_limits<uint64_t>::max();

    for (size_t m : {size_t(1), size_t(24), size_t(25), size_t(192),
                     size_t(193), size_t(257)}) {
        std::vector<float> query(m * D, 0.25f);
        std::vector<float> qcum(m * (D + 1), 0.0f);
        for (size_t i = 0; i < m; ++i)
            for (size_t z = 0; z < D; ++z) {
                float q = query[i * D + order[z]];
                qcum[i * (D + 1) + z + 1] = qcum[i * (D + 1) + z] + q * q;
            }
        if (fused_panel_maxsim_brute(data.data(), groups, 1, docs, group_docs,
                query.data(), m, D, K, 2, ids, scores) == error) return 1;
        for (BondFn fn : {fused_panel_maxsim_bond, fused_panel_maxsim_bond_cheap,
                          fused_panel_maxsim_bond_token})
            if (fn(data.data(), groups, 1, docs, group_docs, query.data(), m, D,
                    order, qcum.data(), checkpoints, 8, 1.0f,
                    -std::numeric_limits<float>::infinity(), K, 2,
                    ids, scores, stats) == error) return 2;
    }

    std::vector<float> query(D, 0.25f);
    float qcum[D + 1] = {0, 0.0625f, 0.125f, 0.1875f, 0.25f};
    const uint64_t bad_docs[3] = {0, 17, 32};
    if (fused_panel_maxsim_brute(data.data(), groups, 1, bad_docs, group_docs,
            query.data(), 1, D, K, 1, ids, scores) != error) return 3;
    if (fused_panel_maxsim_bond(data.data(), groups, 1, docs, group_docs,
            query.data(), 1, D, order, qcum, checkpoints, 9, 1.0f,
            -std::numeric_limits<float>::infinity(), K, 1,
            ids, scores, stats) != error) return 4;
    return 0;
}
