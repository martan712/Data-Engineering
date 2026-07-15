#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace native_validation {

static constexpr uint64_t ERROR = std::numeric_limits<uint64_t>::max();

inline bool checked_product(size_t a, size_t b) {
    return a == 0 || b <= std::numeric_limits<size_t>::max() / a;
}

inline bool offsets(const uint64_t* values, size_t count) {
    if (values == nullptr || count == 0 || values[0] != 0) return false;
    for (size_t i = 1; i < count; ++i) {
        if (values[i] < values[i - 1] ||
            values[i] > std::numeric_limits<size_t>::max()) return false;
    }
    return true;
}

inline bool finite_values(const float* values, size_t count) {
    if (values == nullptr) return false;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}

inline bool query(const float* values, size_t m, size_t D) {
    return m > 0 && D > 0 && checked_product(m, D) &&
           finite_values(values, m * D);
}

inline bool order(const uint32_t* values, size_t D) {
    if (values == nullptr || D == 0 || D > std::numeric_limits<uint32_t>::max())
        return false;
    for (size_t i = 0; i < D; ++i) {
        if (values[i] >= D) return false;
        for (size_t j = 0; j < i; ++j)
            if (values[j] == values[i]) return false;
    }
    return true;
}

inline bool qcum(const float* values, size_t m, size_t D) {
    if (values == nullptr || D == std::numeric_limits<size_t>::max() ||
        !checked_product(m, D + 1)) return false;
    constexpr float eps = 8.0f * std::numeric_limits<float>::epsilon();
    for (size_t i = 0; i < m; ++i) {
        const float* row = values + i * (D + 1);
        if (!std::isfinite(row[0]) || row[0] != 0.0f) return false;
        for (size_t z = 1; z <= D; ++z) {
            if (!std::isfinite(row[z]) || row[z] + eps < row[z - 1]) return false;
        }
    }
    return true;
}

inline bool qcum_matches(
        const float* values, const float* query_values,
        const uint32_t* dimension_order, size_t m, size_t D) {
    if (!qcum(values, m, D)) return false;
    for (size_t i = 0; i < m; ++i) {
        float total = 0.0f;
        for (size_t z = 0; z < D; ++z) {
            float q = query_values[i * D + dimension_order[z]];
            total += q * q;
            float observed = values[i * (D + 1) + z + 1];
            float tolerance = 8e-7f + 8e-6f * std::fabs(total);
            if (std::fabs(observed - total) > tolerance) return false;
        }
    }
    return true;
}

inline bool scalar_parameters(float shrink, float tau_seed) {
    return std::isfinite(shrink) && shrink >= 0.0f && shrink <= 1.0f &&
           !std::isnan(tau_seed) && tau_seed != std::numeric_limits<float>::infinity();
}

inline bool document_corpus(
        const float* data, const uint64_t* doc_offsets, size_t n_docs,
        size_t D) {
    if (data == nullptr || n_docs == 0 ||
        n_docs > std::numeric_limits<uint32_t>::max() ||
        !offsets(doc_offsets, n_docs + 1) ||
        !checked_product(static_cast<size_t>(doc_offsets[n_docs]), D)) return false;
    for (size_t d = 0; d < n_docs; ++d) {
        uint64_t length = doc_offsets[d + 1] - doc_offsets[d];
        if (length > std::numeric_limits<uint32_t>::max()) return false;
    }
    return true;
}

inline bool scratch_extents(const uint64_t* offsets, size_t count, size_t factor) {
    if (offsets == nullptr || count == 0) return false;
    for (size_t i = 0; i + 1 < count; ++i) {
        size_t length = static_cast<size_t>(offsets[i + 1] - offsets[i]);
        if (!checked_product(length, factor)) return false;
    }
    return true;
}

inline bool grouped_corpus(
        const float* data, const uint64_t* group_offsets, size_t n_groups,
        const uint64_t* doc_offsets, size_t n_docs,
        const uint64_t* group_doc_starts, size_t D, size_t alignment = 1) {
    if (data == nullptr || n_groups == 0 || n_docs == 0 || alignment == 0 ||
        n_docs > std::numeric_limits<uint32_t>::max() ||
        !offsets(group_offsets, n_groups + 1) ||
        !offsets(doc_offsets, n_docs + 1) ||
        !offsets(group_doc_starts, n_groups + 1) ||
        group_doc_starts[n_groups] != n_docs ||
        group_offsets[n_groups] != doc_offsets[n_docs] ||
        !checked_product(static_cast<size_t>(group_offsets[n_groups]), D)) return false;

    for (size_t g = 0; g <= n_groups; ++g) {
        if (group_doc_starts[g] > n_docs ||
            group_offsets[g] != doc_offsets[group_doc_starts[g]] ||
            group_offsets[g] % alignment != 0) return false;
        if (g < n_groups &&
            group_offsets[g + 1] - group_offsets[g] >
                std::numeric_limits<uint32_t>::max()) return false;
    }
    for (size_t d = 0; d <= n_docs; ++d)
        if (doc_offsets[d] % alignment != 0) return false;
    return true;
}

}  // namespace native_validation
