#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace py = pybind11;

namespace {

void validate_values(const py::array &values, const char *name) {
    if (values.ndim() != 2) {
        throw py::value_error(std::string(name) + " must be a 2D array");
    }
    if (!values.dtype().is(py::dtype::of<float>())) {
        throw py::type_error(std::string(name) + " must have dtype float32");
    }
    if ((values.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
}

void validate_offsets(
    const py::array &offsets,
    const char *name,
    py::ssize_t token_count,
    bool require_nonempty_items) {
    if (offsets.ndim() != 1) {
        throw py::value_error(std::string(name) + " must be a 1D array");
    }
    if (!offsets.dtype().is(py::dtype::of<std::int64_t>())) {
        throw py::type_error(std::string(name) + " must have dtype int64");
    }
    if ((offsets.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (offsets.shape(0) == 0) {
        throw py::value_error(std::string(name) + " must contain at least one offset");
    }

    const auto *data = static_cast<const std::int64_t *>(offsets.data());
    if (data[0] != 0) {
        throw py::value_error(std::string(name) + " must start at zero");
    }

    for (py::ssize_t index = 1; index < offsets.shape(0); ++index) {
        const std::int64_t previous = data[index - 1];
        const std::int64_t current = data[index];
        if (current < previous) {
            throw py::value_error(std::string(name) + " must be nondecreasing");
        }
        if (current < 0 || current > token_count) {
            throw py::value_error(std::string(name) + " contains an out-of-range offset");
        }
        if (require_nonempty_items && current == previous) {
            throw py::value_error(std::string(name) + " must describe non-empty documents");
        }
    }

    if (data[offsets.shape(0) - 1] != token_count) {
        throw py::value_error(
            std::string(name) + " must end at the number of packed token vectors");
    }
}

template <typename Accumulator>
inline Accumulator dot_product(
    const float *query_token,
    const float *document_token,
    py::ssize_t dimension) noexcept {
    Accumulator dot = Accumulator{0};
#if defined(_OPENMP)
#pragma omp simd reduction(+ : dot)
#endif
    for (py::ssize_t component = 0; component < dimension; ++component) {
        dot += static_cast<Accumulator>(query_token[component]) *
            static_cast<Accumulator>(document_token[component]);
    }
    return dot;
}

template <typename Accumulator>
py::array_t<Accumulator> maxsim_scores_impl(
    const py::array &document_values,
    const py::array &document_offsets,
    const py::array &query_values,
    const py::array &query_offsets) {
    validate_values(document_values, "document_values");
    validate_values(query_values, "query_values");

    const py::ssize_t dimension = document_values.shape(1);
    if (dimension == 0) {
        throw py::value_error("embedding dimension must be greater than zero");
    }
    if (query_values.shape(1) != dimension) {
        throw py::value_error("document and query embedding dimensions must match");
    }

    validate_offsets(
        document_offsets,
        "document_offsets",
        document_values.shape(0),
        true);
    validate_offsets(query_offsets, "query_offsets", query_values.shape(0), false);

    const py::ssize_t document_count = document_offsets.shape(0) - 1;
    const py::ssize_t query_count = query_offsets.shape(0) - 1;
    if (document_count != 0 &&
        query_count > std::numeric_limits<py::ssize_t>::max() / document_count) {
        throw py::value_error("score matrix is too large");
    }

    py::array_t<Accumulator> output({query_count, document_count});
    Accumulator *scores = output.mutable_data();

    const auto *documents = static_cast<const float *>(document_values.data());
    const auto *queries = static_cast<const float *>(query_values.data());
    const auto *document_starts =
        static_cast<const std::int64_t *>(document_offsets.data());
    const auto *query_starts = static_cast<const std::int64_t *>(query_offsets.data());
    const py::ssize_t pair_count = query_count * document_count;

    {
        py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t pair_index = 0; pair_index < pair_count; ++pair_index) {
            const py::ssize_t query_index = pair_index / document_count;
            const py::ssize_t document_index = pair_index % document_count;
            Accumulator score = Accumulator{0};

            for (std::int64_t query_token_index = query_starts[query_index];
                 query_token_index < query_starts[query_index + 1];
                 ++query_token_index) {
                const float *query_token = queries + query_token_index * dimension;
                Accumulator best = -std::numeric_limits<Accumulator>::infinity();

                for (std::int64_t document_token_index =
                         document_starts[document_index];
                     document_token_index < document_starts[document_index + 1];
                     ++document_token_index) {
                    const float *document_token =
                        documents + document_token_index * dimension;
                    const Accumulator dot = dot_product<Accumulator>(
                        query_token,
                        document_token,
                        dimension);
                    if (dot > best || std::isnan(dot)) {
                        best = dot;
                    }
                    if (std::isnan(best)) {
                        break;
                    }
                }

                score += best;
                if (std::isnan(score)) {
                    break;
                }
            }

            scores[pair_index] = score;
        }
    }

    return output;
}

py::array_t<float> maxsim_scores(
    const py::array &document_values,
    const py::array &document_offsets,
    const py::array &query_values,
    const py::array &query_offsets) {
    return maxsim_scores_impl<float>(
        document_values,
        document_offsets,
        query_values,
        query_offsets);
}

py::array_t<double> maxsim_scores_f64(
    const py::array &document_values,
    const py::array &document_offsets,
    const py::array &query_values,
    const py::array &query_offsets) {
    return maxsim_scores_impl<double>(
        document_values,
        document_offsets,
        query_values,
        query_offsets);
}

}  // namespace

PYBIND11_MODULE(_exact_maxsim, module) {
    module.doc() = "Exact fused ColBERT MaxSim scoring for packed float32 embeddings.";
#if defined(_OPENMP)
    module.attr("openmp_enabled") = true;
#else
    module.attr("openmp_enabled") = false;
#endif
    module.def(
        "maxsim_scores",
        &maxsim_scores,
        py::arg("document_values").noconvert(),
        py::arg("document_offsets").noconvert(),
        py::arg("query_values").noconvert(),
        py::arg("query_offsets").noconvert(),
        R"doc(
Compute exact inner-product MaxSim scores for all query-document pairs.

The output has shape (number_of_queries, number_of_documents). Input values
must be C-contiguous float32 matrices, and offsets must be C-contiguous int64
vectors that start at zero and end at the corresponding matrix row count.
)doc");
    module.def(
        "maxsim_scores_f64",
        &maxsim_scores_f64,
        py::arg("document_values").noconvert(),
        py::arg("document_offsets").noconvert(),
        py::arg("query_values").noconvert(),
        py::arg("query_offsets").noconvert(),
        R"doc(
Compute exact MaxSim scores with float64 products and accumulation.

Inputs remain packed float32 embeddings. The float64 output is intended for
precision-matched comparison with exact-safe bound kernels that use double
partial sums.
)doc");
}
