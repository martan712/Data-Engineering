#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kPositiveInfinity = std::numeric_limits<double>::infinity();
constexpr double kNegativeInfinity = -std::numeric_limits<double>::infinity();

double add_up(double left, double right) noexcept {
    return std::nextafter(left + right, kPositiveInfinity);
}

double add_down(double left, double right) noexcept {
    return std::nextafter(left + right, kNegativeInfinity);
}

double subtract_down(double left, double right) noexcept {
    return std::nextafter(left - right, kNegativeInfinity);
}

double divide_up(double numerator, double denominator) noexcept {
    if (numerator == 0.0) {
        return 0.0;
    }
    return std::nextafter(numerator / denominator, kPositiveInfinity);
}

double multiply_up(double left, double right) noexcept {
    if (left == 0.0 || right == 0.0) {
        return 0.0;
    }
    return std::nextafter(left * right, kPositiveInfinity);
}

double sqrt_up(double value) noexcept {
    if (value == 0.0) {
        return 0.0;
    }
    return std::nextafter(std::sqrt(value), kPositiveInfinity);
}

double summation_gamma_up(std::size_t term_count) {
    const long double unit_roundoff =
        static_cast<long double>(std::numeric_limits<double>::epsilon()) / 2.0L;
    const long double scaled_roundoff =
        static_cast<long double>(term_count) * unit_roundoff;
    if (scaled_roundoff >= 0.5L) {
        throw py::value_error(
            "embedding dimension is too large for a finite summation error bound");
    }
    const long double gamma = scaled_roundoff / (1.0L - scaled_roundoff);
    double rounded = static_cast<double>(gamma);
    if (static_cast<long double>(rounded) < gamma) {
        rounded = std::nextafter(rounded, kPositiveInfinity);
    }
    return std::nextafter(rounded, kPositiveInfinity);
}

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

    const auto *data = static_cast<const float *>(values.data());
    for (py::ssize_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(data[index])) {
            throw py::value_error(std::string(name) + " must contain only finite values");
        }
    }
}

std::vector<std::size_t> validate_offsets(
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

    std::vector<std::size_t> result(static_cast<std::size_t>(offsets.shape(0)));
    result[0] = 0;
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
        result[static_cast<std::size_t>(index)] = static_cast<std::size_t>(current);
    }

    if (data[offsets.shape(0) - 1] != token_count) {
        throw py::value_error(
            std::string(name) + " must end at the number of packed token vectors");
    }
    return result;
}

std::vector<std::size_t> validate_checkpoints(
    const py::array &checkpoints,
    std::size_t dimension) {
    if (checkpoints.ndim() != 1) {
        throw py::value_error("checkpoints must be a 1D array");
    }
    if (!checkpoints.dtype().is(py::dtype::of<std::int64_t>())) {
        throw py::type_error("checkpoints must have dtype int64");
    }
    if ((checkpoints.flags() & py::array::c_style) == 0) {
        throw py::value_error("checkpoints must be C-contiguous");
    }
    if (checkpoints.shape(0) == 0) {
        throw py::value_error("checkpoints must not be empty");
    }

    const auto *data = static_cast<const std::int64_t *>(checkpoints.data());
    std::vector<std::size_t> result(static_cast<std::size_t>(checkpoints.shape(0)));
    std::int64_t previous = 0;
    for (py::ssize_t index = 0; index < checkpoints.shape(0); ++index) {
        const std::int64_t current = data[index];
        if (current <= previous) {
            throw py::value_error("checkpoints must be strictly increasing and positive");
        }
        if (current > static_cast<std::int64_t>(dimension)) {
            throw py::value_error("checkpoints must not exceed the embedding dimension");
        }
        result[static_cast<std::size_t>(index)] = static_cast<std::size_t>(current);
        previous = current;
    }
    if (result.back() != dimension) {
        throw py::value_error("checkpoints must end at the embedding dimension");
    }
    return result;
}

std::size_t checked_size_product(
    std::size_t left,
    std::size_t right,
    const char *message) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw py::value_error(message);
    }
    return left * right;
}

std::uint64_t checked_work_product(
    std::size_t first,
    std::size_t second,
    std::size_t third) {
    constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    const auto first_u64 = static_cast<std::uint64_t>(first);
    const auto second_u64 = static_cast<std::uint64_t>(second);
    const auto third_u64 = static_cast<std::uint64_t>(third);
    if (first_u64 != 0 && second_u64 > maximum / first_u64) {
        throw py::value_error("component-product diagnostic would overflow uint64");
    }
    const std::uint64_t pair_count = first_u64 * second_u64;
    if (pair_count != 0 && third_u64 > maximum / pair_count) {
        throw py::value_error("component-product diagnostic would overflow uint64");
    }
    return pair_count * third_u64;
}

std::size_t parse_count(const py::handle &value, const char *name) {
    if (!PyLong_CheckExact(value.ptr())) {
        throw py::type_error(std::string(name) + " must be an int");
    }
    const Py_ssize_t parsed = PyLong_AsSsize_t(value.ptr());
    if (parsed == -1 && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    if (parsed < 0) {
        throw py::value_error(std::string(name) + " must be nonnegative");
    }
    return static_cast<std::size_t>(parsed);
}

struct PreparedQuery {
    const float *values = nullptr;
    std::size_t token_count = 0;
    std::vector<double> residual_norms;
};

struct ExactScore {
    double score = 0.0;
    double lower_bound = 0.0;
};

struct Candidate {
    std::int64_t document_id = 0;
    double score = 0.0;
};

struct QueryResult {
    std::vector<Candidate> top_k;
    std::uint64_t documents_pruned = 0;
    std::uint64_t documents_exactly_scored = 0;
    std::uint64_t component_products = 0;
    std::uint64_t full_component_products = 0;
    std::vector<std::uint64_t> pruned_by_checkpoint;
};

class BondMaxSimIndex {
public:
    BondMaxSimIndex(
        const py::array &document_values,
        const py::array &document_offsets,
        const py::array &checkpoints) {
        validate_values(document_values, "document_values");
        if (document_values.shape(1) == 0) {
            throw py::value_error("embedding dimension must be greater than zero");
        }

        dimension_ = static_cast<std::size_t>(document_values.shape(1));
        total_document_tokens_ = static_cast<std::size_t>(document_values.shape(0));
        document_offsets_ = validate_offsets(
            document_offsets,
            "document_offsets",
            document_values.shape(0),
            true);
        document_count_ = document_offsets_.size() - 1;
        checkpoints_ = validate_checkpoints(checkpoints, dimension_);
        summation_gammas_.reserve(checkpoints_.size());
        one_minus_summation_gammas_.reserve(checkpoints_.size());
        for (const std::size_t checkpoint : checkpoints_) {
            const double gamma = summation_gamma_up(checkpoint);
            const double one_minus_gamma = std::nextafter(
                1.0 - gamma,
                kNegativeInfinity);
            if (one_minus_gamma <= 0.0) {
                throw py::value_error(
                    "embedding dimension is too large for a finite summation error bound");
            }
            summation_gammas_.push_back(gamma);
            one_minus_summation_gammas_.push_back(one_minus_gamma);
        }

        const std::size_t vertical_size = checked_size_product(
            total_document_tokens_,
            dimension_,
            "document index is too large");
        const std::size_t residual_size = checked_size_product(
            total_document_tokens_,
            checkpoints_.size(),
            "document residual-norm index is too large");
        vertical_documents_.resize(vertical_size);
        document_residual_norms_.resize(residual_size);

        const auto *source = static_cast<const float *>(document_values.data());
        py::gil_scoped_release release;
        build_vertical_documents(source);
        build_document_residual_norms();
    }

    py::dict search(
        const py::array &query_values,
        const py::array &query_offsets,
        const py::object &k_object,
        const py::object &seed_count_object) const {
        validate_values(query_values, "query_values");
        if (query_values.shape(1) != static_cast<py::ssize_t>(dimension_)) {
            throw py::value_error("document and query embedding dimensions must match");
        }
        const std::vector<std::size_t> query_starts = validate_offsets(
            query_offsets,
            "query_offsets",
            query_values.shape(0),
            false);

        const std::size_t k = parse_count(k_object, "k");
        const std::size_t seed_count = parse_count(seed_count_object, "seed_count");
        if (k == 0) {
            throw py::value_error("k must be greater than zero");
        }
        if (k > document_count_) {
            throw py::value_error("k must not exceed the document count");
        }
        if (seed_count < k) {
            throw py::value_error("seed_count must be at least k");
        }
        if (seed_count > document_count_) {
            throw py::value_error("seed_count must not exceed the document count");
        }

        const std::size_t query_count = query_starts.size() - 1;
        std::vector<std::vector<std::size_t>> seeds_by_query(
            query_count,
            std::vector<std::size_t>(seed_count));
        for (auto &seeds : seeds_by_query) {
            std::iota(seeds.begin(), seeds.end(), 0);
        }
        return execute_search(query_values, query_starts, k, seeds_by_query);
    }

    py::dict search_with_seed_ids(
        const py::array &query_values,
        const py::array &query_offsets,
        const py::object &k_object,
        const py::array &seed_ids) const {
        validate_values(query_values, "query_values");
        if (query_values.shape(1) != static_cast<py::ssize_t>(dimension_)) {
            throw py::value_error("document and query embedding dimensions must match");
        }
        const std::vector<std::size_t> query_starts = validate_offsets(
            query_offsets,
            "query_offsets",
            query_values.shape(0),
            false);
        const std::size_t query_count = query_starts.size() - 1;

        const std::size_t k = parse_count(k_object, "k");
        if (k == 0) {
            throw py::value_error("k must be greater than zero");
        }
        if (k > document_count_) {
            throw py::value_error("k must not exceed the document count");
        }
        if (seed_ids.ndim() != 2) {
            throw py::value_error("seed_ids must be a 2D array");
        }
        if (!seed_ids.dtype().is(py::dtype::of<std::int64_t>())) {
            throw py::type_error("seed_ids must have dtype int64");
        }
        if ((seed_ids.flags() & py::array::c_style) == 0) {
            throw py::value_error("seed_ids must be C-contiguous");
        }
        if (seed_ids.shape(0) != static_cast<py::ssize_t>(query_count)) {
            throw py::value_error("seed_ids rows must match the query count");
        }
        const std::size_t seed_count = static_cast<std::size_t>(seed_ids.shape(1));
        if (seed_count < k) {
            throw py::value_error("seed_ids must contain at least k columns");
        }
        if (seed_count > document_count_) {
            throw py::value_error("seed_ids must not contain more columns than documents");
        }

        const auto *seed_data = static_cast<const std::int64_t *>(seed_ids.data());
        std::vector<std::vector<std::size_t>> seeds_by_query(
            query_count,
            std::vector<std::size_t>(seed_count));
        for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
            std::vector<unsigned char> seen(document_count_, 0);
            for (std::size_t seed_index = 0; seed_index < seed_count; ++seed_index) {
                const std::int64_t document_id =
                    seed_data[query_index * seed_count + seed_index];
                if (document_id < 0 ||
                    document_id >= static_cast<std::int64_t>(document_count_)) {
                    throw py::value_error("seed_ids contains an out-of-range document ID");
                }
                const std::size_t parsed_id = static_cast<std::size_t>(document_id);
                if (seen[parsed_id] != 0) {
                    throw py::value_error("seed_ids rows must contain unique document IDs");
                }
                seen[parsed_id] = 1;
                seeds_by_query[query_index][seed_index] = parsed_id;
            }
        }
        return execute_search(query_values, query_starts, k, seeds_by_query);
    }

    py::ssize_t document_count() const noexcept {
        return static_cast<py::ssize_t>(document_count_);
    }

    py::ssize_t dimension() const noexcept {
        return static_cast<py::ssize_t>(dimension_);
    }

    py::tuple checkpoints() const {
        py::tuple result(checkpoints_.size());
        for (std::size_t index = 0; index < checkpoints_.size(); ++index) {
            result[index] = py::int_(checkpoints_[index]);
        }
        return result;
    }

private:
    py::dict execute_search(
        const py::array &query_values,
        const std::vector<std::size_t> &query_starts,
        std::size_t k,
        const std::vector<std::vector<std::size_t>> &seeds_by_query) const {
        const std::size_t query_count = query_starts.size() - 1;
        checked_size_product(query_count, k, "top-k output is too large");

        std::vector<std::uint64_t> full_work_by_query(query_count);
        std::uint64_t full_work_total = 0;
        for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
            const std::size_t query_token_count =
                query_starts[query_index + 1] - query_starts[query_index];
            const std::uint64_t query_work = checked_work_product(
                query_token_count,
                total_document_tokens_,
                dimension_);
            if (query_work > std::numeric_limits<std::uint64_t>::max() - full_work_total) {
                throw py::value_error("component-product diagnostic would overflow uint64");
            }
            full_work_by_query[query_index] = query_work;
            full_work_total += query_work;
        }

        const auto *queries = static_cast<const float *>(query_values.data());
        std::vector<PreparedQuery> prepared_queries(query_count);
        std::vector<QueryResult> query_results(query_count);
        {
            py::gil_scoped_release release;
            for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
                prepared_queries[query_index] = prepare_query(
                    queries + query_starts[query_index] * dimension_,
                    query_starts[query_index + 1] - query_starts[query_index]);
            }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (py::ssize_t query_index = 0;
                 query_index < static_cast<py::ssize_t>(query_count);
                 ++query_index) {
                const std::size_t index = static_cast<std::size_t>(query_index);
                query_results[index] = search_one(
                    prepared_queries[index],
                    k,
                    seeds_by_query[index],
                    full_work_by_query[index]);
            }
        }
        return make_python_result(query_results, query_count, k, full_work_total);
    }

    void build_vertical_documents(const float *source) {
        for (std::size_t document_id = 0; document_id < document_count_; ++document_id) {
            const std::size_t start = document_offsets_[document_id];
            const std::size_t token_count = document_offsets_[document_id + 1] - start;
            const std::size_t vertical_base = start * dimension_;
            for (std::size_t component = 0; component < dimension_; ++component) {
                for (std::size_t token = 0; token < token_count; ++token) {
                    vertical_documents_[vertical_base + component * token_count + token] =
                        source[(start + token) * dimension_ + component];
                }
            }
        }
    }

    void build_document_residual_norms() {
        const std::size_t checkpoint_count = checkpoints_.size();
        for (std::size_t document_id = 0; document_id < document_count_; ++document_id) {
            const std::size_t start = document_offsets_[document_id];
            const std::size_t token_count = document_offsets_[document_id + 1] - start;
            const std::size_t vertical_base = start * dimension_;
            const std::size_t residual_base = start * checkpoint_count;

            for (std::size_t token = 0; token < token_count; ++token) {
                document_residual_norms_[
                    residual_base + (checkpoint_count - 1) * token_count + token] = 0.0;
                std::size_t pending_checkpoint = checkpoint_count - 1;
                double suffix_squared_norm = 0.0;
                for (std::size_t component = dimension_; component-- > 0;) {
                    const double value = static_cast<double>(
                        vertical_documents_[vertical_base + component * token_count + token]);
                    suffix_squared_norm = add_up(suffix_squared_norm, value * value);
                    if (pending_checkpoint != 0 &&
                        checkpoints_[pending_checkpoint - 1] == component) {
                        document_residual_norms_[
                            residual_base +
                            (pending_checkpoint - 1) * token_count + token] =
                            sqrt_up(suffix_squared_norm);
                        --pending_checkpoint;
                    }
                }
            }
        }
    }

    PreparedQuery prepare_query(const float *values, std::size_t token_count) const {
        PreparedQuery query;
        query.values = values;
        query.token_count = token_count;
        query.residual_norms.resize(token_count * checkpoints_.size());

        for (std::size_t token = 0; token < token_count; ++token) {
            query.residual_norms[(checkpoints_.size() - 1) * token_count + token] = 0.0;
            std::size_t pending_checkpoint = checkpoints_.size() - 1;
            double suffix_squared_norm = 0.0;
            for (std::size_t component = dimension_; component-- > 0;) {
                const double value = static_cast<double>(values[token * dimension_ + component]);
                suffix_squared_norm = add_up(suffix_squared_norm, value * value);
                if (pending_checkpoint != 0 &&
                    checkpoints_[pending_checkpoint - 1] == component) {
                    query.residual_norms[
                        (pending_checkpoint - 1) * token_count + token] =
                        sqrt_up(suffix_squared_norm);
                    --pending_checkpoint;
                }
            }
        }
        return query;
    }

    ExactScore exact_document(
        const PreparedQuery &query,
        std::size_t document_id,
        std::uint64_t &component_products) const {
        const std::size_t start = document_offsets_[document_id];
        const std::size_t document_token_count = document_offsets_[document_id + 1] - start;
        const std::size_t pair_count = query.token_count * document_token_count;
        const std::size_t vertical_base = start * dimension_;
        std::vector<double> partial_scores(pair_count, 0.0);
        std::vector<double> absolute_product_sums(pair_count, 0.0);

        for (std::size_t component = 0; component < dimension_; ++component) {
            for (std::size_t query_token = 0; query_token < query.token_count; ++query_token) {
                const double query_value = static_cast<double>(
                    query.values[query_token * dimension_ + component]);
                for (std::size_t document_token = 0;
                     document_token < document_token_count;
                     ++document_token) {
                    const std::size_t pair_index =
                        query_token * document_token_count + document_token;
                    const double product = query_value * static_cast<double>(
                        vertical_documents_[
                            vertical_base + component * document_token_count + document_token]);
                    partial_scores[pair_index] += product;
                    absolute_product_sums[pair_index] += std::abs(product);
                }
            }
        }
        component_products += checked_work_product(
            query.token_count,
            document_token_count,
            dimension_);
        return finish_exact_score(
            query.token_count,
            document_token_count,
            partial_scores,
            absolute_product_sums);
    }

    ExactScore finish_exact_score(
        std::size_t query_token_count,
        std::size_t document_token_count,
        const std::vector<double> &partial_scores,
        const std::vector<double> &absolute_product_sums) const {
        ExactScore result;
        const std::size_t full_checkpoint_index = checkpoints_.size() - 1;
        for (std::size_t query_token = 0;
             query_token < query_token_count;
             ++query_token) {
            double best_score = kNegativeInfinity;
            double best_lower = kNegativeInfinity;
            for (std::size_t document_token = 0;
                 document_token < document_token_count;
                 ++document_token) {
                const std::size_t pair_index =
                    query_token * document_token_count + document_token;
                best_score = std::max(best_score, partial_scores[pair_index]);
                const double error = summation_error(
                    absolute_product_sums[pair_index],
                    full_checkpoint_index);
                const double pair_lower = subtract_down(
                    partial_scores[pair_index],
                    error);
                best_lower = std::max(best_lower, pair_lower);
            }
            result.score += best_score;
            result.lower_bound = add_down(result.lower_bound, best_lower);
        }
        return result;
    }

    double summation_error(
        double computed_absolute_sum,
        std::size_t checkpoint_index) const noexcept {
        if (computed_absolute_sum == 0.0) {
            return 0.0;
        }
        const double inflated_absolute_sum = divide_up(
            computed_absolute_sum,
            one_minus_summation_gammas_[checkpoint_index]);
        return multiply_up(
            summation_gammas_[checkpoint_index],
            inflated_absolute_sum);
    }

    double document_upper_bound(
        const PreparedQuery &query,
        std::size_t document_id,
        std::size_t checkpoint_index,
        const std::vector<double> &partial_scores,
        const std::vector<double> &absolute_product_sums) const {
        const std::size_t start = document_offsets_[document_id];
        const std::size_t document_token_count = document_offsets_[document_id + 1] - start;
        const std::size_t residual_base = start * checkpoints_.size();
        double document_upper = 0.0;

        for (std::size_t query_token = 0;
             query_token < query.token_count;
             ++query_token) {
            double best_pair_upper = kNegativeInfinity;
            const double query_residual = query.residual_norms[
                checkpoint_index * query.token_count + query_token];
            for (std::size_t document_token = 0;
                 document_token < document_token_count;
                 ++document_token) {
                const std::size_t pair_index =
                    query_token * document_token_count + document_token;
                const double document_residual = document_residual_norms_[
                    residual_base + checkpoint_index * document_token_count + document_token];
                const double residual_product = multiply_up(
                    query_residual,
                    document_residual);
                const double error = summation_error(
                    absolute_product_sums[pair_index],
                    checkpoint_index);
                const double partial_upper = add_up(
                    partial_scores[pair_index],
                    error);
                const double pair_upper = add_up(
                    partial_upper,
                    residual_product);
                best_pair_upper = std::max(best_pair_upper, pair_upper);
            }
            document_upper = add_up(document_upper, best_pair_upper);
        }
        return document_upper;
    }

    QueryResult search_one(
        const PreparedQuery &query,
        std::size_t k,
        const std::vector<std::size_t> &seed_ids,
        std::uint64_t full_component_products) const {
        QueryResult result;
        result.full_component_products = full_component_products;
        result.pruned_by_checkpoint.assign(checkpoints_.size(), 0);
        std::vector<Candidate> exact_candidates;
        exact_candidates.reserve(document_count_);
        std::vector<unsigned char> is_seed(document_count_, 0);
        std::priority_queue<double, std::vector<double>, std::greater<double>> lower_top_k;

        const auto add_lower_bound = [&](double lower_bound) {
            if (lower_top_k.size() < k) {
                lower_top_k.push(lower_bound);
            } else if (lower_bound > lower_top_k.top()) {
                lower_top_k.pop();
                lower_top_k.push(lower_bound);
            }
        };

        for (const std::size_t document_id : seed_ids) {
            is_seed[document_id] = 1;
            const ExactScore exact = exact_document(
                query,
                document_id,
                result.component_products);
            exact_candidates.push_back(
                {static_cast<std::int64_t>(document_id), exact.score});
            add_lower_bound(exact.lower_bound);
            ++result.documents_exactly_scored;
        }

        for (std::size_t document_id = 0; document_id < document_count_; ++document_id) {
            if (is_seed[document_id] != 0) {
                continue;
            }
            const std::size_t start = document_offsets_[document_id];
            const std::size_t document_token_count = document_offsets_[document_id + 1] - start;
            const std::size_t pair_count = query.token_count * document_token_count;
            const std::size_t vertical_base = start * dimension_;
            std::vector<double> partial_scores(pair_count, 0.0);
            std::vector<double> absolute_product_sums(pair_count, 0.0);
            std::size_t previous_checkpoint = 0;
            bool pruned = false;

            for (std::size_t checkpoint_index = 0;
                 checkpoint_index < checkpoints_.size();
                 ++checkpoint_index) {
                const std::size_t checkpoint = checkpoints_[checkpoint_index];
                for (std::size_t component = previous_checkpoint;
                     component < checkpoint;
                     ++component) {
                    for (std::size_t query_token = 0;
                         query_token < query.token_count;
                         ++query_token) {
                        const double query_value = static_cast<double>(
                            query.values[query_token * dimension_ + component]);
                        for (std::size_t document_token = 0;
                             document_token < document_token_count;
                             ++document_token) {
                            const std::size_t pair_index =
                                query_token * document_token_count + document_token;
                            const double product = query_value * static_cast<double>(
                                vertical_documents_[
                                    vertical_base +
                                    component * document_token_count +
                                    document_token]);
                            partial_scores[pair_index] += product;
                            absolute_product_sums[pair_index] += std::abs(product);
                        }
                    }
                }
                result.component_products += checked_work_product(
                    query.token_count,
                    document_token_count,
                    checkpoint - previous_checkpoint);

                if (checkpoint == dimension_) {
                    const ExactScore exact = finish_exact_score(
                        query.token_count,
                        document_token_count,
                        partial_scores,
                        absolute_product_sums);
                    exact_candidates.push_back(
                        {static_cast<std::int64_t>(document_id), exact.score});
                    add_lower_bound(exact.lower_bound);
                    ++result.documents_exactly_scored;
                    break;
                }

                const double upper_bound = document_upper_bound(
                    query,
                    document_id,
                    checkpoint_index,
                    partial_scores,
                    absolute_product_sums);
                if (upper_bound < lower_top_k.top()) {
                    ++result.documents_pruned;
                    ++result.pruned_by_checkpoint[checkpoint_index];
                    pruned = true;
                    break;
                }
                previous_checkpoint = checkpoint;
            }

            if (pruned) {
                continue;
            }
        }

        std::sort(
            exact_candidates.begin(),
            exact_candidates.end(),
            [](const Candidate &left, const Candidate &right) {
                if (left.score != right.score) {
                    return left.score > right.score;
                }
                return left.document_id < right.document_id;
            });
        exact_candidates.resize(k);
        result.top_k = std::move(exact_candidates);
        return result;
    }

    py::dict make_python_result(
        const std::vector<QueryResult> &query_results,
        std::size_t query_count,
        std::size_t k,
        std::uint64_t full_work_total) const {
        py::array_t<std::int64_t> ids(
            {static_cast<py::ssize_t>(query_count), static_cast<py::ssize_t>(k)});
        py::array_t<double> scores(
            {static_cast<py::ssize_t>(query_count), static_cast<py::ssize_t>(k)});
        py::array_t<std::int64_t> pruned_by_checkpoint(
            static_cast<py::ssize_t>(checkpoints_.size()));
        auto *id_data = ids.mutable_data();
        auto *score_data = scores.mutable_data();
        auto *checkpoint_data = pruned_by_checkpoint.mutable_data();
        std::fill(
            checkpoint_data,
            checkpoint_data + checkpoints_.size(),
            static_cast<std::int64_t>(0));

        std::uint64_t documents_pruned = 0;
        std::uint64_t documents_exactly_scored = 0;
        std::uint64_t component_products = 0;
        for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
            const QueryResult &query_result = query_results[query_index];
            documents_pruned += query_result.documents_pruned;
            documents_exactly_scored += query_result.documents_exactly_scored;
            component_products += query_result.component_products;
            for (std::size_t checkpoint_index = 0;
                 checkpoint_index < checkpoints_.size();
                 ++checkpoint_index) {
                checkpoint_data[checkpoint_index] += static_cast<std::int64_t>(
                    query_result.pruned_by_checkpoint[checkpoint_index]);
            }
            for (std::size_t rank = 0; rank < k; ++rank) {
                const std::size_t output_index = query_index * k + rank;
                id_data[output_index] = query_result.top_k[rank].document_id;
                score_data[output_index] = query_result.top_k[rank].score;
            }
        }

        const double work_ratio = full_work_total == 0
            ? 0.0
            : static_cast<double>(component_products) /
                static_cast<double>(full_work_total);
        py::dict output;
        output["ids"] = std::move(ids);
        output["scores"] = std::move(scores);
        output["documents_pruned"] = py::cast(documents_pruned);
        output["documents_exactly_scored"] = py::cast(documents_exactly_scored);
        output["component_products"] = py::cast(component_products);
        output["full_component_products"] = py::cast(full_work_total);
        output["work_ratio"] = work_ratio;
        output["pruned_by_checkpoint"] = std::move(pruned_by_checkpoint);
        return output;
    }

    std::size_t dimension_ = 0;
    std::size_t document_count_ = 0;
    std::size_t total_document_tokens_ = 0;
    std::vector<std::size_t> document_offsets_;
    std::vector<std::size_t> checkpoints_;
    std::vector<double> summation_gammas_;
    std::vector<double> one_minus_summation_gammas_;
    std::vector<float> vertical_documents_;
    std::vector<double> document_residual_norms_;
};

}  // namespace

PYBIND11_MODULE(_bond_maxsim, module) {
    module.doc() = "Exact-safe checkpoint-pruned ColBERT MaxSim search.";
#if defined(_OPENMP)
    module.attr("openmp_enabled") = true;
#else
    module.attr("openmp_enabled") = false;
#endif

    py::class_<BondMaxSimIndex>(module, "BondMaxSimIndex")
        .def(
            py::init<const py::array &, const py::array &, const py::array &>(),
            py::arg("document_values").noconvert(),
            py::arg("document_offsets").noconvert(),
            py::arg("checkpoints").noconvert())
        .def(
            "search",
            &BondMaxSimIndex::search,
            py::arg("query_values").noconvert(),
            py::arg("query_offsets").noconvert(),
            py::arg("k"),
            py::arg("seed_count"),
            R"doc(
Return exact deterministic top-k MaxSim results and pruning diagnostics.

Seed documents are IDs [0, seed_count). The threshold is the kth-largest
conservative lower bound among documents that have been fully scored.
)doc")
        .def(
            "search_with_seed_ids",
            &BondMaxSimIndex::search_with_seed_ids,
            py::arg("query_values").noconvert(),
            py::arg("query_offsets").noconvert(),
            py::arg("k"),
            py::arg("seed_ids").noconvert(),
            R"doc(
Return exact top-k results using explicit fully scored seed IDs per query.

seed_ids must be a C-contiguous int64 matrix with one unique row per query and
at least k columns. Seed selection itself is outside this method's timer.
)doc")
        .def_property_readonly("document_count", &BondMaxSimIndex::document_count)
        .def_property_readonly("dimension", &BondMaxSimIndex::dimension)
        .def_property_readonly("checkpoints", &BondMaxSimIndex::checkpoints);
}
