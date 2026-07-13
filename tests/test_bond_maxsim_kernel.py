from __future__ import annotations

import unittest

import numpy as np

from experiments.kernels import BondMaxSimIndex, bond_maxsim_extension_available


def pack(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(matrices) + 1, dtype=np.int64)
    for index, matrix in enumerate(matrices):
        offsets[index + 1] = offsets[index] + matrix.shape[0]
    values = np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)
    return values, offsets


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix64 = np.asarray(matrix, dtype=np.float64)
    matrix64 /= np.linalg.norm(matrix64, axis=1, keepdims=True)
    return np.ascontiguousarray(matrix64, dtype=np.float32)


def oracle(
    queries: list[np.ndarray],
    documents: list[np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_scores = np.empty((len(queries), len(documents)), dtype=np.float64)
    ids = np.empty((len(queries), k), dtype=np.int64)
    top_scores = np.empty((len(queries), k), dtype=np.float64)
    tie_breakers = np.arange(len(documents), dtype=np.int64)
    for query_index, query in enumerate(queries):
        query64 = query.astype(np.float64)
        for document_index, document in enumerate(documents):
            dots = query64 @ document.astype(np.float64).T
            all_scores[query_index, document_index] = np.max(dots, axis=1).sum(
                dtype=np.float64
            )
        order = np.lexsort((tie_breakers, -all_scores[query_index]))[:k]
        ids[query_index] = order
        top_scores[query_index] = all_scores[query_index, order]
    return ids, top_scores, all_scores


@unittest.skipUnless(
    bond_maxsim_extension_available(),
    "BOND-MaxSim extension is unavailable (expected on unbuilt Windows checkouts)",
)
class BondMaxSimKernelTests(unittest.TestCase):
    def assert_matches_oracle(
        self,
        documents: list[np.ndarray],
        queries: list[np.ndarray],
        checkpoints: list[int],
        k: int,
        seed_count: int,
    ) -> dict[str, object]:
        document_values, document_offsets = pack(documents)
        query_values, query_offsets = pack(queries)
        index = BondMaxSimIndex(
            document_values,
            document_offsets,
            np.asarray(checkpoints, dtype=np.int64),
        )
        result = index.search(query_values, query_offsets, k, seed_count)
        expected_ids, expected_scores, _ = oracle(queries, documents, k)
        np.testing.assert_array_equal(result["ids"], expected_ids)
        np.testing.assert_allclose(
            result["scores"], expected_scores, rtol=2e-13, atol=2e-13
        )
        self.assertEqual(result["ids"].dtype, np.int64)
        self.assertEqual(result["scores"].dtype, np.float64)
        self.assertEqual(
            result["documents_pruned"] + result["documents_exactly_scored"],
            len(queries) * len(documents),
        )
        self.assertEqual(
            int(result["pruned_by_checkpoint"].sum()), result["documents_pruned"]
        )
        self.assertLessEqual(result["component_products"], result["full_component_products"])
        return result

    def test_randomized_normalized_variable_lengths_match_float64_oracle(self) -> None:
        rng = np.random.default_rng(20260713)
        dimension = 12
        documents = [
            normalize_rows(rng.normal(size=(length, dimension)))
            for length in [1, 3, 2, 5, 4, 2, 6, 1, 3, 5, 2, 4, 1, 6, 3, 2, 5]
        ]
        queries = [
            normalize_rows(rng.normal(size=(length, dimension)))
            for length in [1, 4, 2, 3, 5]
        ]
        result = self.assert_matches_oracle(documents, queries, [3, 7, 12], 5, 6)
        expected_full = (
            sum(query.shape[0] for query in queries)
            * sum(document.shape[0] for document in documents)
            * dimension
        )
        self.assertEqual(result["full_component_products"], expected_full)
        self.assertAlmostEqual(
            result["work_ratio"],
            result["component_products"] / expected_full,
        )

    def test_all_negative_similarities(self) -> None:
        documents = [
            np.array([[-0.8, -0.7, 0.1, 0.0]], dtype=np.float32),
            np.array([[-0.2, -0.3, 0.0, 0.1]], dtype=np.float32),
            np.array([[-0.4, -0.1, 0.2, 0.0]], dtype=np.float32),
            np.array([[-0.6, -0.5, 0.0, 0.2]], dtype=np.float32),
        ]
        queries = [
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        ]
        result = self.assert_matches_oracle(documents, queries, [2, 4], 2, 2)
        self.assertTrue(np.all(result["scores"] < 0.0))

    def test_equal_scores_at_k_boundary_are_not_pruned(self) -> None:
        documents = [
            np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float32) for _ in range(5)
        ]
        queries = [np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)]
        result = self.assert_matches_oracle(documents, queries, [1, 4], 2, 2)
        np.testing.assert_array_equal(result["ids"], np.array([[0, 1]], dtype=np.int64))
        self.assertEqual(result["documents_pruned"], 0)
        self.assertEqual(result["documents_exactly_scored"], 5)

    def test_near_threshold_winner_is_not_pruned(self) -> None:
        below_one = np.nextafter(
            np.float32(1.0),
            np.float32(-np.inf),
            dtype=np.float32,
        )
        documents = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[below_one, 1.0]], dtype=np.float32),
        ]
        queries = [np.array([[1.0, 1.0e-7]], dtype=np.float32)]

        result = self.assert_matches_oracle(documents, queries, [1, 2], 1, 1)

        # The second document is below the seed after the first dimension, but
        # its tiny residual contribution makes it the true winner.
        np.testing.assert_array_equal(result["ids"], np.array([[1]], dtype=np.int64))
        self.assertEqual(result["documents_exactly_scored"], 2)

    def test_deliberately_prunable_data_reduces_component_work(self) -> None:
        dimension = 8
        documents = [
            np.array([[1.0] + [0.0] * (dimension - 1)], dtype=np.float32),
            np.array([[1.0] + [0.0] * (dimension - 1)], dtype=np.float32),
        ] + [
            np.array([[-1.0] + [0.0] * (dimension - 1)], dtype=np.float32)
            for _ in range(10)
        ]
        queries = [
            np.array([[1.0] + [0.0] * (dimension - 1)], dtype=np.float32)
        ]
        result = self.assert_matches_oracle(documents, queries, [2, 4, 8], 2, 2)
        self.assertGreater(result["documents_pruned"], 0)
        self.assertLess(result["work_ratio"], 1.0)
        self.assertGreater(result["pruned_by_checkpoint"][0], 0)

    def test_explicit_nonprefix_seeds_strengthen_the_safe_threshold(self) -> None:
        dimension = 8
        documents = [
            np.array([[-1.0] + [0.0] * (dimension - 1)], dtype=np.float32)
            for _ in range(10)
        ] + [
            np.array([[1.0] + [0.0] * (dimension - 1)], dtype=np.float32),
            np.array([[0.9] + [0.0] * (dimension - 1)], dtype=np.float32),
        ]
        queries = [np.array([[1.0] + [0.0] * (dimension - 1)], dtype=np.float32)]
        document_values, document_offsets = pack(documents)
        query_values, query_offsets = pack(queries)
        index = BondMaxSimIndex(
            document_values,
            document_offsets,
            np.array([2, 4, 8], dtype=np.int64),
        )

        prefix = index.search(query_values, query_offsets, 2, 2)
        explicit = index.search_with_seed_ids(
            query_values,
            query_offsets,
            2,
            np.array([[10, 11]], dtype=np.int64),
        )
        expected_ids, expected_scores, _ = oracle(queries, documents, 2)

        np.testing.assert_array_equal(explicit["ids"], expected_ids)
        np.testing.assert_allclose(explicit["scores"], expected_scores, rtol=2e-13, atol=2e-13)
        self.assertGreater(explicit["documents_pruned"], prefix["documents_pruned"])
        self.assertLess(explicit["work_ratio"], prefix["work_ratio"])

    def test_constructor_owns_document_data(self) -> None:
        documents = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[-1.0, 0.0]], dtype=np.float32),
        ]
        values, offsets = pack(documents)
        index = BondMaxSimIndex(values, offsets, np.array([1, 2], dtype=np.int64))
        values[:] = 0.0
        query_values, query_offsets = pack(
            [np.array([[1.0, 0.0]], dtype=np.float32)]
        )
        result = index.search(query_values, query_offsets, 1, 1)
        np.testing.assert_array_equal(result["ids"], np.array([[0]], dtype=np.int64))
        np.testing.assert_allclose(result["scores"], np.array([[1.0]]))
        self.assertEqual(index.document_count, 2)
        self.assertEqual(index.dimension, 2)
        self.assertEqual(index.checkpoints, (1, 2))

    def test_rejects_invalid_constructor_inputs(self) -> None:
        values = np.ascontiguousarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        offsets = np.array([0, 1, 2], dtype=np.int64)
        checkpoints = np.array([1, 3], dtype=np.int64)

        with self.assertRaises(TypeError):
            BondMaxSimIndex(values.astype(np.float64), offsets, checkpoints)
        with self.assertRaises(ValueError):
            BondMaxSimIndex(values[:, ::-1], offsets, checkpoints)
        nonfinite = values.copy()
        nonfinite[0, 0] = np.inf
        with self.assertRaises(ValueError):
            BondMaxSimIndex(nonfinite, offsets, checkpoints)
        with self.assertRaises(TypeError):
            BondMaxSimIndex(values, offsets.astype(np.int32), checkpoints)
        for invalid_offsets in (
            np.array([1, 2, 2], dtype=np.int64),
            np.array([0, 2, 1], dtype=np.int64),
            np.array([0, 1, 3], dtype=np.int64),
            np.array([0, 0, 2], dtype=np.int64),
        ):
            with self.subTest(offsets=invalid_offsets.tolist()), self.assertRaises(ValueError):
                BondMaxSimIndex(values, invalid_offsets, checkpoints)
        with self.assertRaises(TypeError):
            BondMaxSimIndex(values, offsets, checkpoints.astype(np.int32))
        for invalid_checkpoints in (
            np.array([], dtype=np.int64),
            np.array([0, 3], dtype=np.int64),
            np.array([1, 1, 3], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([1, 4], dtype=np.int64),
        ):
            with self.subTest(checkpoints=invalid_checkpoints.tolist()), self.assertRaises(ValueError):
                BondMaxSimIndex(values, offsets, invalid_checkpoints)

    def test_rejects_invalid_search_inputs_and_parameters(self) -> None:
        document_values = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        document_offsets = np.array([0, 1, 2], dtype=np.int64)
        index = BondMaxSimIndex(
            document_values,
            document_offsets,
            np.array([1, 2], dtype=np.int64),
        )
        query_values = np.array([[1.0, 0.0]], dtype=np.float32)
        query_offsets = np.array([0, 1], dtype=np.int64)

        with self.assertRaises(TypeError):
            index.search(query_values.astype(np.float64), query_offsets, 1, 1)
        with self.assertRaises(ValueError):
            index.search(query_values[:, ::-1], query_offsets, 1, 1)
        nonfinite = query_values.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            index.search(nonfinite, query_offsets, 1, 1)
        with self.assertRaises(ValueError):
            index.search(np.ones((1, 3), dtype=np.float32), query_offsets, 1, 1)
        with self.assertRaises(TypeError):
            index.search(query_values, query_offsets.astype(np.int32), 1, 1)
        with self.assertRaises(ValueError):
            index.search(query_values, np.array([0, 2], dtype=np.int64), 1, 1)
        for k, seed_count, exception in (
            (0, 1, ValueError),
            (3, 3, ValueError),
            (2, 1, ValueError),
            (1, 3, ValueError),
            (True, 1, TypeError),
            (1.0, 1, TypeError),
            (1, True, TypeError),
        ):
            with self.subTest(k=k, seed_count=seed_count), self.assertRaises(exception):
                index.search(query_values, query_offsets, k, seed_count)

        valid_seed_ids = np.array([[0]], dtype=np.int64)
        result = index.search_with_seed_ids(
            query_values,
            query_offsets,
            1,
            valid_seed_ids,
        )
        np.testing.assert_array_equal(result["ids"], np.array([[0]], dtype=np.int64))
        invalid_seed_cases = (
            valid_seed_ids.astype(np.int32),
            np.array([0], dtype=np.int64),
            np.array([[0], [1]], dtype=np.int64),
            np.array([[0, 0]], dtype=np.int64),
            np.array([[-1]], dtype=np.int64),
            np.array([[2]], dtype=np.int64),
        )
        for invalid_seed_ids in invalid_seed_cases:
            with self.subTest(seed_ids=invalid_seed_ids.tolist()), self.assertRaises(
                (TypeError, ValueError)
            ):
                index.search_with_seed_ids(
                    query_values,
                    query_offsets,
                    1,
                    invalid_seed_ids,
                )
        noncontiguous_seed_ids = np.array([[0, 1]], dtype=np.int64)[:, ::-1]
        with self.assertRaises(ValueError):
            index.search_with_seed_ids(
                query_values,
                query_offsets,
                1,
                noncontiguous_seed_ids,
            )
        with self.assertRaises(ValueError):
            index.search_with_seed_ids(
                query_values,
                query_offsets,
                2,
                valid_seed_ids,
            )


if __name__ == "__main__":
    unittest.main()
