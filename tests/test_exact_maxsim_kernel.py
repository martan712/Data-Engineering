from __future__ import annotations

import unittest

import numpy as np

from experiments.kernels import (
    exact_maxsim_scores,
    exact_maxsim_scores_f64,
    extension_available,
)


def pack(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(matrices) + 1, dtype=np.int64)
    for index, matrix in enumerate(matrices):
        offsets[index + 1] = offsets[index] + matrix.shape[0]
    values = np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)
    return values, offsets


@unittest.skipUnless(
    extension_available(),
    "exact MaxSim extension is not built",
)
class ExactMaxSimKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32),
            np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.8], [0.2, 0.2, 0.2]],
                dtype=np.float32,
            ),
            np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            np.array([[0.3, 0.3, 0.3]], dtype=np.float32),
        ]
        self.queries = [
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        ]
        self.document_values, self.document_offsets = pack(self.documents)
        self.query_values, self.query_offsets = pack(self.queries)

    def test_scores_and_rankings_match_numpy(self) -> None:
        expected = np.empty(
            (len(self.queries), len(self.documents)),
            dtype=np.float32,
        )
        for query_index, query in enumerate(self.queries):
            for document_index, document in enumerate(self.documents):
                similarities = query @ document.T
                expected[query_index, document_index] = np.max(
                    similarities,
                    axis=1,
                ).sum(dtype=np.float32)

        actual = exact_maxsim_scores(
            self.document_values,
            self.document_offsets,
            self.query_values,
            self.query_offsets,
        )

        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

        tie_breakers = np.arange(len(self.documents))
        expected_rankings = [
            np.lexsort((tie_breakers, -scores)).tolist() for scores in expected
        ]
        actual_rankings = [
            np.lexsort((tie_breakers, -scores)).tolist() for scores in actual
        ]
        self.assertEqual(actual_rankings, expected_rankings)

    def test_rejects_wrong_dtype_and_invalid_offsets(self) -> None:
        with self.assertRaisesRegex(TypeError, "document_values must have dtype float32"):
            exact_maxsim_scores(
                self.document_values.astype(np.float64),
                self.document_offsets,
                self.query_values,
                self.query_offsets,
            )

        invalid_offsets = self.document_offsets.copy()
        invalid_offsets[-1] += 1
        with self.assertRaisesRegex(ValueError, "out-of-range offset"):
            exact_maxsim_scores(
                self.document_values,
                invalid_offsets,
                self.query_values,
                self.query_offsets,
            )

    def test_float64_accumulation_matches_float64_numpy(self) -> None:
        expected = np.empty(
            (len(self.queries), len(self.documents)),
            dtype=np.float64,
        )
        for query_index, query in enumerate(self.queries):
            query_f64 = query.astype(np.float64)
            for document_index, document in enumerate(self.documents):
                similarities = query_f64 @ document.astype(np.float64).T
                expected[query_index, document_index] = np.max(
                    similarities,
                    axis=1,
                ).sum(dtype=np.float64)

        actual = exact_maxsim_scores_f64(
            self.document_values,
            self.document_offsets,
            self.query_values,
            self.query_offsets,
        )

        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
