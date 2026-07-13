from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from candidate_pipeline import (  # noqa: E402
    actual_rerank_count,
    build_candidate_features,
    select_candidates,
)


class CandidatePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.token_to_doc = np.array([0, 0, 1, 2, 2], dtype=np.int64)
        self.hits = [
            [(0, 0.9), (2, 0.7), (3, 0.6)],
            [(1, 0.8), (4, 0.5), (2, 0.4)],
        ]
        self.features = build_candidate_features(self.hits, self.token_to_doc, 3)

    def test_aggregation_records_pool_counts_and_best_similarity(self) -> None:
        self.assertEqual(self.features.candidate_doc_indices.tolist(), [0, 1, 2])
        np.testing.assert_allclose(self.features.approx_scores, [1.7, 1.1, 1.1])
        np.testing.assert_array_equal(self.features.retrieved_token_count, [2, 2, 2])
        np.testing.assert_array_equal(self.features.matched_query_token_count, [2, 2, 2])

    def test_approx_selection_is_deterministic_for_ties(self) -> None:
        self.assertEqual(select_candidates(self.features, 2, "approx_score"), [0, 1])

    def test_zero_budget_means_rerank_the_full_pool(self) -> None:
        self.assertEqual(select_candidates(self.features, 0), [0, 1, 2])

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_candidates(self.features, 2, "unknown")

    def test_actual_rerank_count_uses_realized_candidate_counts(self) -> None:
        self.assertEqual(actual_rerank_count([[1, 2], [3], []]), 3)


if __name__ == "__main__":
    unittest.main()

