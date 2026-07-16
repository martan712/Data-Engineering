from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils_colbert import (  # noqa: E402
    aggregate_token_retrieval_scores,
    compute_qrels_metrics,
    l2_normalize,
    load_packed_embeddings,
    maxsim_score,
    rank_documents,
    save_packed_embeddings,
    unpack_embeddings,
)
from benchmarking import (  # noqa: E402
    BenchmarkObservation,
    run_interleaved_trials,
    run_trials,
    sha256_file,
    summarize_samples,
)


class ColbertUtilityTests(unittest.TestCase):
    def test_l2_normalize_handles_nonzero_and_zero_rows(self) -> None:
        matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

        normalized = l2_normalize(matrix)

        np.testing.assert_allclose(normalized[0], [0.6, 0.8], atol=1e-6)
        np.testing.assert_array_equal(normalized[1], [0.0, 0.0])
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(normalized.flags.c_contiguous)

    def test_maxsim_matches_hand_computed_score(self) -> None:
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        document = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)

        self.assertAlmostEqual(maxsim_score(query, document), 1.5, places=6)

    def test_rank_documents_breaks_score_ties_by_document_index(self) -> None:
        scores = np.array([1.0, 2.0, 2.0], dtype=np.float32)

        self.assertEqual(rank_documents(scores), [1, 2, 0])

    def test_token_hits_aggregate_to_per_document_maxima(self) -> None:
        token_to_doc = np.array([0, 0, 1], dtype=np.int64)
        hits = [
            [(0, 0.9), (2, 0.4)],
            [(1, 0.8), (2, 0.7)],
        ]

        scores, candidates, retrieved_count = aggregate_token_retrieval_scores(
            hits,
            token_to_doc,
            num_documents=2,
        )

        np.testing.assert_allclose(scores, [1.7, 1.1], atol=1e-6)
        self.assertEqual(candidates, {0, 1})
        self.assertEqual(retrieved_count, 4)

    def test_packed_embeddings_round_trip_variable_lengths(self) -> None:
        arrays = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packed.npz"
            save_packed_embeddings(path, ["d1", "d2"], ["one", "two"], arrays)

            packed = load_packed_embeddings(path)
            unpacked = unpack_embeddings(packed["values"], packed["offsets"])

        self.assertEqual(packed["ids"], ["d1", "d2"])
        self.assertEqual(packed["texts"], ["one", "two"])
        self.assertEqual(packed["offsets"].tolist(), [0, 1, 3])
        np.testing.assert_array_equal(unpacked[0], arrays[0])
        np.testing.assert_array_equal(unpacked[1], arrays[1])

    def test_packed_embeddings_reject_inconsistent_metadata(self) -> None:
        arrays = [np.array([[1.0, 2.0]], dtype=np.float32)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packed.npz"
            with self.assertRaisesRegex(ValueError, "equal length"):
                save_packed_embeddings(path, ["d1"], [], arrays)

    def test_load_packed_embeddings_rejects_invalid_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez(
                path,
                ids=np.array(["d1"]),
                texts=np.array(["one"]),
                values=np.array([[1.0, 2.0]], dtype=np.float32),
                offsets=np.array([0, 2], dtype=np.int64),
                shapes=np.array([[1, 2]], dtype=np.int64),
            )

            with self.assertRaisesRegex(ValueError, "final offset"):
                load_packed_embeddings(path)

    def test_maxsim_rejects_incompatible_token_matrices(self) -> None:
        query = np.ones((1, 2), dtype=np.float32)
        wrong_dimension = np.ones((1, 3), dtype=np.float32)
        empty_document = np.empty((0, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "same dimension"):
            maxsim_score(query, wrong_dimension)
        with self.assertRaisesRegex(ValueError, "at least one"):
            maxsim_score(query, empty_document)

    def test_qrels_metrics_use_macro_recall_and_reciprocal_rank(self) -> None:
        rankings = {"q1": ["d2", "d1", "d4"]}
        qrels = {"q1": ["d1", "d3"]}

        metrics = compute_qrels_metrics(rankings, qrels, k_values=(1, 3), mrr_k=10)

        self.assertEqual(metrics["query_count"], 1)
        self.assertEqual(metrics["macro"]["recall@1"], 0.0)
        self.assertEqual(metrics["macro"]["recall@3"], 0.5)
        self.assertEqual(metrics["macro"]["mrr@10"], 0.5)


class BenchmarkingTests(unittest.TestCase):
    def test_summarize_samples_reports_median_and_percentiles(self) -> None:
        summary = summarize_samples([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 3.0)
        self.assertEqual(summary["p25"], 2.0)
        self.assertEqual(summary["p75"], 4.0)
        self.assertAlmostEqual(summary["p95"], 4.8)

    def test_run_trials_separates_outer_and_component_timings(self) -> None:
        calls = []

        def action() -> BenchmarkObservation:
            calls.append(len(calls))
            return BenchmarkObservation(value=len(calls), stage_seconds={"search": 0.001})

        value, timing = run_trials(action, warmup_runs=2, measured_runs=3)

        self.assertEqual(len(calls), 5)
        self.assertEqual(value, 5)
        self.assertEqual(timing["measured_runs"], 3)
        self.assertEqual(len(timing["end_to_end_seconds"]), 3)
        self.assertEqual(timing["stage_seconds"]["search"], [0.001, 0.001, 0.001])

    def test_interleaved_trials_rotate_method_order(self) -> None:
        calls = []

        def action(name: str):
            def run() -> BenchmarkObservation:
                calls.append(name)
                return BenchmarkObservation(value=name, stage_seconds={"work": 0.001})

            return run

        values, timing = run_interleaved_trials(
            {"exact": action("exact"), "ivf": action("ivf")},
            warmup_runs=1,
            measured_runs=3,
        )

        self.assertEqual(calls, ["exact", "ivf", "exact", "ivf", "ivf", "exact", "exact", "ivf"])
        self.assertEqual(values, {"exact": "exact", "ivf": "ivf"})
        self.assertEqual(
            timing["measured_execution_order"],
            [["exact", "ivf"], ["ivf", "exact"], ["exact", "ivf"]],
        )

    def test_sha256_file_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"colbert")

            digest = sha256_file(path)

        self.assertEqual(
            digest,
            "a0d15b71b21180599480720674c8a90b53730dd8441a23cfb3e22f600cbeb8a2",
        )


if __name__ == "__main__":
    unittest.main()
