from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from benchmarking import sha256_directory  # noqa: E402
from plaid_benchmark import (  # noqa: E402
    PlaidConfig,
    parse_plaid_config,
    select_validation_configs,
)


def arm(nprobe: int, full_scores: int, recall: float, median: float) -> dict:
    return {
        "config": {"n_ivf_probe": nprobe, "n_full_scores": full_scores},
        "exact_recovery": {"mean_recall@10": recall},
        "timing": {"end_to_end_summary": {"median": median}},
    }


class PlaidBenchmarkTests(unittest.TestCase):
    def test_parse_config_and_label(self) -> None:
        config = parse_plaid_config("16:200")
        self.assertEqual(config, PlaidConfig(16, 200))
        self.assertEqual(config.label, "plaid_np16_c200")
        with self.assertRaises(ValueError):
            parse_plaid_config("16")
        with self.assertRaises(ValueError):
            parse_plaid_config("0:20")

    def test_validation_selection_uses_fastest_eligible_and_best_quality(self) -> None:
        arms = {
            "plaid_np8_c50": arm(8, 50, 0.84, 0.20),
            "plaid_np16_c100": arm(16, 100, 0.91, 0.30),
            "plaid_np32_c100": arm(32, 100, 0.96, 0.45),
            "plaid_np64_c200": arm(64, 200, 0.98, 0.70),
        }
        result = select_validation_configs(arms, recall_targets=(0.85, 0.95))
        self.assertEqual(
            [item["name"] for item in result["selected"]],
            ["plaid_np16_c100", "plaid_np32_c100", "plaid_np64_c200"],
        )

    def test_directory_hash_covers_names_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "b.bin").write_bytes(b"beta")
            first = sha256_directory(root)
            self.assertEqual(first["files"], 2)
            self.assertEqual(first["bytes"], 9)
            (root / "nested" / "b.bin").write_bytes(b"gamma")
            second = sha256_directory(root)
            self.assertNotEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
