from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "experiments" / "pipeline" / "10_controlled_bond_pilot.py"
SPEC = importlib.util.spec_from_file_location("controlled_bond_pilot", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BondRunnerNumericalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.array([[0, 1, 2]], dtype=np.int64)
        self.scores = np.array([[1.0, 0.5, 0.5, 0.1]], dtype=np.float32)

    def test_accepts_only_set_identical_float_tie_reordering(self) -> None:
        candidate = np.array([[0, 2, 1]], dtype=np.int64)

        result = MODULE.numerical_order_validation(
            self.reference,
            self.scores,
            candidate,
        )

        self.assertFalse(result["rankings_identical"])
        self.assertTrue(result["topk_sets_identical"])
        self.assertTrue(result["tie_equivalent_order"])

    def test_rejects_non_tie_inversion(self) -> None:
        candidate = np.array([[1, 0, 2]], dtype=np.int64)

        result = MODULE.numerical_order_validation(
            self.reference,
            self.scores,
            candidate,
        )

        self.assertFalse(result["tie_equivalent_order"])
        self.assertEqual(len(result["non_tie_inversions"]), 1)

    def test_rejects_topk_set_loss(self) -> None:
        candidate = np.array([[0, 1, 3]], dtype=np.int64)

        result = MODULE.numerical_order_validation(
            self.reference,
            self.scores,
            candidate,
        )

        self.assertFalse(result["topk_sets_identical"])
        self.assertFalse(result["tie_equivalent_order"])


if __name__ == "__main__":
    unittest.main()
