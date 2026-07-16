from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from benchmarking import sha256_file  # noqa: E402


class LegacyResultManifestTests(unittest.TestCase):
    def assert_manifest_matches_directory(
        self,
        result_dir: Path,
        *,
        wall_clock_claims_allowed: bool,
    ) -> dict:
        manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
        declared_names = {item["name"] for item in manifest["files"]}
        actual_names = {path.name for path in result_dir.glob("*.json")}
        actual_names.remove("manifest.json")

        self.assertEqual(
            manifest["wall_clock_claims_allowed"],
            wall_clock_claims_allowed,
        )
        self.assertEqual(declared_names, actual_names)

        for item in manifest["files"]:
            path = result_dir / item["name"]
            self.assertEqual(path.stat().st_size, item["bytes"], item["name"])
            self.assertEqual(sha256_file(path), item["sha256"], item["name"])
        return manifest

    def test_manifest_matches_every_preserved_legacy_result(self) -> None:
        manifest = self.assert_manifest_matches_directory(
            PROJECT_ROOT / "results" / "legacy",
            wall_clock_claims_allowed=False,
        )
        self.assertEqual(manifest["status"], "legacy_exploratory")

    def test_manifest_matches_every_controlled_pilot_result(self) -> None:
        manifest = self.assert_manifest_matches_directory(
            PROJECT_ROOT / "results" / "controlled",
            wall_clock_claims_allowed=False,
        )
        self.assertEqual(manifest["status"], "controlled_pilots_pending_clean_rerun")

    def test_validation_manifest_freezes_plaid_selection(self) -> None:
        result_dir = PROJECT_ROOT / "results" / "validation"
        manifest = self.assert_manifest_matches_directory(
            result_dir,
            wall_clock_claims_allowed=False,
        )
        self.assertEqual(manifest["status"], "validation_only_parameter_selection")
        selection = json.loads(
            (result_dir / "plaid_release_selection.json").read_text(encoding="utf-8")
        )
        self.assertFalse(selection["validation_result"]["git"]["dirty"])
        self.assertEqual(
            [
                (item["config"]["n_ivf_probe"], item["config"]["n_full_scores"])
                for item in selection["selected"]
            ],
            [(8, 100), (8, 400)],
        )

    def test_final_manifest_matches_clean_result_metadata(self) -> None:
        result_dir = PROJECT_ROOT / "results" / "final"
        manifest = self.assert_manifest_matches_directory(
            result_dir,
            wall_clock_claims_allowed=True,
        )
        self.assertEqual(manifest["status"], "controlled_clean_release")

        for item in manifest["files"]:
            result = json.loads(
                (result_dir / item["name"]).read_text(encoding="utf-8")
            )
            metadata = result["metadata"]
            self.assertFalse(metadata["git"]["dirty"], item["name"])
            self.assertEqual(
                metadata["git"]["commit"],
                manifest["source_commit"],
                item["name"],
            )
            self.assertEqual(
                metadata["runtime"]["initial_cpu_affinity"],
                manifest["cpu_affinity"],
                item["name"],
            )
            self.assertGreaterEqual(result["config"]["measured_runs"], 5)

    def test_final_results_satisfy_release_correctness_contracts(self) -> None:
        result_dir = PROJECT_ROOT / "results" / "final"
        bond_names = (
            "scifact_bond_raw_test_40q_final.json",
            "scifact_bond_pca_test_40q_final.json",
            "nfcorpus_bond_raw_test_40q_final.json",
        )
        for name in bond_names:
            result = json.loads((result_dir / name).read_text(encoding="utf-8"))
            self.assertEqual(
                result["config"]["exact_accumulation"],
                "float64 products and accumulation",
            )
            for arm in result["bond_arms"].values():
                self.assertTrue(arm["exact_recovery"]["rankings_identical"], name)
                self.assertTrue(arm["exact_recovery"]["topk_sets_identical"], name)
                self.assertLess(arm["score_validation"]["max_abs_error"], 1e-12)

        ivf_names = (
            "scifact_ivf_test_40q_final.json",
            "nfcorpus_ivf_test_40q_final.json",
        )
        for name in ivf_names:
            result = json.loads((result_dir / name).read_text(encoding="utf-8"))
            for agreement in result["cross_engine_agreement"].values():
                self.assertTrue(agreement["candidate_pools_equal"], name)
                self.assertTrue(agreement["topk_rankings_equal"], name)
                self.assertTrue(agreement["retrieved_hit_counts_equal"], name)

            for budget in result["config"]["c_values"]:
                label = "full" if budget == 0 else str(budget)
                faiss = result["approximate_arms"][f"faiss_ivf_c{label}"]
                pdx = result["approximate_arms"][f"pdx_ivf_c{label}"]
                self.assertEqual(
                    faiss["exact_recovery"]["mean_recall@10"],
                    pdx["exact_recovery"]["mean_recall@10"],
                    name,
                )

        scifact = json.loads(
            (result_dir / "scifact_ivf_test_40q_final.json").read_text(encoding="utf-8")
        )
        nfcorpus = json.loads(
            (result_dir / "nfcorpus_ivf_test_40q_final.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            scifact["approximate_arms"]["pdx_ivf_cfull"]["exact_recovery"]
            ["mean_recall@10"],
            1.0,
        )
        self.assertEqual(
            nfcorpus["approximate_arms"]["pdx_ivf_cfull"]["exact_recovery"]
            ["mean_recall@10"],
            0.975,
        )


if __name__ == "__main__":
    unittest.main()
