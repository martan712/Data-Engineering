from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from benchmarking import sha256_file  # noqa: E402


class LegacyResultManifestTests(unittest.TestCase):
    def assert_manifest_matches_directory(self, result_dir: Path) -> dict:
        manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
        declared_names = {item["name"] for item in manifest["files"]}
        actual_names = {path.name for path in result_dir.glob("*.json")}
        actual_names.remove("manifest.json")

        self.assertFalse(manifest["wall_clock_claims_allowed"])
        self.assertEqual(declared_names, actual_names)

        for item in manifest["files"]:
            path = result_dir / item["name"]
            self.assertEqual(path.stat().st_size, item["bytes"], item["name"])
            self.assertEqual(sha256_file(path), item["sha256"], item["name"])
        return manifest

    def test_manifest_matches_every_preserved_legacy_result(self) -> None:
        manifest = self.assert_manifest_matches_directory(PROJECT_ROOT / "results" / "legacy")
        self.assertEqual(manifest["status"], "legacy_exploratory")

    def test_manifest_matches_every_controlled_pilot_result(self) -> None:
        manifest = self.assert_manifest_matches_directory(PROJECT_ROOT / "results" / "controlled")
        self.assertEqual(manifest["status"], "controlled_pilots_pending_clean_rerun")


if __name__ == "__main__":
    unittest.main()
