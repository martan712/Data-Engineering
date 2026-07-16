"""Guard the main report tables against stale copied values.

The benchmark JSON files are the source of truth.  The LaTeX tables are kept
readable and editable, so this test checks the copied, rounded values whenever
the normal test suite is run.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FINAL_RESULTS = ROOT / "results" / "final"
REPORT_SECTIONS = ROOT / "report" / "sections"


def load_result(name: str) -> dict:
    return json.loads((FINAL_RESULTS / name).read_text(encoding="utf-8"))


def compact_tex(name: str) -> str:
    """Remove layout-only whitespace so table rows are easy to compare."""
    text = (REPORT_SECTIONS / name).read_text(encoding="utf-8")
    return " ".join(text.split())


def format_decimal(value: float, places: int) -> str:
    """Format report values with conventional decimal half-up rounding."""
    quantum = Decimal(1).scaleb(-places)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{places}f}"


class ReportConsistencyTests(unittest.TestCase):
    def assert_table_row(self, section: str, cells: list[str]) -> None:
        row = " & ".join(cells) + r" \\"
        self.assertIn(row, compact_tex(section))

    def test_release_identity_is_reported(self) -> None:
        manifest = load_result("manifest.json")
        report = (ROOT / "report" / "sections" / "reproducibility.tex").read_text(
            encoding="utf-8"
        )

        self.assertEqual(7, len(manifest["files"]))
        self.assertIn(manifest["source_commit"][:7], report)

    def test_ivf_tables_match_release_artifacts(self) -> None:
        datasets = (
            ("SciFact", "scifact_ivf_test_40q_final.json"),
            ("NFCorpus", "nfcorpus_ivf_test_40q_final.json"),
        )

        for dataset_name, filename in datasets:
            result = load_result(filename)
            exact = result["exact_reference"]
            exact_timing = exact["timing"]["end_to_end_summary"]
            self.assert_table_row(
                "results_ivf.tex",
                [
                    "Compiled exact",
                    f"{exact_timing['median']:.4f}",
                    f"{exact_timing['p95']:.4f}",
                    "1.0000",
                    f"{result['dataset']['documents']:.1f}",
                    "1.0000",
                ],
            )

            for key, arm in result["approximate_arms"].items():
                engine = "FAISS-IVF" if key.startswith("faiss") else "PDX-IVF"
                budget = key.rsplit("_c", 1)[1]
                label = (
                    f"{engine}, full pool"
                    if budget == "full"
                    else f"{engine}, $C{{=}}{budget}$"
                )
                timing = arm["timing"]["end_to_end_summary"]
                self.assert_table_row(
                    "results_ivf.tex",
                    [
                        label,
                        f"{timing['median']:.4f}",
                        f"{timing['p95']:.4f}",
                        f"{arm['exact_recovery']['mean_recall@10']:.4f}",
                        format_decimal(arm["work"]["mean_selected_candidates"], 1),
                        f"{arm['work']['reranked_comparison_ratio']:.4f}",
                    ],
                )

            with self.subTest(dataset=dataset_name):
                self.assertEqual(40, result["dataset"]["queries"])

    def test_plaid_tables_match_release_artifacts(self) -> None:
        selected_files = (
            ("SciFact", "scifact_ivf_test_40q_final.json"),
            ("NFCorpus", "nfcorpus_ivf_test_40q_final.json"),
        )
        for dataset_name, filename in selected_files:
            result = load_result(filename)
            for arm in result["plaid_arms"].values():
                timing = arm["timing"]["end_to_end_summary"]
                full_scores = arm["config"]["n_full_scores"]
                self.assert_table_row(
                    "results_plaid.tex",
                    [
                        f"{dataset_name}, $F{{=}}{full_scores}$",
                        f"{timing['median']:.4f}",
                        f"{timing['p95']:.4f}",
                        f"{arm['exact_recovery']['mean_recall@10']:.4f}",
                        f"{arm['qrels_metrics']['recall@10']:.4f}",
                        f"{arm['configured_work']['upper_bound_full_score_ratio']:.4f}",
                    ],
                )

        sensitivity_files = (
            ("SciFact", "scifact_plaid_fullscore_sensitivity_40q_final.json"),
            ("NFCorpus", "nfcorpus_plaid_fullscore_sensitivity_40q_final.json"),
        )
        for dataset_name, filename in sensitivity_files:
            result = load_result(filename)
            arm = next(iter(result["plaid_arms"].values()))
            exact_median = result["exact_reference"]["timing"][
                "end_to_end_summary"
            ]["median"]
            plaid_median = arm["timing"]["end_to_end_summary"]["median"]
            self.assert_table_row(
                "results_plaid.tex",
                [
                    dataset_name,
                    str(arm["config"]["n_full_scores"]),
                    f"{exact_median:.4f}",
                    f"{plaid_median:.4f}",
                    f"{plaid_median / exact_median:.2f}",
                    f"{arm['exact_recovery']['mean_recall@10']:.4f}",
                    f"{arm['qrels_metrics']['recall@10']:.4f}",
                ],
            )

    def test_bond_table_matches_release_artifacts(self) -> None:
        rows = (
            (
                "SciFact raw, prefix-500",
                "scifact_bond_raw_test_40q_final.json",
                "bond_seed500",
            ),
            (
                "SciFact raw, free oracle",
                "scifact_bond_raw_test_40q_final.json",
                "bond_oracle_topk_seeds",
            ),
            (
                "SciFact PCA, prefix-500",
                "scifact_bond_pca_test_40q_final.json",
                "bond_seed500",
            ),
            (
                "SciFact PCA, free oracle",
                "scifact_bond_pca_test_40q_final.json",
                "bond_oracle_topk_seeds",
            ),
            (
                "NFCorpus raw, prefix-500",
                "nfcorpus_bond_raw_test_40q_final.json",
                "bond_seed500",
            ),
        )

        for label, filename, arm_name in rows:
            result = load_result(filename)
            arm = result["bond_arms"][arm_name]
            exact_median = result["exact_reference"]["timing"][
                "end_to_end_summary"
            ]["median"]
            bond_median = arm["timing"]["end_to_end_summary"]["median"]
            self.assert_table_row(
                "results_bond.tex",
                [
                    label,
                    f"{bond_median:.4f}",
                    f"{exact_median:.4f}",
                    f"{bond_median / exact_median:.2f}",
                    f"{100 * arm['work']['pruned_document_fraction']:.2f}",
                    f"{100 * arm['work']['component_product_ratio']:.2f}",
                ],
            )

    def test_appendix_dataset_counts_match_release_artifacts(self) -> None:
        datasets = (
            ("SciFact", "scifact_ivf_test_40q_final.json"),
            ("NFCorpus", "nfcorpus_ivf_test_40q_final.json"),
        )
        for dataset_name, filename in datasets:
            dataset = load_result(filename)["dataset"]
            self.assert_table_row(
                "appendix.tex",
                [
                    dataset_name,
                    str(dataset["documents"]),
                    str(dataset["document_token_vectors"]),
                    str(dataset["queries"]),
                    str(dataset["query_token_vectors"]),
                ],
            )


if __name__ == "__main__":
    unittest.main()
