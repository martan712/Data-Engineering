"""Integrity checks for the accepted finalization contracts."""

from pathlib import Path


CONTRACT_DIR = Path(__file__).parents[1] / "docs" / "contracts"
CONTRACTS = {
    "agreement.md": (
        "strict_top_k_set_equal",
        "validate_boundary_tie_equivalence",
        "fp32_tie_max_ulps",
    ),
    "timing.md": ("paired_margin_pct", "best_observed", "confirmation"),
    "candidate-work.md": (
        "configured_candidate_cap",
        "documents_fully_scored",
        "system_cap",
    ),
    "query-workloads.md": (
        "beir-<dataset>-mechanism-seed42-n50-v1",
        "beir-<dataset>-qrels-test-v1",
    ),
    "results-and-artifacts.md": (
        "bondmaxsim.result-envelope",
        "RFC 6901",
        "validation_audit",
    ),
    "data-and-reproducibility.md": (
        "CPython 3.12.13",
        "paper-native",
        "row[\"text\"]",
    ),
    "terminology.md": (
        "BOND-style document-level early termination over embedding dimensions",
        "CoRECT-backed metric cross-validation",
        "does not compute",
    ),
}


def test_all_seven_contracts_are_accepted_and_define_public_terms():
    assert set(path.name for path in CONTRACT_DIR.glob("*.md")) == {
        "README.md",
        *CONTRACTS,
    }
    for filename, required_terms in CONTRACTS.items():
        source = (CONTRACT_DIR / filename).read_text(encoding="utf-8")
        assert "Status: Accepted" in source
        assert "Version: 1" in source
        for term in required_terms:
            assert term in source, f"{filename} does not define {term!r}"


def test_contract_index_links_every_record():
    index = (CONTRACT_DIR / "README.md").read_text(encoding="utf-8")
    for filename in CONTRACTS:
        assert f"]({filename})" in index
