from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.results.catalog import validate_catalog, validate_evidence_manifest


REQUIRED_FIELDS = {
    "evidence_id",
    "kind",
    "paper_label",
    "claim_marker",
    "status",
    "references",
    "filters",
    "grouping",
    "derivation",
    "units",
    "rounding",
    "renderer_command",
    "output_path",
    "qualifications",
}


def _resolve_pointer(document: Any, pointer: str) -> Any:
    current = document
    for encoded in pointer[1:].split("/") if pointer else ():
        token = encoded.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def test_manifest_covers_every_labeled_paper_table_and_figure() -> None:
    catalog = validate_catalog(REPO_ROOT / "artifacts/catalog.yaml", workspace=REPO_ROOT)
    manifest = validate_evidence_manifest(REPO_ROOT / "artifacts/paper_evidence.yaml", catalog)
    source = (REPO_ROOT / "report/main.tex").read_text(encoding="utf-8")
    paper_assets = set(re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", source))
    registered = {
        entry["paper_label"]
        for entry in manifest["evidence"]
        if entry["kind"] in {"table", "figure"}
    }
    assert registered == paper_assets


def test_evidence_entries_are_complete_and_select_real_json_leaves() -> None:
    catalog = validate_catalog(REPO_ROOT / "artifacts/catalog.yaml", workspace=REPO_ROOT)
    manifest = validate_evidence_manifest(REPO_ROOT / "artifacts/paper_evidence.yaml", catalog)
    artifacts = {entry["artifact_id"]: entry for entry in catalog["artifacts"]}
    assert len(manifest["evidence"]) == 19
    assert sum(entry["kind"] == "numeric_claim" for entry in manifest["evidence"]) == 11
    for entry in manifest["evidence"]:
        assert REQUIRED_FIELDS <= entry.keys()
        assert entry["renderer_command"] == "uv run python -m bondmaxsim.render.paper"
        assert entry["output_path"].startswith("report/generated/")
        assert entry["qualifications"]
        for reference in entry["references"]:
            artifact = artifacts[reference["artifact_id"]]
            document = json.loads((REPO_ROOT / artifact["path"]).read_text(encoding="utf-8"))
            _resolve_pointer(document, reference["pointer"])


def test_superseded_e08_timing_is_not_headline_evidence() -> None:
    manifest = json.loads((REPO_ROOT / "artifacts/paper_evidence.yaml").read_text(encoding="utf-8"))
    e08 = next(entry for entry in manifest["evidence"] if entry["evidence_id"] == "paper.figure.checkpoint-ablation")
    assert e08["status"] == "diagnostic"
    assert any(reference["artifact_id"].endswith("r12c_interleaved_exact_safe_1t") for reference in e08["references"])
    for entry in manifest["evidence"]:
        if entry["status"] != "diagnostic":
            assert not any(
                reference["artifact_id"].startswith("stage3_mechanism_e08_")
                and "ms_per_query" in reference["pointer"]
                for reference in entry["references"]
            )
