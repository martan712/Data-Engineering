from __future__ import annotations

import json
from pathlib import Path

import pytest

from bondmaxsim.results import load_result, validate_frozen_inputs
from bondmaxsim.results.io import atomic_write_envelope
from bondmaxsim.results.models import (
    ExperimentResultEnvelope,
    ResultValidationError,
    default_provenance,
)


def _digest(character: str) -> str:
    return character * 64


def _hashes(*, index: str | None = None) -> dict[str, str | None]:
    return {
        "configuration_sha256": _digest("a"),
        "data_sha256": _digest("b"),
        "index_sha256": index,
    }


def _envelope(*, index: str | None = None) -> ExperimentResultEnvelope:
    provenance = default_provenance(command="pytest frozen hashes")
    provenance.update(_hashes(index=index))
    return ExperimentResultEnvelope.create(
        artifact_id="fixture-frozen-result",
        experiment_id="fixture-frozen-experiment",
        dataset_id="fixture",
        workload_id="fixture-frozen-workload-v1",
        method_configuration={"arm": "fixture"},
        protocol={"class": "fixture"},
        environment={"snapshot_id": "fixture"},
        provenance=provenance,
        payload_kind="validation_audit",
        payload={"checks": [{"status": "pass"}], "passed": True},
        created_at_utc="2026-07-15T12:00:00Z",
    )


def test_loader_accepts_exact_frozen_hashes(tmp_path: Path):
    output = tmp_path / "result.json"
    envelope = _envelope(index=_digest("c"))
    atomic_write_envelope(output, envelope)

    assert load_result(output, frozen_hashes=_hashes(index=_digest("c"))) == envelope


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("configuration_sha256", _digest("d")),
        ("data_sha256", _digest("d")),
        ("index_sha256", _digest("d")),
    ],
)
def test_loader_refuses_each_mismatched_frozen_hash(
    tmp_path: Path, field: str, replacement: str
):
    output = tmp_path / "result.json"
    atomic_write_envelope(output, _envelope(index=_digest("c")))
    expected = _hashes(index=_digest("c"))
    expected[field] = replacement

    with pytest.raises(ResultValidationError, match=field):
        load_result(output, frozen_hashes=expected)


def test_indexless_result_requires_explicit_indexless_freeze():
    envelope = _envelope()
    assert validate_frozen_inputs(envelope, _hashes()) == envelope

    with pytest.raises(ResultValidationError, match="index_sha256"):
        validate_frozen_inputs(envelope, _hashes(index=_digest("c")))


@pytest.mark.parametrize(
    "expected,match",
    [
        (
            {"configuration_sha256": _digest("a"), "data_sha256": _digest("b")},
            "missing",
        ),
        (
            {**_hashes(), "other_sha256": _digest("c")},
            "unexpected",
        ),
        (
            {**_hashes(), "data_sha256": "not-a-digest"},
            "lowercase SHA-256",
        ),
        (
            {**_hashes(), "configuration_sha256": None},
            "lowercase SHA-256",
        ),
    ],
)
def test_frozen_hash_contract_is_complete_and_strict(expected, match):
    with pytest.raises(ResultValidationError, match=match):
        validate_frozen_inputs(_envelope(), expected)


def test_historical_result_remains_readable_but_is_ineligible_for_freeze(
    tmp_path: Path,
):
    historical = tmp_path / "historical.json"
    historical.write_text(
        json.dumps(
            {
                "dataset": "scifact",
                "method": "dense",
                "num_docs": 2,
                "num_queries": 1,
                "recall_vs_exact_at_10": 1.0,
            }
        ),
        encoding="utf-8",
    )

    assert load_result(historical).payload["historical_migration"] is True
    with pytest.raises(ResultValidationError, match="historical result"):
        load_result(historical, frozen_hashes=_hashes())
