from __future__ import annotations

import copy

import pytest

from bondmaxsim.data.config import (
    DATASETS,
    DataConfigurationError,
    load_data_configuration,
    validate_data_configuration,
)


def test_final_data_configuration_is_frozen_and_fully_pinned():
    frozen = load_data_configuration()
    assert set(frozen.document["datasets"]) == set(DATASETS)
    assert len(frozen.sha256) == 64
    assert frozen.document["sources"]["model"]["revision"] == (
        "cbbe53366e564450558f5e639dd499171f127538"
    )
    assert frozen.document["encoding"]["document_length"] == 300
    assert frozen.document["encoding"]["query_length"] == 48
    assert frozen.document["encoding"]["do_query_expansion"] is False
    assert frozen.document["text"]["document_fields"] == ["text"]


def test_configuration_rejects_an_unpinned_dataset_revision():
    document = copy.deepcopy(load_data_configuration().document)
    document["datasets"]["scifact"]["corpus_queries_revision"] = "main"
    with pytest.raises(DataConfigurationError, match="not immutable"):
        validate_data_configuration(document)


def test_configuration_rejects_implicit_encoder_behavior():
    document = copy.deepcopy(load_data_configuration().document)
    del document["encoding"]["truncation"]
    with pytest.raises(DataConfigurationError, match="truncation"):
        validate_data_configuration(document)
