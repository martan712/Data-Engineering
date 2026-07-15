from __future__ import annotations

import json
import sys

from bondmaxsim.compat.corect import (
    CORECT_PINNED_COMMIT,
    inspect_corect_checkout,
    load_corect_evaluate_results,
)
from bondmaxsim.schema import ResultRecord


def test_corect_checkout_and_feature_are_pinned():
    capabilities = inspect_corect_checkout()
    assert capabilities.revision == CORECT_PINNED_COMMIT
    assert capabilities.standard_metric_function == "corect.utils.evaluate_results"
    assert not capabilities.relevance_composition


def test_corect_import_shim_does_not_persist_on_sys_path():
    before = list(sys.path)
    assert callable(load_corect_evaluate_results())
    assert sys.path == before


def test_result_record_reads_historical_corect_field(tmp_path):
    historical = {
        "dataset": "fixture",
        "num_docs": 2,
        "num_queries": 1,
        "method": "exact",
        "candidate_budget": None,
        "dimension_order": "natural",
        "threshold_policy": "none",
        "recall_vs_exact_at_10": 1.0,
        "nDCG_at_10": 1.0,
        "recall_at_100": 1.0,
        "MRR_at_10": 1.0,
        "CoRECT_RC_metrics": {"NDCG@10": 1.0},
        "ms_per_query": 1.0,
        "qps": 1000.0,
        "cells_scanned_pct": 100.0,
        "pruned_docs_pct": 0.0,
        "bound_checks_per_query": 0,
        "machine": "fixture",
        "os": "linux",
        "thread_count": 1,
    }
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(historical), encoding="utf-8")
    record = ResultRecord.from_json(path)
    assert record.corect_standard_metrics == {"NDCG@10": 1.0}
