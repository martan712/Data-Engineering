from __future__ import annotations

import numpy as np

from bondmaxsim.data.fixture import fixture_arrays
from bondmaxsim.experiments.mechanism import PreparedFusedWorkload
from bondmaxsim.testbed.config import RunConfig


def _fixture_workload():
    arrays = fixture_arrays()
    starts = arrays["query_starts"]
    values = arrays["query_values"]
    queries = [
        values[int(start):int(starts[index + 1]) if index + 1 < len(starts) else len(values)]
        for index, start in enumerate(starts)
    ]
    return PreparedFusedWorkload(
        arrays["doc_values"],
        arrays["doc_starts"],
        queries,
        [f"q{index}" for index in range(len(queries))],
    )


def test_prepared_native_arm_separates_operation_validation_and_accounting():
    workload = _fixture_workload()
    config = RunConfig(
        dataset="synthetic-small-v1",
        method="fused_panel_maxsim_bond",
        dimension_order="natural",
        threshold_policy="oracle",
        k=3,
        shrink=1.0,
        checkpoints=(8,),
    )
    arm = workload.prepare_arm(config, scanner="bond", n_threads=1)
    result = arm.operation()
    agreement = arm.validate(result)
    metadata = arm.metadata(result)
    assert agreement.exact_gate_passed
    assert agreement.query_count == 4
    assert len(metadata["pruning_accounting"]) == 4
    assert all(
        0 <= row["cells_scanned_pct"] <= 100
        for row in metadata["pruning_accounting"]
    )


def test_prepared_dense_arm_reports_full_accounting():
    workload = _fixture_workload()
    config = RunConfig(
        dataset="synthetic-small-v1",
        method="fused_panel_maxsim_bond",
        dimension_order="natural",
        threshold_policy="none",
        k=3,
        shrink=1.0,
    )
    arm = workload.prepare_arm(config, scanner="brute", n_threads=1)
    result = arm.operation()
    assert arm.validate(result).exact_gate_passed
    assert all(
        row["cells_scanned_pct"] == 100.0
        for row in arm.metadata(result)["pruning_accounting"]
    )
