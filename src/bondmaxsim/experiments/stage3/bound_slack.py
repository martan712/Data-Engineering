"""E01 bound-slack accounting on the versioned artifact path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bondmaxsim.experiments.stage3.common import (
    configuration_sha256,
    load_mechanism_data,
    write_accounting_result,
)
from bondmaxsim.oracle.bound_trajectory import doc_ub_trajectory, make_prefix_grid
from bondmaxsim.ordering.orders import bond_order, natural_order
from bondmaxsim.results.models import ExperimentResultEnvelope


@dataclass(frozen=True)
class BoundSlackConfig:
    dataset: str = "scifact"
    orders: tuple[str, ...] = ("natural", "bond", "pca")
    k: int = 10
    query_count: int = 50
    query_seed: int = 42
    grid_points: int = 33
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "BoundSlackConfig":
        return cls(dataset="synthetic-small-v1", orders=("natural", "bond"), k=3, fixture=True)


@dataclass(frozen=True)
class BoundSlackRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


def _pca_rotation(dimension: int) -> np.ndarray:
    generator = np.random.default_rng(123)
    matrix = generator.standard_normal((dimension, dimension))
    rotation, _ = np.linalg.qr(matrix)
    return rotation.astype(np.float32)


def _trajectory_row(
    *,
    order_name: str,
    queries: tuple[np.ndarray, ...],
    flat_tokens: np.ndarray,
    flat_tokens_rotated: np.ndarray,
    doc_starts: np.ndarray,
    prefix_grid: np.ndarray,
    token_mean: np.ndarray,
    rotation: np.ndarray,
) -> dict[str, Any]:
    values: list[list[float]] = [[] for _ in prefix_grid]
    for query in queries:
        if order_name == "pca":
            query_effective = (query @ rotation).astype(np.float32)
            corpus_effective = flat_tokens_rotated
            order = natural_order(query)
        elif order_name == "bond":
            query_effective = query
            corpus_effective = flat_tokens
            order = bond_order(query, token_mean)
        elif order_name == "natural":
            query_effective = query
            corpus_effective = flat_tokens
            order = natural_order(query)
        else:
            raise ValueError(f"unknown dimension order {order_name!r}")
        upper_bounds, exact_scores = doc_ub_trajectory(
            query_effective, corpus_effective, doc_starts, order, prefix_grid
        )
        denominator = np.where(exact_scores > 1e-8, exact_scores, 1e-8)
        slack = upper_bounds / denominator[:, np.newaxis]
        for index in range(len(prefix_grid)):
            values[index].extend(slack[:, index].tolist())
    means = [float(np.mean(row)) for row in values]
    d4_index = max(0, int(np.searchsorted(prefix_grid, flat_tokens.shape[1] // 4, side="right")) - 1)
    d2_index = max(0, int(np.searchsorted(prefix_grid, flat_tokens.shape[1] // 2, side="right")) - 1)
    return {
        "dimension_order": order_name,
        "query_count": len(queries),
        "prefix_grid": prefix_grid.tolist(),
        "slack_mean": means,
        "slack_p10": [float(np.percentile(row, 10)) for row in values],
        "slack_p50": [float(np.percentile(row, 50)) for row in values],
        "slack_p90": [float(np.percentile(row, 90)) for row in values],
        "slack_at_d4": means[d4_index],
        "slack_at_d2": means[d2_index],
    }


def run_bound_slack(
    config: BoundSlackConfig,
    *,
    output_dir: Path,
    session_id: str = "stage3-e01-accounting",
) -> BoundSlackRun:
    data = load_mechanism_data(
        config.dataset,
        fixture=config.fixture,
        query_count=config.query_count,
        query_seed=config.query_seed,
    )
    dimension = int(data.flat_tokens.shape[1])
    rotation = _pca_rotation(dimension)
    rotated = (data.flat_tokens @ rotation).astype(np.float32)
    token_mean = data.flat_tokens.mean(axis=0).astype(np.float32)
    prefix_grid = make_prefix_grid(dimension, min(config.grid_points, dimension + 1))
    rows = [
        _trajectory_row(
            order_name=order,
            queries=data.queries,
            flat_tokens=data.flat_tokens,
            flat_tokens_rotated=rotated,
            doc_starts=data.doc_starts,
            prefix_grid=prefix_grid,
            token_mean=token_mean,
            rotation=rotation,
        )
        for order in config.orders
    ]
    experiment_id = "stage3-e01-bound-slack"
    command = (
        "uv run python -m experiments.stage3_mechanism.e01_bound_slack "
        f"--dataset {config.dataset}" + (" --fixture" if config.fixture else "")
    )
    envelope, path = write_accounting_result(
        experiment_id=experiment_id,
        artifact_id=f"{experiment_id}-{data.dataset}",
        data=data,
        command=command,
        method_configuration={
            "instrument": "numpy-bound-trajectory",
            "configuration": asdict(config),
            "orders": list(config.orders),
        },
        rows=rows,
        configuration_hash=configuration_sha256(config),
        output_dir=output_dir,
        session_id=session_id,
        protocol={"aggregation": "all query-document pairs"},
    )
    return BoundSlackRun(envelope, path)
