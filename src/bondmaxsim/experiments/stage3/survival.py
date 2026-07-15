"""E02 direct accounting-kernel execution and survival-curve artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bondmaxsim.data.packing import DEFAULT_FETCH, build_qcum
from bondmaxsim.experiments.stage3.common import (
    MechanismData,
    configuration_sha256,
    load_mechanism_data,
    write_accounting_result,
)
from bondmaxsim.kernels.wide_block import load_wide_block_kernel, run_wide_block_accounting_validated
from bondmaxsim.oracle.agreement import batch_agreement_result, validate_boundary_tie_equivalence
from bondmaxsim.oracle.bound_trajectory import (
    dims_to_prune_pct,
    early_token_pruning_rate,
    survival_trajectories,
)
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores
from bondmaxsim.ordering.orders import natural_order
from bondmaxsim.results.models import ExperimentResultEnvelope
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.packing_cache import PackingCache
from bondmaxsim.testbed.thresholds import resolve_tau_seed
from bondmaxsim.threshold.policies import oracle_threshold


@dataclass(frozen=True)
class SurvivalConfig:
    dataset: str = "scifact"
    policies: tuple[str, ...] = ("self_bound", "oracle", "seed")
    orders: tuple[str, ...] = ("natural", "bond", "pca")
    k: int = 10
    shrink: float = 1.0
    query_count: int = 50
    query_seed: int = 42
    crosscheck_queries: int = 10
    fixture: bool = False

    @classmethod
    def fixture_config(cls) -> "SurvivalConfig":
        return cls(dataset="synthetic-small-v1", policies=("oracle",), orders=("natural",), k=3, crosscheck_queries=2, fixture=True)


@dataclass(frozen=True)
class SurvivalRun:
    envelope: ExperimentResultEnvelope
    output_path: Path


def fetch_boundary_dims(dimension: int) -> np.ndarray:
    cumulative = np.cumsum(DEFAULT_FETCH.astype(np.int64))
    used = int(np.searchsorted(cumulative, dimension)) + 1
    return np.minimum(cumulative[:used], dimension)


class PreparedWideAccountingWorkload:
    def __init__(self, data: MechanismData, k: int) -> None:
        self.data = data
        self.k = k
        self.packing = PackingCache(data.flat_tokens, data.doc_starts)
        self.lib = load_wide_block_kernel()
        self.full_scores = tuple(
            exact_maxsim_scores(query, data.flat_tokens, data.doc_starts)
            for query in data.queries
        )
        pairs = tuple(topk_from_scores(scores, k) for scores in self.full_scores)
        self.exact_ids = tuple(pair[0] for pair in pairs)
        self.exact_scores = tuple(pair[1] for pair in pairs)

    def run(self, policy: str, order_name: str, shrink: float) -> dict[str, Any]:
        config = RunConfig(
            dataset=self.data.dataset,
            method="wide_block_maxsim_bond",
            dimension_order=order_name,
            threshold_policy=policy,
            k=self.k,
            shrink=shrink,
        )
        results = []
        agreements = []
        for query_id, query, oracle_ids, oracle_scores, full_scores in zip(
            self.data.query_ids,
            self.data.queries,
            self.exact_ids,
            self.exact_scores,
            self.full_scores,
        ):
            corpus, effective_query, order = self.packing.dispatch_order_wide_corpus(query, order_name)
            cumulative = build_qcum(effective_query, order)
            threshold = resolve_tau_seed(config, query, order, order_name, self.packing)
            ids, scores, stats, doc_live, token_live = run_wide_block_accounting_validated(
                self.lib,
                corpus,
                effective_query,
                order,
                cumulative,
                shrink=shrink,
                tau_seed=threshold,
                K=self.k,
                collect_block_stats=True,
            )
            agreement = validate_boundary_tie_equivalence(
                ids.astype(np.int64),
                oracle_ids,
                oracle_scores,
                k=self.k,
                num_documents=self.packing.num_docs,
                exact_scores_by_id=full_scores,
                returned_scores=scores,
                ranked_tie_break="none",
            )
            agreements.append(agreement)
            full_cells = len(effective_query) * self.packing.T * self.packing.D
            results.append(
                {
                    "query_id": query_id,
                    "cells_scanned": int(stats[0]),
                    "cells_scanned_pct": 100.0 * int(stats[0]) / full_cells,
                    "documents_pruned": int(stats[1]),
                    "pruned_docs_pct": 100.0 * int(stats[1]) / self.packing.num_docs,
                    "tokens_pruned": int(stats[2]),
                    "tokens_pruned_pct": 100.0 * int(stats[2]) / self.packing.T,
                    "block_doc_live": doc_live.tolist(),
                    "block_token_live": token_live.tolist(),
                }
            )
        summary = batch_agreement_result(agreements)
        if not summary.exact_gate_passed:
            raise RuntimeError(f"E02 exact gate failed for {policy}/{order_name}: {summary.failure_codes}")
        return {"queries": results, "agreement": summary.to_dict()}


def _aggregate_arm(
    result: dict[str, Any],
    *,
    policy: str,
    order: str,
    dimensions: np.ndarray,
    number_documents: int,
    number_tokens: int,
    dimension: int,
) -> dict[str, Any]:
    queries = result["queries"]
    used = len(dimensions)
    doc_survival = np.mean(
        [np.asarray(row["block_doc_live"][:used], dtype=np.float64) / number_documents for row in queries],
        axis=0,
    )
    token_survival = np.mean(
        [np.asarray(row["block_token_live"][:used], dtype=np.float64) / number_tokens for row in queries],
        axis=0,
    )
    return {
        "threshold_policy": policy,
        "dimension_order": order,
        "query_count": len(queries),
        "cells_scanned_pct": float(np.mean([row["cells_scanned_pct"] for row in queries])),
        "pruned_docs_pct": float(np.mean([row["pruned_docs_pct"] for row in queries])),
        "tokens_pruned_pct": float(np.mean([row["tokens_pruned_pct"] for row in queries])),
        "fetch_boundary_dims": dimensions.tolist(),
        "doc_survival": doc_survival.tolist(),
        "token_survival": token_survival.tolist(),
        "dims_to_prune_50pct_docs": dims_to_prune_pct(doc_survival, dimensions, 50),
        "dims_to_prune_90pct_docs": dims_to_prune_pct(doc_survival, dimensions, 90),
        "early_token_pruning_rate": early_token_pruning_rate(token_survival, dimensions, dimension),
        "agreement": result["agreement"],
        "per_query_accounting": queries,
    }


def _numpy_crosscheck(
    data: MechanismData,
    dimensions: np.ndarray,
    kernel_queries: list[dict[str, Any]],
    k: int,
    query_count: int,
) -> dict[str, Any]:
    selected_queries = data.queries[:query_count]
    doc_rows, token_rows = [], []
    for query in selected_queries:
        scores = exact_maxsim_scores(query, data.flat_tokens, data.doc_starts)
        tau = oracle_threshold(scores, k)
        docs, tokens = survival_trajectories(
            query,
            data.flat_tokens,
            data.doc_starts,
            natural_order(query),
            tau,
            dimensions,
        )
        doc_rows.append(docs)
        token_rows.append(tokens)
    numpy_docs = np.mean(doc_rows, axis=0)
    numpy_tokens = np.mean(token_rows, axis=0)
    kernel_docs = np.mean(
        [np.asarray(row["block_doc_live"][: len(dimensions)]) / len(data.doc_starts) for row in kernel_queries[:query_count]],
        axis=0,
    )
    kernel_tokens = np.mean(
        [np.asarray(row["block_token_live"][: len(dimensions)]) / len(data.flat_tokens) for row in kernel_queries[:query_count]],
        axis=0,
    )
    return {
        "arm": "oracle-natural",
        "query_count": query_count,
        "numpy_doc_survival": numpy_docs.tolist(),
        "numpy_token_survival": numpy_tokens.tolist(),
        "kernel_doc_survival": kernel_docs.tolist(),
        "kernel_token_survival": kernel_tokens.tolist(),
        "doc_max_abs_diff": float(np.max(np.abs(kernel_docs - numpy_docs))),
        "token_max_abs_diff": float(np.max(np.abs(kernel_tokens - numpy_tokens))),
    }


def run_survival(
    config: SurvivalConfig,
    *,
    output_dir: Path,
    session_id: str = "stage3-e02-accounting",
) -> SurvivalRun:
    data = load_mechanism_data(config.dataset, fixture=config.fixture, query_count=config.query_count, query_seed=config.query_seed)
    workload = PreparedWideAccountingWorkload(data, config.k)
    dimensions = fetch_boundary_dims(workload.packing.D)
    raw: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    for policy in config.policies:
        for order in config.orders:
            result = workload.run(policy, order, config.shrink)
            raw[(policy, order)] = result
            rows.append(
                _aggregate_arm(
                    result,
                    policy=policy,
                    order=order,
                    dimensions=dimensions,
                    number_documents=workload.packing.num_docs,
                    number_tokens=workload.packing.T,
                    dimension=workload.packing.D,
                )
            )
    if ("oracle", "natural") in raw:
        crosscheck_count = min(config.crosscheck_queries, len(data.queries))
        crosscheck = _numpy_crosscheck(data, dimensions, raw[("oracle", "natural")]["queries"], config.k, crosscheck_count)
        rows.append({"row_type": "numpy-crosscheck", "query_count": crosscheck_count, **crosscheck})
    experiment_id = "stage3-e02-survival"
    command = (
        "uv run python -m experiments.stage3_mechanism.e02_pruning_rate "
        f"--dataset {config.dataset}" + (" --fixture" if config.fixture else "")
    )
    envelope, path = write_accounting_result(
        experiment_id=experiment_id,
        artifact_id=f"{experiment_id}-{data.dataset}",
        data=data,
        command=command,
        method_configuration={
            "instrument": "wide-block-accounting-native",
            "configuration": asdict(config),
            "policies": list(config.policies),
            "orders": list(config.orders),
        },
        rows=rows,
        configuration_hash=configuration_sha256(config),
        output_dir=output_dir,
        session_id=session_id,
        protocol={"native_operation": "one validated accounting pass per arm", "timing_evidence": False},
    )
    return SurvivalRun(envelope, path)
