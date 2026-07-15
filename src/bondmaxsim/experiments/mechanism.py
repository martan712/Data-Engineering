"""Prepared single-pass native arms for shared experiment timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from bondmaxsim.data.packing import build_qcum
from bondmaxsim.kernels.fused_panel import (
    load_fused_panel_kernel,
    run_fused_panel_bond_validated,
    run_fused_panel_brute_validated,
)
from bondmaxsim.oracle.agreement import (
    AgreementBatchResult,
    batch_agreement_result,
    validate_boundary_tie_equivalence,
)
from bondmaxsim.oracle.exact_maxsim import exact_maxsim_scores, topk_from_scores
from bondmaxsim.testbed.config import RunConfig
from bondmaxsim.testbed.packing_cache import PackingCache
from bondmaxsim.testbed.thresholds import oracle_tau_seed, resolve_tau_seed


@dataclass(frozen=True)
class QueryPassResult:
    ids: np.ndarray
    scores: np.ndarray
    stats: np.ndarray


@dataclass(frozen=True)
class MechanismPassResult:
    queries: tuple[QueryPassResult, ...]


class PreparedFusedWorkload:
    """One validated corpus packing and exact oracle shared by many arms."""

    def __init__(
        self,
        flat_tokens: np.ndarray,
        doc_starts: np.ndarray,
        queries: list[np.ndarray],
        query_ids: list[str],
    ) -> None:
        if len(queries) != len(query_ids) or not queries:
            raise ValueError("queries and query IDs must be non-empty and aligned")
        self.query_ids = tuple(query_ids)
        self.queries = tuple(np.ascontiguousarray(query, dtype=np.float32) for query in queries)
        self.packing = PackingCache(flat_tokens, doc_starts)
        self.num_documents = self.packing.num_docs
        self.dimension = self.packing.D
        self.lib = load_fused_panel_kernel()
        self.exact_full_scores = tuple(
            exact_maxsim_scores(query, self.packing.flat_tokens, self.packing.doc_starts)
            for query in self.queries
        )
        self._exact_topk: dict[int, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]] = {}

    def exact_topk(self, k: int) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        if k not in self._exact_topk:
            pairs = [topk_from_scores(scores, k) for scores in self.exact_full_scores]
            self._exact_topk[k] = (
                tuple(ids for ids, _ in pairs),
                tuple(scores for _, scores in pairs),
            )
        return self._exact_topk[k]

    def prepare_arm(
        self,
        config: RunConfig,
        *,
        scanner: Literal["bond", "brute"] = "bond",
        n_threads: int = 1,
        bound: Literal["tight", "cheap"] = "tight",
        level: Literal["doc", "token"] = "doc",
    ) -> "PreparedFusedArm":
        return PreparedFusedArm(
            self,
            config,
            scanner=scanner,
            n_threads=n_threads,
            bound=bound,
            level=level,
        )


class PreparedFusedArm:
    """Prepared query inputs plus one native workload-pass operation."""

    def __init__(
        self,
        workload: PreparedFusedWorkload,
        config: RunConfig,
        *,
        scanner: Literal["bond", "brute"],
        n_threads: int,
        bound: Literal["tight", "cheap"],
        level: Literal["doc", "token"],
    ) -> None:
        self.workload = workload
        self.config = config
        self.query_ids = workload.query_ids
        self.scanner = scanner
        self.n_threads = n_threads
        self.bound = bound
        self.level = level
        self.packing = workload.packing
        self.num_documents = workload.num_documents
        self.dimension = workload.dimension
        self.lib = workload.lib
        self.exact_full_scores = workload.exact_full_scores
        self.exact_ids, self.exact_scores = workload.exact_topk(config.k)
        self.prepared: list[tuple[Any, ...]] = []
        for query_index, query in enumerate(workload.queries):
            corpus, query_effective, order = self.packing.dispatch_order_panel_corpus(
                query, config.dimension_order
            )
            if scanner == "bond":
                query_cumulative = build_qcum(query_effective, order)
                if config.threshold_policy == "oracle":
                    threshold = oracle_tau_seed(
                        self.exact_full_scores[query_index], config.k
                    )
                else:
                    threshold = resolve_tau_seed(
                        config,
                        query,
                        order,
                        config.dimension_order,
                        self.packing,
                    )
                self.prepared.append(
                    (corpus, query_effective, order, query_cumulative, threshold)
                )
            else:
                self.prepared.append((corpus, query_effective))
        self.checkpoints = (
            np.asarray(config.checkpoints, dtype=np.uint32)
            if config.checkpoints
            else None
        )

    def operation(self) -> MechanismPassResult:
        rows = []
        for prepared in self.prepared:
            if self.scanner == "bond":
                corpus, query, order, query_cumulative, threshold = prepared
                ids, scores, stats = run_fused_panel_bond_validated(
                    self.lib,
                    corpus,
                    query,
                    order,
                    query_cumulative,
                    shrink=self.config.shrink,
                    tau_seed=threshold,
                    K=self.config.k,
                    n_threads=self.n_threads,
                    level=self.level,
                    checkpoints=self.checkpoints,
                    bound=self.bound,
                )
            else:
                corpus, query = prepared
                ids, scores = run_fused_panel_brute_validated(
                    self.lib,
                    corpus,
                    query,
                    self.config.k,
                    n_threads=self.n_threads,
                )
                stats = np.array([0, 0, 0], dtype=np.uint64)
            rows.append(QueryPassResult(ids, scores, stats))
        return MechanismPassResult(tuple(rows))

    def validate(self, result: MechanismPassResult) -> AgreementBatchResult:
        agreements = []
        for query_result, exact_ids, exact_scores, full_scores in zip(
            result.queries,
            self.exact_ids,
            self.exact_scores,
            self.exact_full_scores,
        ):
            agreements.append(
                validate_boundary_tie_equivalence(
                    query_result.ids.astype(np.int64),
                    exact_ids,
                    exact_scores,
                    k=self.config.k,
                    num_documents=self.num_documents,
                    exact_scores_by_id=full_scores,
                    returned_scores=query_result.scores,
                    ranked_tie_break="none",
                )
            )
        return batch_agreement_result(agreements)

    def metadata(self, result: MechanismPassResult) -> dict[str, Any]:
        rows = []
        for query_id, prepared, query_result in zip(
            self.query_ids, self.prepared, result.queries
        ):
            if self.scanner == "bond":
                corpus, query = prepared[0], prepared[1]
                full_cells = int(len(query) * int(corpus.doc_offsets[-1]) * self.dimension)
                cells_scanned = int(query_result.stats[0])
                documents_pruned = int(query_result.stats[1])
            else:
                corpus, query = prepared
                full_cells = int(len(query) * int(corpus.doc_offsets[-1]) * self.dimension)
                cells_scanned = full_cells
                documents_pruned = 0
            rows.append(
                {
                    "query_id": query_id,
                    "cells_scanned": cells_scanned,
                    "cells_scanned_pct": (
                        100.0 * cells_scanned / full_cells if full_cells else 0.0
                    ),
                    "documents_pruned": documents_pruned,
                    "pruned_docs_pct": (
                        100.0 * documents_pruned / self.num_documents
                        if self.num_documents
                        else 0.0
                    ),
                }
            )
        return {"pruning_accounting": rows}
