"""Controlled PLAID helpers kept independent from the command-line runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from benchmarking import BenchmarkObservation, sha256_directory


@dataclass(frozen=True, order=True)
class PlaidConfig:
    """The two online PLAID controls used in the validation sweep."""

    n_ivf_probe: int
    n_full_scores: int

    def __post_init__(self) -> None:
        if self.n_ivf_probe <= 0 or self.n_full_scores <= 0:
            raise ValueError("PLAID n_ivf_probe and n_full_scores must be positive")

    @property
    def label(self) -> str:
        return f"plaid_np{self.n_ivf_probe}_c{self.n_full_scores}"

    def as_dict(self) -> dict[str, int]:
        return {
            "n_ivf_probe": self.n_ivf_probe,
            "n_full_scores": self.n_full_scores,
        }


def parse_plaid_config(value: str) -> PlaidConfig:
    """Parse one CLI value written as NPROBE:FULL_SCORES."""
    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError("PLAID configurations must use NPROBE:FULL_SCORES")
    try:
        return PlaidConfig(int(pieces[0]), int(pieces[1]))
    except ValueError as error:
        raise ValueError(
            "PLAID configurations must contain two positive integers"
        ) from error


def build_plaid_indexes(
    documents: Sequence[np.ndarray],
    document_ids: Sequence[str],
    configs: Sequence[PlaidConfig],
    *,
    index_folder: Path,
    index_name: str,
    nbits: int,
    kmeans_niters: int,
    rebuild: bool,
    threads: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build or load one physical PLAID index and configured search handles."""
    if not configs:
        return {}, {}
    if nbits <= 0 or kmeans_niters <= 0 or threads <= 0:
        raise ValueError("PLAID build parameters must be positive")

    try:
        import torch
        from pylate import indexes
    except ModuleNotFoundError as error:
        raise RuntimeError("PyLate, fast-plaid, and torch are required for PLAID") from error

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before parallel work starts.
        pass

    unique_configs = list(dict.fromkeys(configs))
    index_folder.mkdir(parents=True, exist_ok=True)

    def construct(config: PlaidConfig, *, override: bool) -> Any:
        return indexes.PLAID(
            index_folder=str(index_folder),
            index_name=index_name,
            override=override,
            use_fast=True,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            n_ivf_probe=config.n_ivf_probe,
            n_full_scores=config.n_full_scores,
            show_progress=False,
            device="cpu",
            use_triton=False,
        )

    handles: dict[str, Any] = {}
    load_seconds: dict[str, float] = {}
    first = unique_configs[0]
    start = perf_counter()
    first_handle = construct(first, override=rebuild)
    construction_seconds = perf_counter() - start
    is_indexed = bool(getattr(getattr(first_handle, "_index", None), "is_indexed", False))
    build_seconds = 0.0
    if rebuild:
        start = perf_counter()
        first_handle.add_documents(
            documents_ids=list(document_ids),
            documents_embeddings=list(documents),
        )
        build_seconds = perf_counter() - start
        is_indexed = True
    elif not is_indexed:
        raise RuntimeError(
            f"PLAID index {index_folder / index_name} does not exist; use --plaid-rebuild-index"
        )
    handles[first.label] = first_handle
    load_seconds[first.label] = construction_seconds

    for config in unique_configs[1:]:
        start = perf_counter()
        handle = construct(config, override=False)
        load_seconds[config.label] = perf_counter() - start
        if not bool(getattr(getattr(handle, "_index", None), "is_indexed", False)):
            raise RuntimeError(f"PLAID index disappeared while loading {config.label}")
        handles[config.label] = handle

    physical_path = index_folder / index_name
    return handles, {
        "index_folder": str(index_folder.resolve()),
        "index_name": index_name,
        "physical_index": sha256_directory(physical_path),
        "rebuilt": rebuild,
        "constructor_seconds": construction_seconds,
        "build_seconds": build_seconds,
        "total_build_seconds": construction_seconds + build_seconds,
        "handle_load_seconds": load_seconds,
        "nbits": nbits,
        "kmeans_niters": kmeans_niters,
        "use_fast": True,
        "device": "cpu",
        "use_triton": False,
        "documents": len(document_ids),
        "indexed": is_indexed,
    }


def plaid_action(
    index: Any,
    queries: Sequence[np.ndarray],
    document_ids: Sequence[str],
    k: int,
) -> Callable[[], BenchmarkObservation]:
    """Return one complete online PLAID action with strict output validation."""
    id_to_index = {str(document_id): index for index, document_id in enumerate(document_ids)}
    query_inputs = [np.ascontiguousarray(query, dtype=np.float32) for query in queries]

    def run() -> BenchmarkObservation:
        start = perf_counter()
        raw_results = index(query_inputs, k=k)
        retrieve_seconds = perf_counter() - start
        if len(raw_results) != len(query_inputs):
            raise RuntimeError(
                f"PLAID returned {len(raw_results)} query rows for {len(query_inputs)} queries"
            )

        rankings: list[list[int]] = []
        scores: list[list[float]] = []
        for query_results in raw_results:
            ranking: list[int] = []
            row_scores: list[float] = []
            seen: set[int] = set()
            for hit in query_results:
                document_id = str(hit["id"])
                if document_id not in id_to_index:
                    raise RuntimeError(f"PLAID returned unknown document id {document_id!r}")
                document_index = id_to_index[document_id]
                if document_index in seen:
                    continue
                seen.add(document_index)
                ranking.append(document_index)
                row_scores.append(float(hit["score"]))
            if len(ranking) != k:
                raise RuntimeError(f"PLAID returned {len(ranking)} unique documents; expected {k}")
            rankings.append(ranking)
            scores.append(row_scores)

        return BenchmarkObservation(
            value={"rankings": rankings, "scores": scores},
            stage_seconds={"opaque_plaid_retrieval": retrieve_seconds},
        )

    return run


def select_validation_configs(
    plaid_arms: Mapping[str, Mapping[str, Any]],
    *,
    recall_targets: Sequence[float] = (0.85, 0.95),
) -> dict[str, Any]:
    """Select release points using validation quality and latency only."""
    records = []
    for name, arm in plaid_arms.items():
        config = PlaidConfig(
            int(arm["config"]["n_ivf_probe"]),
            int(arm["config"]["n_full_scores"]),
        )
        records.append(
            {
                "name": name,
                "config": config,
                "recall@10": float(arm["exact_recovery"]["mean_recall@10"]),
                "median_seconds": float(
                    arm["timing"]["end_to_end_summary"]["median"]
                ),
            }
        )
    if not records:
        raise ValueError("Validation result contains no PLAID arms")

    selected: list[dict[str, Any]] = []
    decisions = []
    for target in recall_targets:
        eligible = [record for record in records if record["recall@10"] >= target]
        if not eligible:
            decisions.append({"criterion": f"fastest_recall_at_least_{target}", "selected": None})
            continue
        choice = min(
            eligible,
            key=lambda item: (
                item["median_seconds"],
                -item["recall@10"],
                item["config"].n_full_scores,
                item["config"].n_ivf_probe,
            ),
        )
        selected.append(choice)
        decisions.append(
            {
                "criterion": f"fastest_recall_at_least_{target}",
                "selected": choice["name"],
            }
        )

    best_quality = min(
        records,
        key=lambda item: (
            -item["recall@10"],
            item["median_seconds"],
            item["config"].n_full_scores,
            item["config"].n_ivf_probe,
        ),
    )
    selected.append(best_quality)
    decisions.append(
        {
            "criterion": "highest_recall_then_fastest",
            "selected": best_quality["name"],
        }
    )

    deduplicated = list({item["name"]: item for item in selected}.values())
    return {
        "recall_targets": list(recall_targets),
        "decisions": decisions,
        "selected": [
            {
                "name": item["name"],
                "config": item["config"].as_dict(),
                "validation_exact_recall@10": item["recall@10"],
                "validation_median_seconds": item["median_seconds"],
            }
            for item in deduplicated
        ],
    }
