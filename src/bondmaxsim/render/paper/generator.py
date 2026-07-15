"""Generate publication inputs exclusively from validated saved artifacts.

The artifact IDs are stable publication contracts.  Catalog paths may change
when provisional results are replaced by final reruns, but the derivations and
commands in this module do not.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.results import load_result
from bondmaxsim.results.models import ExperimentResultEnvelope

GENERATED_ROOT = Path("report/generated")
DATASETS = ("nfcorpus", "scifact", "arguana", "scidocs")
DISPLAY_DATASET = {
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
    "arguana": "ArguAna",
    "scidocs": "SciDocs",
}


class StalePaperOutputError(RuntimeError):
    """Raised when tracked publication inputs do not match their artifacts."""


def _catalog_paths(workspace: Path) -> dict[str, Path]:
    path = workspace / "artifacts/catalog.yaml"
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"artifact catalog is not the supported JSON-subset YAML: {error}") from error
    return {
        entry["artifact_id"]: workspace / entry["path"]
        for entry in document.get("artifacts", [])
        if isinstance(entry, Mapping)
        and isinstance(entry.get("artifact_id"), str)
        and isinstance(entry.get("path"), str)
    }


class _Artifacts:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.catalog_paths = _catalog_paths(workspace)

    def envelope(self, artifact_id: str) -> ExperimentResultEnvelope:
        path = self.catalog_paths.get(
            artifact_id, self.workspace / "results/json" / f"{artifact_id}.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"artifact {artifact_id!r} has no readable path: {path}")
        return load_result(path)

    def data(self, artifact_id: str) -> Mapping[str, Any]:
        """Return science data from a validated current or migrated envelope."""
        envelope = self.envelope(artifact_id)
        source = envelope.payload.get("source_data")
        return source if isinstance(source, Mapping) else envelope.payload


def _csv(rows: Iterable[Mapping[str, Any]], fields: tuple[str, ...]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return stream.getvalue()


def _dense_rows(artifacts: _Artifacts) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        data = artifacts.data(f"stage3_mechanism_e03_order_ablation_{dataset}")
        arms = data.get("arms")
        if not isinstance(arms, list):
            raise ValueError(f"{dataset} E03 artifact has no arms")
        by_order = {arm.get("dimension_order"): arm for arm in arms}
        try:
            fused, blas = by_order["dense_fused"], by_order["dense_numpy"]
        except KeyError as error:
            raise ValueError(f"{dataset} E03 artifact lacks dense controls") from error
        rows.append(
            {
                "dataset": dataset,
                "fused_1t_ms_per_query": float(fused["ms_per_query_1t"]),
                "blas_1t_ms_per_query": float(blas["ms_per_query_1t"]),
                "fused_mt_ms_per_query": float(fused["ms_per_query_mt"]),
                "blas_mt_ms_per_query": float(blas["ms_per_query_mt"]),
            }
        )
    return rows


def _render_dense_table(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            f"{DISPLAY_DATASET[row['dataset']]} & {row['fused_1t_ms_per_query']:.1f} & "
            f"{row['blas_1t_ms_per_query']:.1f} & {row['fused_mt_ms_per_query']:.1f} & "
            f"{row['blas_mt_ms_per_query']:.1f} \\\\"
        )
    return "\n".join(rendered) + "\n"


def _exact_safe_rows(artifacts: _Artifacts) -> list[dict[str, Any]]:
    data = artifacts.data("stage3_mechanism_r12c_interleaved_exact_safe_mt")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("R12c artifact has no per-dataset results")
    by_dataset = {row["dataset"]: row for row in results}
    rows = []
    for dataset in DATASETS:
        source = by_dataset[dataset]
        best = max(source["arms"], key=lambda arm: arm["margin_vs_interleaved_dense_pct"])
        prune = best.get("pruned_docs_pct")
        if prune is None:
            prune = best.get("e08_ref", {}).get("e08_prune_pct")
        if prune is None:
            raise ValueError(f"R12c {dataset} best arm lacks pruning accounting")
        rows.append(
            {
                "dataset": dataset,
                "dense_ms_per_query": float(source["dense_interleaved_ms_per_query"]),
                "margin_pct": float(best["margin_vs_interleaved_dense_pct"]),
                "docs_pruned_pct": float(prune),
                "checkpoints": ",".join(str(value) for value in best["checkpoints"]),
            }
        )
    return rows


def _render_exact_safe_table(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            f"{DISPLAY_DATASET[row['dataset']]} & {row['dense_ms_per_query']:.1f} & "
            f"${row['margin_pct']:+.1f}\\%$ & {row['docs_pruned_pct']:.0f}\\% \\\\"
        )
    return "\n".join(rendered) + "\n"


def _historical_rows(data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    rows = data.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"historical artifact has no {key!r} rows")
    return rows


def _bound_slack_plot_data(artifacts: _Artifacts) -> str:
    data = artifacts.data("stage3_mechanism_e01_bound_slack_scifact")
    prefix = data["prefix_grid"]
    rows = []
    for order in data["orders"]:
        series = data["results"][order]
        for index, dimension in enumerate(prefix):
            rows.append(
                {
                    "dataset": "scifact",
                    "dimension_order": order,
                    "dimensions_scanned": dimension,
                    "slack_mean": series["slack_mean"][index],
                    "slack_p10": series["slack_p10"][index],
                    "slack_p90": series["slack_p90"][index],
                }
            )
    return _csv(rows, ("dataset", "dimension_order", "dimensions_scanned", "slack_mean", "slack_p10", "slack_p90"))


def _survival_plot_data(artifacts: _Artifacts) -> str:
    data = artifacts.data("stage3_mechanism_e02_pruning_rate_scifact")
    dimensions = data["fetch_boundary_dims"]
    rows = []
    for arm in _historical_rows(data, "arms"):
        for index, dimension in enumerate(dimensions):
            rows.append(
                {
                    "dataset": "scifact",
                    "threshold_policy": arm["threshold_policy"],
                    "dimension_order": arm["dimension_order"],
                    "dimensions_scanned": dimension,
                    "document_survival": arm["doc_survival"][index],
                    "token_survival": arm["token_survival"][index],
                }
            )
    return _csv(rows, ("dataset", "threshold_policy", "dimension_order", "dimensions_scanned", "document_survival", "token_survival"))


def _checkpoint_plot_data(artifacts: _Artifacts) -> str:
    data = artifacts.data("stage3_mechanism_e08_checkpoint_ablation_scifact")
    rows = []
    for arm in _historical_rows(data, "arms"):
        rows.append(
            {
                "dataset": "scifact",
                "dimension_order": arm["dimension_order"],
                "checkpoints": ",".join(str(value) for value in arm["checkpoints"]),
                "ms_per_query_1t": arm["ms_per_query_1t"],
                "dense_ms_per_query_1t": data["dense_fused_baseline"]["ms_per_query_1t"],
                "recall_vs_exact_at_10": arm["recall_vs_exact_at_10"],
                "docs_pruned_pct": arm["pruned_docs_pct_fused"],
            }
        )
    return _csv(rows, ("dataset", "dimension_order", "checkpoints", "ms_per_query_1t", "dense_ms_per_query_1t", "recall_vs_exact_at_10", "docs_pruned_pct"))


def _fixed_candidate_plot_data(artifacts: _Artifacts) -> str:
    data = artifacts.data("stage4_integration_e01_fixed_candidate_arms_scidocs")
    rows = []
    for arm in data["runs"]["mt"]["arms"]:
        rows.append(
            {
                "dataset": "scidocs",
                "arm": arm["arm"],
                "candidate_budget": arm.get("candidate_budget"),
                "ms_per_query_total": arm["ms_per_query_total"],
                "recall_vs_exact_at_10": arm["recall_vs_exact_at_10"],
                "comparison_scope": "uncapped_reference" if arm["arm"] == "dense_fused" else "historical_system_cap",
            }
        )
    return _csv(rows, ("dataset", "arm", "candidate_budget", "ms_per_query_total", "recall_vs_exact_at_10", "comparison_scope"))


def _headline_macros(
    artifacts: _Artifacts,
    dense_rows: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
) -> str:
    dense_speedups = [row["blas_mt_ms_per_query"] / row["fused_mt_ms_per_query"] for row in dense_rows]
    fixed = artifacts.data("stage4_integration_e01_fixed_candidate_arms_scidocs")["runs"]["mt"]["arms"]
    fixed_by_arm = {arm["arm"]: arm for arm in fixed}
    recovery = []
    seed_costs = []
    for dataset in DATASETS:
        source = artifacts.data(f"stage4_integration_e02_seeded_tau_recovery_{dataset}")
        for arm in source["runs"]["mt"]["arms"]:
            if arm["arm"] == "ivf_seed_strong":
                recovery.append(100.0 * float(arm["prune_recovery_vs_oracle"]))
                seed_costs.append(float(arm["seed_cost_ms_per_query"]))
    return "".join(
        (
            "% Generated by bondmaxsim.render.paper; do not edit.\n",
            f"\\newcommand{{\\DenseMedianSpeedup}}{{{median(dense_speedups):.1f}\\ensuremath{{\\times}}}}\n",
            f"\\newcommand{{\\ExactSafeBestMargin}}{{{max(row['margin_pct'] for row in exact_rows):.1f}\\%}}\n",
            f"\\newcommand{{\\ExactSafePrunedRange}}{{{min(row['docs_pruned_pct'] for row in exact_rows):.0f}--{max(row['docs_pruned_pct'] for row in exact_rows):.0f}\\%}}\n",
            f"\\newcommand{{\\ExactSafeWinningDatasets}}{{{sum(row['margin_pct'] > 3.0 for row in exact_rows)}}}\n",
            f"\\newcommand{{\\SciDocsDenseLatency}}{{{fixed_by_arm['dense_fused']['ms_per_query_total']:.1f}~ms/q}}\n",
            f"\\newcommand{{\\SciDocsFaissHundredLatency}}{{{fixed_by_arm['faiss@100']['ms_per_query_total']:.1f}~ms/q}}\n",
            f"\\newcommand{{\\SciDocsPdxHundredLatency}}{{{fixed_by_arm['pdx@100']['ms_per_query_total']:.1f}~ms/q}}\n",
            f"\\newcommand{{\\StrongSeedRecovery}}{{{max(recovery):.0f}\\%}}\n",
            f"\\newcommand{{\\StrongSeedCostRange}}{{{min(seed_costs):.0f}--{max(seed_costs):.0f}~ms/q}}\n",
        )
    )


def _render_experiment_table() -> str:
    rows = (
        ("RQ1", "E1 bound slack; E2 survival", "accounting kernel"),
        ("RQ2", "E3 kernels (dense arm)", "fused kernel"),
        ("RQ3", "E3 kernels; E4 checkpoint sets; E5 bound tightness", "fused kernel, interleaved"),
        ("RQ4", "E6 fixed budgets; E7 seeding; E8 partition frontier", "full pipelines, interleaved"),
        ("--", "E9 retrieval quality", "pipelines, qrels"),
    )
    return "".join(f"{rq} & {experiments} & {instrument} \\\\\n" for rq, experiments, instrument in rows)


def _ir_table_rows(artifacts: _Artifacts) -> str:
    bm25 = artifacts.data("stage5_corect_e03_bm25_baseline")["datasets"]
    order = ("scifact", "nfcorpus", "arguana", "scidocs")
    selected = {
        "scifact": ("dense_fused", "openblas", "bond_exact_safe", "partitioned@16", "partitioned@32", "faiss@1000", "plaid@1000"),
        "nfcorpus": ("dense_fused", "openblas", "bond_exact_safe", "partitioned@16", "partitioned@32", "faiss@1000", "plaid@1000"),
        "arguana": ("dense_fused", "openblas", "bond_exact_safe", "partitioned@16", "partitioned@32", "faiss@1000", "plaid@1000"),
        "scidocs": ("dense_fused", "openblas", "bond_exact_safe", "partitioned@16", "partitioned@32", "faiss@100", "faiss@1000", "plaid@1000"),
    }
    labels = {
        "dense_fused": "dense fused (exact)",
        "openblas": "OpenBLAS (exact)",
        "bond_exact_safe": "exact-safe BOND",
        "partitioned@16": "partitioned, nprobe 16",
        "partitioned@32": "partitioned, nprobe 32",
        "faiss@100": "FAISS-IVF ($B{=}100$)",
        "faiss@1000": "FAISS-IVF ($B{=}1000$)",
        "plaid@1000": "PLAID ($B{=}1000$)",
    }
    blocks = []
    for dataset in order:
        data = artifacts.data(f"stage5_corect_e01_ir_evaluation_{dataset}")
        arms = {arm["arm"]: arm for arm in data["runs"]["mt"]["arms"]}
        rows = []
        for arm_id in selected[dataset]:
            arm = arms[arm_id]
            rows.append(
                f" & {labels[arm_id]} & {arm['nDCG_at_10']:.3f} & {arm['recall_at_100']:.3f} & "
                f"{arm['MRR_at_10']:.3f} & {arm['recall_vs_exact_at_10']:.3f} & {arm['ms_per_query']:.1f} \\\\"
            )
        lexical = bm25[dataset]
        rows.append(
            f" & BM25 & {lexical['nDCG_at_10']:.3f} & {lexical['recall_at_100']:.3f} & "
            f"{lexical['MRR_at_10']:.3f} & -- & {lexical['ms_per_query']:.1f} \\\\"
        )
        blocks.append(f"\\multirow{{{len(rows)}}}{{*}}{{{DISPLAY_DATASET[dataset]}}}\n" + "\n".join(rows))
    return "\n\\midrule\n".join(blocks) + "\n"


def _render_outputs(workspace: Path) -> dict[Path, str]:
    artifacts = _Artifacts(workspace)
    dense = _dense_rows(artifacts)
    exact = _exact_safe_rows(artifacts)
    return {
        GENERATED_ROOT / "tables/experiment-map-rows.tex": _render_experiment_table(),
        GENERATED_ROOT / "tables/dense-kernel-rows.tex": _render_dense_table(dense),
        GENERATED_ROOT / "tables/exact-safe-rows.tex": _render_exact_safe_table(exact),
        GENERATED_ROOT / "tables/ir-quality-rows.tex": _ir_table_rows(artifacts),
        GENERATED_ROOT / "headline-macros.tex": _headline_macros(artifacts, dense, exact),
        GENERATED_ROOT / "plot-data/bound-slack-scifact.csv": _bound_slack_plot_data(artifacts),
        GENERATED_ROOT / "plot-data/survival-scifact.csv": _survival_plot_data(artifacts),
        GENERATED_ROOT / "plot-data/checkpoint-ablation-scifact.csv": _checkpoint_plot_data(artifacts),
        GENERATED_ROOT / "plot-data/fixed-candidates-scidocs.csv": _fixed_candidate_plot_data(artifacts),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def generate_paper_assets(*, workspace: Path | str = REPO_ROOT) -> tuple[Path, ...]:
    """Write every deterministic paper input from saved artifacts."""
    root = Path(workspace).resolve()
    outputs = _render_outputs(root)
    paths = []
    for relative, content in sorted(outputs.items(), key=lambda item: str(item[0])):
        destination = root / relative
        _atomic_write_text(destination, content)
        paths.append(destination)
    return tuple(paths)


def check_generated_outputs(*, workspace: Path | str = REPO_ROOT) -> tuple[Path, ...]:
    """Return generated paths or raise without writing when any output is stale."""
    root = Path(workspace).resolve()
    outputs = _render_outputs(root)
    stale = []
    paths = []
    for relative, expected in sorted(outputs.items(), key=lambda item: str(item[0])):
        path = root / relative
        paths.append(path)
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            stale.append(f"missing {relative}")
            continue
        if actual != expected:
            stale.append(f"stale {relative}")
    if stale:
        raise StalePaperOutputError("paper generation check failed: " + "; ".join(stale))
    return tuple(paths)
