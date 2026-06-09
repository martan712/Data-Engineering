"""MaxSim-aware candidate selection over saved realish PDX candidate pools.

This script reuses Iteration 7 candidate pools. It does not rebuild PDX or
re-encode embeddings. Exact reranking is simulated by preserving the order from
the saved oracle-pool exact MaxSim rerank.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

from utils_colbert import compute_qrels_metrics, save_json, topk_overlap


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
INPUT_PATH = RESULTS_DIR / "realish_coverage_preserving_selection.json"
EXACT_PATH = RESULTS_DIR / "realish_colbert_maxsim_numpy.json"
OUTPUT_PATH = RESULTS_DIR / "realish_maxsim_aware_selection.json"

L_VALUES = [10, 20, 50, 100]
C_VALUES = [20, 30, 50, 75]
REPORT_K_VALUES = [1, 3, 5, 10]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def diagnostics_by_doc(query: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        record["doc_id"]: record
        for record in query["selected_candidate_diagnostics"]
    }


def sort_docs(diagnostics: dict[str, dict[str, Any]], key_parts) -> list[str]:
    return sorted(
        diagnostics,
        key=lambda doc_id: tuple(part(doc_id, diagnostics[doc_id]) for part in key_parts),
    )


def top_by_feature(diagnostics: dict[str, dict[str, Any]], feature: str, n: int) -> list[str]:
    return sort_docs(
        diagnostics,
        [
            lambda _doc_id, record: -float(record[feature]),
            lambda _doc_id, record: int(record["doc_index"]),
        ],
    )[:n]


def combined_scores(diagnostics: dict[str, dict[str, Any]]) -> dict[str, float]:
    approx = normalize({
        doc_id: float(record["approx_score"])
        for doc_id, record in diagnostics.items()
    })
    matched = normalize({
        doc_id: float(record["matched_query_token_count"])
        for doc_id, record in diagnostics.items()
    })
    retrieved = normalize({
        doc_id: float(record["retrieved_token_count"])
        for doc_id, record in diagnostics.items()
    })
    average = normalize({
        doc_id: float(record["average_best_token_similarity"])
        for doc_id, record in diagnostics.items()
    })
    minimum = normalize({
        doc_id: float(record["min_best_token_similarity"])
        for doc_id, record in diagnostics.items()
    })
    maximum = normalize({
        doc_id: float(record["max_best_token_similarity"])
        for doc_id, record in diagnostics.items()
    })
    return {
        doc_id: (
            0.35 * approx[doc_id]
            + 0.20 * matched[doc_id]
            + 0.15 * retrieved[doc_id]
            + 0.15 * average[doc_id]
            + 0.10 * minimum[doc_id]
            + 0.05 * maximum[doc_id]
        )
        for doc_id in diagnostics
    }


def select_multi_signal_union(diagnostics: dict[str, dict[str, Any]], top_c: int) -> list[str]:
    features = [
        "approx_score",
        "retrieved_token_count",
        "matched_query_token_count",
        "average_best_token_similarity",
        "min_best_token_similarity",
    ]
    per_feature = max(1, math.ceil(top_c / len(features)))
    selected: set[str] = set()
    for feature in features:
        selected.update(top_by_feature(diagnostics, feature, per_feature))

    scores = combined_scores(diagnostics)
    ordered = sorted(
        selected,
        key=lambda doc_id: (
            -scores[doc_id],
            -float(diagnostics[doc_id]["matched_query_token_count"]),
            -float(diagnostics[doc_id]["approx_score"]),
            int(diagnostics[doc_id]["doc_index"]),
        ),
    )
    if len(ordered) < top_c:
        for doc_id in sorted(
            diagnostics,
            key=lambda doc_id: (
                -scores[doc_id],
                -float(diagnostics[doc_id]["approx_score"]),
                int(diagnostics[doc_id]["doc_index"]),
            ),
        ):
            if doc_id not in selected:
                ordered.append(doc_id)
                selected.add(doc_id)
                if len(ordered) >= top_c:
                    break
    return ordered[:top_c]


def select_balanced_feature_rounds(diagnostics: dict[str, dict[str, Any]], top_c: int) -> list[str]:
    rankings = [
        top_by_feature(diagnostics, "approx_score", len(diagnostics)),
        top_by_feature(diagnostics, "retrieved_token_count", len(diagnostics)),
        top_by_feature(diagnostics, "matched_query_token_count", len(diagnostics)),
        top_by_feature(diagnostics, "average_best_token_similarity", len(diagnostics)),
        top_by_feature(diagnostics, "min_best_token_similarity", len(diagnostics)),
    ]
    selected: list[str] = []
    seen: set[str] = set()
    max_depth = max((len(ranking) for ranking in rankings), default=0)
    for depth in range(max_depth):
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            doc_id = ranking[depth]
            if doc_id not in seen:
                selected.append(doc_id)
                seen.add(doc_id)
                if len(selected) >= top_c:
                    return selected
    return selected


def select_maxsim_proxy_score(diagnostics: dict[str, dict[str, Any]], top_c: int) -> list[str]:
    scores = combined_scores(diagnostics)
    return sorted(
        diagnostics,
        key=lambda doc_id: (
            -scores[doc_id],
            -float(diagnostics[doc_id]["matched_query_token_count"]),
            -float(diagnostics[doc_id]["approx_score"]),
            int(diagnostics[doc_id]["doc_index"]),
        ),
    )[:top_c]


def select_min_similarity_guard(diagnostics: dict[str, dict[str, Any]], top_c: int) -> list[str]:
    first_take = math.ceil(top_c * 0.35)
    selected = set(top_by_feature(diagnostics, "min_best_token_similarity", first_take))
    selected.update(top_by_feature(diagnostics, "approx_score", top_c - len(selected)))
    scores = combined_scores(diagnostics)
    ordered = sorted(
        selected,
        key=lambda doc_id: (
            -scores[doc_id],
            -float(diagnostics[doc_id]["approx_score"]),
            int(diagnostics[doc_id]["doc_index"]),
        ),
    )
    if len(ordered) < top_c:
        for doc_id in select_maxsim_proxy_score(diagnostics, len(diagnostics)):
            if doc_id not in selected:
                ordered.append(doc_id)
                selected.add(doc_id)
                if len(ordered) >= top_c:
                    break
    return ordered[:top_c]


def select_existing_policy(
    diagnostics: dict[str, dict[str, Any]],
    top_c: int,
    policy: str,
) -> list[str]:
    if policy == "maxsim_proxy_score":
        return select_maxsim_proxy_score(diagnostics, top_c)
    if policy == "multi_signal_union":
        return select_multi_signal_union(diagnostics, top_c)
    if policy == "balanced_feature_rounds":
        return select_balanced_feature_rounds(diagnostics, top_c)
    if policy == "min_similarity_guard":
        return select_min_similarity_guard(diagnostics, top_c)
    raise ValueError(f"Unknown policy: {policy}")


def exact_rerank_subset(selected: list[str], oracle_reranked: list[str]) -> list[str]:
    selected_set = set(selected)
    return [doc_id for doc_id in oracle_reranked if doc_id in selected_set]


def compare_exact(reference: list[str], reranked: list[str]) -> dict[str, Any]:
    return {
        "exact_top1_match": reference[:1] == reranked[:1],
        "exact_top3_all_recovered": set(reference[:3]).issubset(set(reranked[:3])),
        "exact_top5_all_recovered": set(reference[:5]).issubset(set(reranked[:5])),
        "exact_top10_all_recovered": set(reference[:10]).issubset(set(reranked[:10])),
        "exact_recall_at_k": {
            str(k): topk_overlap(reference, reranked, k)["ratio"]
            for k in REPORT_K_VALUES
        },
    }


def selector_loss(reference: list[str], pool: set[str], selected: set[str], k: int) -> str:
    exact_top_k = set(reference[:k])
    if exact_top_k.issubset(selected):
        return "none"
    if not exact_top_k.issubset(pool):
        return "pool_coverage_failure"
    return "selector_failure"


def evaluate_policy(
    *,
    top_l: int,
    top_c: int,
    policy: str,
    oracle_queries: list[dict[str, Any]],
    exact_rankings: dict[str, list[str]],
    qrels: dict[str, list[str]],
    full_exact_comparisons: int,
) -> dict[str, Any]:
    query_results = []
    rankings_by_query = {}
    loss_counts = {
        "5": {"pool_coverage_failure": 0, "selector_failure": 0, "none": 0},
        "10": {"pool_coverage_failure": 0, "selector_failure": 0, "none": 0},
    }

    for query in oracle_queries:
        query_id = query["query_id"]
        diagnostics = diagnostics_by_doc(query)
        selected = select_existing_policy(diagnostics, top_c, policy)
        reranked = exact_rerank_subset(selected, query["reranked_doc_ids"])
        reference = exact_rankings[query_id]
        pool_doc_ids = set(diagnostics)
        selected_doc_ids = set(selected)
        rankings_by_query[query_id] = reranked
        for k in (5, 10):
            loss_counts[str(k)][selector_loss(reference, pool_doc_ids, selected_doc_ids, k)] += 1
        query_results.append(
            {
                "query_id": query_id,
                "selected_candidate_count": len(selected),
                "selected_doc_ids": selected,
                "reranked_doc_ids": reranked,
                "comparison_to_exact": compare_exact(reference, reranked),
            }
        )

    reranked_comparisons = sum(result["selected_candidate_count"] for result in query_results)
    return {
        "policy": policy,
        "top_l": top_l,
        "top_c": top_c,
        "full_exact_comparisons": full_exact_comparisons,
        "reranked_comparisons": reranked_comparisons,
        "reranked_comparison_ratio": reranked_comparisons / full_exact_comparisons,
        "mean_selected_candidate_count": reranked_comparisons / len(query_results),
        "mean_exact_recall_at_k": {
            str(k): sum(
                result["comparison_to_exact"]["exact_recall_at_k"][str(k)]
                for result in query_results
            ) / len(query_results)
            for k in REPORT_K_VALUES
        },
        "exact_top1_match_count": sum(
            result["comparison_to_exact"]["exact_top1_match"]
            for result in query_results
        ),
        "exact_top3_all_recovered_count": sum(
            result["comparison_to_exact"]["exact_top3_all_recovered"]
            for result in query_results
        ),
        "exact_top5_all_recovered_count": sum(
            result["comparison_to_exact"]["exact_top5_all_recovered"]
            for result in query_results
        ),
        "exact_top10_all_recovered_count": sum(
            result["comparison_to_exact"]["exact_top10_all_recovered"]
            for result in query_results
        ),
        "selector_loss": loss_counts,
        "qrels_metrics": compute_qrels_metrics(
            rankings_by_query,
            qrels,
            k_values=(1, 3, 5, 10),
            mrr_k=10,
        ),
        "queries": query_results,
    }


def main() -> None:
    study = load_json(INPUT_PATH)
    exact = load_json(EXACT_PATH)
    qrels = load_json(CORPUS_DIR / "qrels.json")
    exact_rankings = {
        result["query_id"]: result["ranking_doc_ids"]
        for result in exact["query_results"]
    }
    full_exact_comparisons = study["work_reference"]["full_exact_document_comparisons"]
    policies = [
        "maxsim_proxy_score",
        "multi_signal_union",
        "balanced_feature_rounds",
        "min_similarity_guard",
    ]

    runs = []
    all_results = []
    for run in study["runs"]:
        if run["top_l"] not in L_VALUES:
            continue
        policy_results = []
        oracle_queries = run["oracle_pool_upper_bound"]["queries"]
        for policy in policies:
            for top_c in C_VALUES:
                result = evaluate_policy(
                    top_l=run["top_l"],
                    top_c=top_c,
                    policy=policy,
                    oracle_queries=oracle_queries,
                    exact_rankings=exact_rankings,
                    qrels=qrels,
                    full_exact_comparisons=full_exact_comparisons,
                )
                policy_results.append(result)
                all_results.append(result)
        runs.append(
            {
                "top_l": run["top_l"],
                "candidate_generation_seconds": run["candidate_generation_seconds"],
                "candidate_pool_coverage": run["candidate_pool_coverage"]["summary"],
                "policy_results": policy_results,
            }
        )

    best_c20_recall5 = max(
        (result for result in all_results if result["top_c"] == 20),
        key=lambda result: (
            result["mean_exact_recall_at_k"]["5"],
            result["exact_top10_all_recovered_count"],
        ),
    )
    best_c50_recall10 = max(
        (result for result in all_results if result["top_c"] == 50),
        key=lambda result: (
            result["mean_exact_recall_at_k"]["10"],
            result["exact_top10_all_recovered_count"],
        ),
    )

    output = {
        "input": str(INPUT_PATH),
        "method": (
            "Uses saved oracle-pool diagnostics from Iteration 7. Exact rerank "
            "is simulated by preserving the oracle exact MaxSim order for the "
            "selected subset."
        ),
        "documents": study["documents"],
        "queries": study["queries"],
        "work_reference": study["work_reference"],
        "l_values": L_VALUES,
        "c_values": C_VALUES,
        "policies": policies,
        "best_results": {
            "c20_recall5": {
                key: value
                for key, value in best_c20_recall5.items()
                if key != "queries"
            },
            "c50_recall10": {
                key: value
                for key, value in best_c50_recall10.items()
                if key != "queries"
            },
        },
        "runs": runs,
        "note": (
            "Comparison ratios count exact MaxSim document rerank calls only. "
            "They are not end-to-end speedup claims."
        ),
    }
    save_json(OUTPUT_PATH, output)

    print("MaxSim-aware selection study")
    print(f"Input: {INPUT_PATH}")
    print(f"Saved: {OUTPUT_PATH}")
    print("\nBest C=20 by exact recall@5")
    print(
        f"L={best_c20_recall5['top_l']} policy={best_c20_recall5['policy']} "
        f"recall@5={best_c20_recall5['mean_exact_recall_at_k']['5']:.4f} "
        f"top5={best_c20_recall5['exact_top5_all_recovered_count']}/30 "
        f"ratio={best_c20_recall5['reranked_comparison_ratio']:.4f} "
        f"loss@5={best_c20_recall5['selector_loss']['5']}"
    )
    print("\nBest C=50 by exact recall@10")
    print(
        f"L={best_c50_recall10['top_l']} policy={best_c50_recall10['policy']} "
        f"recall@10={best_c50_recall10['mean_exact_recall_at_k']['10']:.4f} "
        f"top10={best_c50_recall10['exact_top10_all_recovered_count']}/30 "
        f"ratio={best_c50_recall10['reranked_comparison_ratio']:.4f} "
        f"loss@10={best_c50_recall10['selector_loss']['10']}"
    )


if __name__ == "__main__":
    main()
