"""Analyze top-C selector failures from Iteration 7.

This script reads the saved coverage-preserving selection study and separates
pool coverage failures from selector failures. It focuses on feature patterns
for exact top-k documents that were present in the pool but not selected.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from utils_colbert import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
INPUT_PATH = RESULTS_DIR / "realish_coverage_preserving_selection.json"
EXACT_PATH = RESULTS_DIR / "realish_colbert_maxsim_numpy.json"
OUTPUT_PATH = RESULTS_DIR / "realish_selector_failure_analysis.json"

FEATURES = [
    "approx_score",
    "retrieved_token_count",
    "matched_query_token_count",
    "matched_query_token_fraction",
    "average_best_token_similarity",
    "min_best_token_similarity",
    "max_best_token_similarity",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_rankings_by_query(exact: dict[str, Any]) -> dict[str, list[str]]:
    return {
        result["query_id"]: result["ranking_doc_ids"]
        for result in exact["query_results"]
    }


def diag_by_doc(query_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        record["doc_id"]: record
        for record in query_result["selected_candidate_diagnostics"]
    }


def selected_docs(query_result: dict[str, Any]) -> set[str]:
    return set(query_result["selected_doc_ids"])


def feature_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    summary = {}
    for feature in FEATURES:
        values = [float(record[feature]) for record in records]
        summary[feature] = {
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def rank_positions(records: dict[str, dict[str, Any]], doc_id: str) -> dict[str, int]:
    positions = {}
    for feature in FEATURES:
        ranked = sorted(
            records.values(),
            key=lambda record: (-float(record[feature]), int(record["doc_index"])),
        )
        for position, record in enumerate(ranked, start=1):
            if record["doc_id"] == doc_id:
                positions[feature] = position
                break
    return positions


def analyze_policy_result(
    *,
    top_l: int,
    policy_result: dict[str, Any],
    oracle_queries_by_id: dict[str, dict[str, Any]],
    exact_rankings: dict[str, list[str]],
) -> dict[str, Any]:
    k_failures = {
        "5": {"pool_coverage_failure": 0, "selector_failure": 0, "none": 0},
        "10": {"pool_coverage_failure": 0, "selector_failure": 0, "none": 0},
    }
    selector_failure_cases = []
    missed_exact_records = []
    selected_non_exact_records = []

    for query_result in policy_result["queries"]:
        query_id = query_result["query_id"]
        exact_ranking = exact_rankings[query_id]
        oracle_query = oracle_queries_by_id[query_id]
        pool_records = diag_by_doc(oracle_query)
        pool_doc_ids = set(pool_records)
        selected_doc_ids = selected_docs(query_result)

        for k in (5, 10):
            exact_top_k = set(exact_ranking[:k])
            if exact_top_k.issubset(selected_doc_ids):
                k_failures[str(k)]["none"] += 1
                continue
            if not exact_top_k.issubset(pool_doc_ids):
                k_failures[str(k)]["pool_coverage_failure"] += 1
                continue
            k_failures[str(k)]["selector_failure"] += 1

            missing = sorted(
                exact_top_k - selected_doc_ids,
                key=lambda doc_id: exact_ranking.index(doc_id),
            )
            selected_non_exact = [
                doc_id
                for doc_id in query_result["selected_doc_ids"]
                if doc_id not in exact_top_k
            ]
            for doc_id in missing:
                record = pool_records[doc_id]
                missed_exact_records.append(record)
                selector_failure_cases.append(
                    {
                        "query_id": query_id,
                        "k": k,
                        "missing_doc_id": doc_id,
                        "exact_rank": exact_ranking.index(doc_id) + 1,
                        "feature_ranks_in_pool": rank_positions(pool_records, doc_id),
                        "diagnostics": {
                            feature: record[feature]
                            for feature in FEATURES
                        },
                    }
                )
            selected_non_exact_records.extend(
                pool_records[doc_id]
                for doc_id in selected_non_exact
                if doc_id in pool_records
            )

    return {
        "top_l": top_l,
        "top_c": policy_result["top_c"],
        "policy": policy_result["policy"],
        "mean_exact_recall_at_k": policy_result["mean_exact_recall_at_k"],
        "exact_top5_all_recovered_count": policy_result["exact_top5_all_recovered_count"],
        "exact_top10_all_recovered_count": policy_result["exact_top10_all_recovered_count"],
        "selector_loss": k_failures,
        "missed_exact_feature_summary": feature_summary(missed_exact_records),
        "selected_non_exact_feature_summary": feature_summary(selected_non_exact_records),
        "selector_failure_cases": selector_failure_cases,
    }


def main() -> None:
    study = load_json(INPUT_PATH)
    exact = load_json(EXACT_PATH)
    exact_rankings = exact_rankings_by_query(exact)

    analyzed_results = []
    for run in study["runs"]:
        top_l = run["top_l"]
        oracle_queries_by_id = {
            query["query_id"]: query
            for query in run["oracle_pool_upper_bound"]["queries"]
        }
        for policy_result in run["policy_results"]:
            if policy_result["top_c"] not in (20, 50, 75):
                continue
            analyzed_results.append(
                analyze_policy_result(
                    top_l=top_l,
                    policy_result=policy_result,
                    oracle_queries_by_id=oracle_queries_by_id,
                    exact_rankings=exact_rankings,
                )
            )

    best_c20_recall5 = max(
        (result for result in analyzed_results if result["top_c"] == 20),
        key=lambda result: (
            result["mean_exact_recall_at_k"]["5"],
            result["exact_top10_all_recovered_count"],
        ),
    )
    best_c50_recall10 = max(
        (result for result in analyzed_results if result["top_c"] == 50),
        key=lambda result: (
            result["mean_exact_recall_at_k"]["10"],
            result["exact_top10_all_recovered_count"],
        ),
    )

    output = {
        "input": str(INPUT_PATH),
        "summary": {
            "best_c20_recall5": {
                key: best_c20_recall5[key]
                for key in [
                    "top_l",
                    "top_c",
                    "policy",
                    "mean_exact_recall_at_k",
                    "selector_loss",
                    "exact_top5_all_recovered_count",
                    "exact_top10_all_recovered_count",
                ]
            },
            "best_c50_recall10": {
                key: best_c50_recall10[key]
                for key in [
                    "top_l",
                    "top_c",
                    "policy",
                    "mean_exact_recall_at_k",
                    "selector_loss",
                    "exact_top5_all_recovered_count",
                    "exact_top10_all_recovered_count",
                ]
            },
        },
        "policy_failure_analysis": analyzed_results,
    }
    save_json(OUTPUT_PATH, output)

    print("Selector failure analysis")
    print(f"Input: {INPUT_PATH}")
    print(f"Saved: {OUTPUT_PATH}")
    print("\nBest C=20 by exact recall@5")
    print(
        f"L={best_c20_recall5['top_l']} policy={best_c20_recall5['policy']} "
        f"recall@5={best_c20_recall5['mean_exact_recall_at_k']['5']:.4f} "
        f"loss@5={best_c20_recall5['selector_loss']['5']}"
    )
    print("\nBest C=50 by exact recall@10")
    print(
        f"L={best_c50_recall10['top_l']} policy={best_c50_recall10['policy']} "
        f"recall@10={best_c50_recall10['mean_exact_recall_at_k']['10']:.4f} "
        f"loss@10={best_c50_recall10['selector_loss']['10']}"
    )


if __name__ == "__main__":
    main()
