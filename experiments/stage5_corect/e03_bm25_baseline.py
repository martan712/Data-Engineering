"""Stage 5 e03: BM25 lexical reference for the e01 IR evaluation.

Single responsibility: score BM25 as a retrieval system on the identical
evaluation protocol as e01 (same evaluable test queries, same qrels, same
rank-based run dicts, k=100, metrics at 10 and 100) so the e01 quality table
gains a stack-independent lexical anchor.  BM25 shares nothing with the
embedding pipeline, so it contextualizes the absolute nDCG numbers, which
trail published BEIR values because the archive cache encoded documents
text-only (no title concatenation).  For consistency BM25 also indexes
text-only documents (bondmaxsim.baselines.bm25).

Latency: BM25 is timed with the same rep protocol as e01 (per-query loop,
query tokenization inside the timer, index build excluded, warmup + REP reps,
best kept) but on its own stack and NOT interleaved with the e01 arms: the
interleaving control exists for margins of a few percent, and BM25 sits two
orders of magnitude below the vector arms, far outside any thermal drift.

Output:
  results/json/stage5_corect_e03_bm25_baseline.json

Usage:
    uv run python -m experiments.stage5_corect.e03_bm25_baseline [dataset ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from bondmaxsim.config import REPO_ROOT
from bondmaxsim.data.beir_ids import load_qrels_tsv, qrels_path
from bondmaxsim.data.loader import load_eval_queries
from bondmaxsim.baselines.bm25 import BM25Baseline
from bondmaxsim.eval.corect import compute_rc_metrics
from bondmaxsim.eval.qrels import compute_quality_metrics

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs"]
K_RETRIEVE = 100                 # as e01; @10 metrics use the ranking prefix
REP = 5                          # as e01: warmup + REP reps, best kept

RESULTS_JSON = REPO_ROOT / "results" / "json"


def _to_run(results) -> dict[str, dict[str, float]]:
    """Rank-based run dict, identical convention to e01 (tie-handling note)."""
    run: dict[str, dict[str, float]] = {}
    for qid, (doc_ids, _) in results:
        n = len(doc_ids)
        run[qid] = {d: float(n - i) for i, d in enumerate(doc_ids)}
    return run


def run_dataset(dataset: str) -> dict:
    from datasets import load_dataset as hf_load_dataset

    qrels = load_qrels_tsv(qrels_path(dataset))
    _, query_ids = load_eval_queries(dataset)

    corpus = hf_load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    doc_ids = [str(r) for r in corpus["_id"]]
    texts = list(corpus["text"])

    queries_ds = hf_load_dataset(f"BeIR/{dataset}", "queries",
                                 split="queries")
    text_by_id = {str(i): t for i, t in
                  zip(queries_ds["_id"], queries_ds["text"])}
    missing = [q for q in query_ids if q not in text_by_id]
    if missing:
        raise ValueError(f"{dataset}: {len(missing)} eval queries missing "
                         f"from the queries split (e.g. {missing[:3]})")

    print(f"\n=== e03 {dataset}: {len(doc_ids)} docs, "
          f"{len(query_ids)} queries ===")
    bm25 = BM25Baseline()
    t0 = time.perf_counter()
    build_stats = bm25.build(doc_ids, texts)
    build_s = time.perf_counter() - t0
    print(f"  index built in {build_s:.1f}s")

    # Quality pass (untimed), one query at a time as the e01 arms run.
    query_texts = [text_by_id[q] for q in query_ids]
    ranked = [bm25.search([t], k=K_RETRIEVE)[0] for t in query_texts]
    run = _to_run(list(zip(query_ids, ranked)))
    row = compute_quality_metrics(run, qrels)
    row["CoRECT_RC_metrics"] = compute_rc_metrics(run, qrels)
    print(f"  BM25 nDCG@10={row['nDCG_at_10']:.4f} "
          f"R@100={row['recall_at_100']:.4f} MRR@10={row['MRR_at_10']:.4f}")

    # Timing: e01 rep protocol (warmup + REP reps, best kept; per-query loop
    # with tokenization inside the timer), standalone stack (see docstring).
    def one_pass():
        for t in query_texts:
            bm25.search([t], k=K_RETRIEVE)

    one_pass()                   # warmup
    reps = []
    for _ in range(REP):
        t0 = time.perf_counter()
        one_pass()
        reps.append((time.perf_counter() - t0) / len(query_texts) * 1e3)
    row["ms_per_query"] = min(reps)
    row["ms_per_query_reps"] = reps
    row["ms_per_query_mean"] = float(np.mean(reps))
    row["ms_per_query_std"] = float(np.std(reps))
    print(f"  ms/q = {row['ms_per_query']:.3f} "
          f"(mean {row['ms_per_query_mean']:.3f} "
          f"± {row['ms_per_query_std']:.3f})")
    return {"n_docs": len(doc_ids), "n_queries": len(query_ids),
            "build": {**build_stats, "build_s": build_s},
            "timing_note": "e01 rep protocol but standalone, not interleaved "
                           "with the e01 arms (see module docstring)",
            **row}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    payload = {
        "experiment": "e03_bm25_baseline",
        "purpose": "BM25 lexical quality reference on the e01 protocol "
                   "(same test queries, qrels, k, rank-based runs); indexes "
                   "text-only documents to match the embedding cache; "
                   "quality only, no latency comparison",
        "k_retrieve": K_RETRIEVE,
        "datasets": {},
    }
    t0 = time.perf_counter()
    for ds in datasets:
        payload["datasets"][ds] = run_dataset(ds)
    out = RESULTS_JSON / "stage5_corect_e03_bm25_baseline.json"
    if out.exists():
        prev = json.loads(out.read_text())
        prev_ds = prev.get("datasets", {})
        prev_ds.update(payload["datasets"])
        payload["datasets"] = prev_ds
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON: {out}\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
