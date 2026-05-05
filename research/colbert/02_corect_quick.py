"""
Quick ColBERT baseline on scifact (BEIR) using the same dataset format as CoRECT.

Pipeline:
  1. Load scifact from mteb HuggingFace (CoRECT-compatible BEIR format)
  2. Encode corpus + queries with PyLate GTE-ModernColBERT-v1
  3. Build PLAID index, retrieve top-100
  4. Compute nDCG@10, recall@100, MRR@10 with ranx
  5. Save results/corect_quick.json

This is the ColBERT baseline that BOND will need to beat in latency at fixed recall.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from pylate import indexes, models, retrieve
from ranx import Qrels, Run, evaluate

OUT = Path("results")
OUT.mkdir(exist_ok=True)
INDEX_DIR = OUT / "plaid_index"

DATASET = "scifact"
TOP_K = 100

# ---------------------------------------------------------------------------
# 1. Load scifact (BEIR/MTEB format, same as CoRECT uses internally)
# ---------------------------------------------------------------------------
print(f"Loading {DATASET} ...")
ds_queries = load_dataset(f"mteb/{DATASET}", "queries", split="queries")
ds_qrels   = load_dataset(f"mteb/{DATASET}", "default", split="test")
ds_corpus  = load_dataset(f"mteb/{DATASET}", "corpus",  split="corpus")

qrels_dict: dict[str, dict[str, int]] = defaultdict(dict)
for row in ds_qrels:
    qrels_dict[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

queries = {str(q["_id"]): q["text"] for q in ds_queries if str(q["_id"]) in qrels_dict}
corpus  = {str(d["_id"]): f"{d['title']} {d['text']}".strip() for d in ds_corpus}

print(f"  queries: {len(queries)}  corpus: {len(corpus)}")

# ---------------------------------------------------------------------------
# 2. Encode
# ---------------------------------------------------------------------------
print("Loading ColBERT model ...")
model = models.ColBERT("lightonai/GTE-ModernColBERT-v1")

query_ids   = list(queries.keys())
corpus_ids  = list(corpus.keys())
query_texts = [queries[qid] for qid in query_ids]
corpus_texts= [corpus[cid] for cid in corpus_ids]

print("Encoding corpus ...")
t0 = time.perf_counter()
doc_embs = model.encode(corpus_texts, is_query=False, show_progress_bar=True, batch_size=64)
encode_corpus_s = time.perf_counter() - t0

print("Encoding queries ...")
t0 = time.perf_counter()
q_embs = model.encode(query_texts, is_query=True, show_progress_bar=True, batch_size=64)
encode_queries_s = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 3. Build PLAID index + retrieve
# ---------------------------------------------------------------------------
print("Building PLAID index ...")
t0 = time.perf_counter()
index = indexes.PLAID(
    index_folder=str(INDEX_DIR),
    index_name=DATASET,
    override=True,
)
index = index.add_documents(
    documents_ids=corpus_ids,
    documents_embeddings=doc_embs,
)
index_build_s = time.perf_counter() - t0

print(f"Retrieving top-{TOP_K} ...")
retriever = retrieve.ColBERT(index=index)
t0 = time.perf_counter()
raw_results = retriever.retrieve(queries_embeddings=q_embs, k=TOP_K)
retrieve_s = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 4. Evaluate with ranx (same metrics as CoRECT)
# ---------------------------------------------------------------------------
run_dict: dict[str, dict[str, float]] = {}
for qid, hits in zip(query_ids, raw_results):
    run_dict[qid] = {h["id"]: float(h["score"]) for h in hits}

qrels = Qrels(dict(qrels_dict))
run   = Run(run_dict)
metrics = evaluate(qrels, run, ["ndcg@10", "ndcg@100", "recall@10", "recall@100", "map@10", "mrr@10"])

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
output = {
    "dataset": DATASET,
    "model": "lightonai/GTE-ModernColBERT-v1",
    "corpus_size": len(corpus),
    "n_queries": len(queries),
    "top_k": TOP_K,
    "timing": {
        "encode_corpus_s":  round(encode_corpus_s,  2),
        "encode_queries_s": round(encode_queries_s, 2),
        "index_build_s":    round(index_build_s,    2),
        "retrieve_s":       round(retrieve_s,       2),
        "ms_per_query":     round(retrieve_s / len(queries) * 1000, 2),
    },
    "metrics": {k: round(v, 4) for k, v in metrics.items()},
}

print(json.dumps(output, indent=2))
(OUT / "corect_quick.json").write_text(json.dumps(output, indent=2))
print(f"\nSaved to {OUT / 'corect_quick.json'}")