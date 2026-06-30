"""
ColBERT brute-force MaxSim baseline on scifact (BEIR).

Differences from 02_corect_quick.py (PLAID):
  - No IVF index: computes exact MaxSim for every (query, doc) pair
  - Embeddings are cached to disk after first encode (skip the 19-min re-encode)
  - This is the correct baseline for BOND, which speeds up exact search

Outputs:
  results/corect_bruteforce.json   — metrics + timing
  results/embeddings/              — cached raw embeddings (reused on next run)
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from pylate import models
from pylate.scores import colbert_scores
from torch.nn.utils.rnn import pad_sequence
from ranx import Qrels, Run, evaluate

OUT       = Path("results")
EMB_DIR   = OUT / "embeddings"
OUT.mkdir(exist_ok=True)
EMB_DIR.mkdir(exist_ok=True)

DATASET = "scifact"
TOP_K   = 100

# ---------------------------------------------------------------------------
# 1. Load scifact (same MTEB/CoRECT format)
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

query_ids    = list(queries.keys())
corpus_ids   = list(corpus.keys())
query_texts  = [queries[qid] for qid in query_ids]
corpus_texts = [corpus[cid] for cid in corpus_ids]

print(f"  queries: {len(queries)}  corpus: {len(corpus)}")

# ---------------------------------------------------------------------------
# 2. Encode (or load from cache)
# ---------------------------------------------------------------------------
doc_cache   = EMB_DIR / "doc_embs.pt"
query_cache = EMB_DIR / "query_embs.pt"
ids_cache   = EMB_DIR / "ids.pt"

model = models.ColBERT("lightonai/GTE-ModernColBERT-v1")

encode_corpus_s = encode_queries_s = 0.0

if doc_cache.exists() and query_cache.exists() and ids_cache.exists():
    saved = torch.load(ids_cache, weights_only=False)
    if saved["corpus_ids"] == corpus_ids and saved["query_ids"] == query_ids:
        print("Loading cached embeddings ...")
        doc_embs   = torch.load(doc_cache,   weights_only=False)
        q_embs     = torch.load(query_cache, weights_only=False)
    else:
        print("Cache id mismatch — re-encoding ...")
        doc_cache.unlink(); query_cache.unlink(); ids_cache.unlink()

if not doc_cache.exists():
    print("Encoding corpus ...")
    t0 = time.perf_counter()
    doc_embs = model.encode(corpus_texts, is_query=False, show_progress_bar=True, batch_size=64)
    encode_corpus_s = time.perf_counter() - t0
    torch.save(doc_embs, doc_cache)

    print("Encoding queries ...")
    t0 = time.perf_counter()
    q_embs = model.encode(query_texts, is_query=True, show_progress_bar=True, batch_size=64)
    encode_queries_s = time.perf_counter() - t0
    torch.save(q_embs, query_cache)
    torch.save({"corpus_ids": corpus_ids, "query_ids": query_ids}, ids_cache)
    print(f"Embeddings cached to {EMB_DIR}")

# ---------------------------------------------------------------------------
# 3. Brute-force MaxSim
#    Pad all doc embeddings once, then score one query at a time against all docs.
# ---------------------------------------------------------------------------
print("Padding document embeddings ...")
doc_tensors  = [torch.tensor(d) if not isinstance(d, torch.Tensor) else d for d in doc_embs]
doc_padded   = pad_sequence(doc_tensors, batch_first=True)        # (N, max_d_tokens, dim)
doc_lengths  = torch.tensor([d.shape[0] for d in doc_tensors])
doc_mask     = (torch.arange(doc_padded.shape[1]).unsqueeze(0) < doc_lengths.unsqueeze(1)).float()

print(f"Running brute-force MaxSim ({len(query_ids)} queries × {len(corpus_ids)} docs) ...")
t0 = time.perf_counter()

run_dict: dict[str, dict[str, float]] = {}
for i, (qid, q_emb) in enumerate(zip(query_ids, q_embs)):
    if i % 50 == 0:
        print(f"  query {i}/{len(query_ids)}")
    q_tensor = torch.tensor(q_emb) if not isinstance(q_emb, torch.Tensor) else q_emb
    q_tensor = q_tensor.unsqueeze(0)                              # (1, q_tokens, dim)

    # scores: (1, N_docs)
    scores = colbert_scores(
        queries_embeddings=q_tensor,
        documents_embeddings=doc_padded,
        documents_mask=doc_mask,
    )
    top_scores, top_idx = scores[0].topk(TOP_K)
    run_dict[qid] = {corpus_ids[idx]: float(s) for idx, s in zip(top_idx.tolist(), top_scores.tolist())}

retrieve_s = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
qrels  = Qrels(dict(qrels_dict))
run    = Run(run_dict)
metrics = evaluate(qrels, run, ["ndcg@10", "ndcg@100", "recall@10", "recall@100", "map@10", "mrr@10"])

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
output = {
    "dataset": DATASET,
    "model": "lightonai/GTE-ModernColBERT-v1",
    "retrieval": "brute_force_maxsim",
    "corpus_size": len(corpus),
    "n_queries": len(queries),
    "top_k": TOP_K,
    "timing": {
        "encode_corpus_s":  round(encode_corpus_s,  2),
        "encode_queries_s": round(encode_queries_s, 2),
        "retrieve_s":       round(retrieve_s,       2),
        "ms_per_query":     round(retrieve_s / len(queries) * 1000, 2),
    },
    "metrics": {k: round(v, 4) for k, v in metrics.items()},
}

print(json.dumps(output, indent=2))
(OUT / "corect_bruteforce.json").write_text(json.dumps(output, indent=2))
print(f"\nSaved to {OUT / 'corect_bruteforce.json'}")
