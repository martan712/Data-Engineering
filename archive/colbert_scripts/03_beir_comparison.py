"""
Run both brute-force MaxSim and PLAID retrieval on any BEIR dataset and compare.

Usage:
    python 03_beir_comparison.py --dataset fiqa
    python 03_beir_comparison.py --dataset trec-covid --top-k 100

Embeddings are cached under results/embeddings/<dataset>/ and reused on re-runs.
Results saved to results/<dataset>_comparison.json.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from pylate import indexes, models
from pylate.rank import rerank as plaid_rerank
from pylate.retrieve import ColBERT as PlaidRetriever
from pylate.scores import colbert_scores
from ranx import Qrels, Run, evaluate
from torch.nn.utils.rnn import pad_sequence

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="fiqa")
parser.add_argument("--top-k",   type=int, default=100)
parser.add_argument("--n-ivf-probe",   type=int, default=32,   help="PLAID IVF probe count")
parser.add_argument("--n-full-scores", type=int, default=8192, help="PLAID full-score candidates")
args = parser.parse_args()

DATASET      = args.dataset
TOP_K        = args.top_k
N_IVF_PROBE  = args.n_ivf_probe
N_FULL_SCORES= args.n_full_scores

OUT     = Path("results")
EMB_DIR = OUT / "embeddings" / DATASET
IDX_DIR = OUT / "plaid_index" / DATASET
OUT.mkdir(exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print(f"\n=== Loading {DATASET} ===")
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
corpus_texts = [corpus[cid]  for cid in corpus_ids]

print(f"  queries: {len(queries)}   corpus: {len(corpus)}")

# ---------------------------------------------------------------------------
# 2. Encode (with cache)
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
        doc_embs = torch.load(doc_cache,   weights_only=False)
        q_embs   = torch.load(query_cache, weights_only=False)
    else:
        print("Cache id mismatch — re-encoding ...")
        doc_cache.unlink(); query_cache.unlink(); ids_cache.unlink()

if not doc_cache.exists():
    print("Encoding corpus ...")
    t0 = time.perf_counter()
    doc_embs = model.encode(corpus_texts, is_query=False, show_progress_bar=True, batch_size=64)
    encode_corpus_s = time.perf_counter() - t0
    torch.save(doc_embs, doc_cache)
    print(f"  done in {encode_corpus_s/60:.1f} min")

    print("Encoding queries ...")
    t0 = time.perf_counter()
    q_embs = model.encode(query_texts, is_query=True, show_progress_bar=True, batch_size=64)
    encode_queries_s = time.perf_counter() - t0
    torch.save(q_embs, query_cache)
    torch.save({"corpus_ids": corpus_ids, "query_ids": query_ids}, ids_cache)
    print(f"  done in {encode_queries_s:.1f} s — embeddings cached to {EMB_DIR}")

# ---------------------------------------------------------------------------
# 3. Brute-force MaxSim
# ---------------------------------------------------------------------------
print("\n=== Brute-force MaxSim ===")
doc_tensors = [torch.tensor(d) if not isinstance(d, torch.Tensor) else d for d in doc_embs]
doc_padded  = pad_sequence(doc_tensors, batch_first=True)
doc_lengths = torch.tensor([d.shape[0] for d in doc_tensors])
doc_mask    = (torch.arange(doc_padded.shape[1]).unsqueeze(0) < doc_lengths.unsqueeze(1)).float()

t0 = time.perf_counter()
bf_run: dict[str, dict[str, float]] = {}
for i, (qid, q_emb) in enumerate(zip(query_ids, q_embs)):
    if i % 100 == 0:
        print(f"  query {i}/{len(query_ids)}")
    q_tensor = (torch.tensor(q_emb) if not isinstance(q_emb, torch.Tensor) else q_emb).unsqueeze(0)
    scores   = colbert_scores(q_tensor, doc_padded, documents_mask=doc_mask)[0]
    top_s, top_i = scores.topk(TOP_K)
    bf_run[qid] = {corpus_ids[idx]: float(s) for idx, s in zip(top_i.tolist(), top_s.tolist())}
bf_retrieve_s = time.perf_counter() - t0
print(f"  done in {bf_retrieve_s:.1f} s  ({bf_retrieve_s/len(query_ids)*1000:.1f} ms/query)")

# ---------------------------------------------------------------------------
# 4. PLAID
# ---------------------------------------------------------------------------
print(f"\n=== PLAID (n_ivf_probe={N_IVF_PROBE}, n_full_scores={N_FULL_SCORES}) ===")
t0 = time.perf_counter()
index = indexes.PLAID(
    index_folder=str(IDX_DIR),
    index_name=DATASET,
    override=True,
    n_ivf_probe=N_IVF_PROBE,
    n_full_scores=N_FULL_SCORES,
)
index = index.add_documents(documents_ids=corpus_ids, documents_embeddings=doc_embs)
plaid_build_s = time.perf_counter() - t0
print(f"  index built in {plaid_build_s:.1f} s")

retriever = PlaidRetriever(index=index)
t0 = time.perf_counter()
raw = retriever.retrieve(queries_embeddings=q_embs, k=TOP_K)
plaid_retrieve_s = time.perf_counter() - t0
print(f"  retrieved in {plaid_retrieve_s:.1f} s  ({plaid_retrieve_s/len(query_ids)*1000:.1f} ms/query)")

plaid_run = {qid: {h["id"]: float(h["score"]) for h in hits}
             for qid, hits in zip(query_ids, raw)}

# ---------------------------------------------------------------------------
# 5. Evaluate both
# ---------------------------------------------------------------------------
METRICS = ["ndcg@10", "ndcg@100", "recall@10", "recall@100", "map@10", "mrr@10"]
qrels_obj = Qrels(dict(qrels_dict))

bf_metrics    = evaluate(qrels_obj, Run(bf_run),    METRICS)
plaid_metrics = evaluate(qrels_obj, Run(plaid_run), METRICS)

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
output = {
    "dataset":   DATASET,
    "model":     "lightonai/GTE-ModernColBERT-v1",
    "corpus_size": len(corpus),
    "n_queries": len(queries),
    "top_k":     TOP_K,
    "brute_force": {
        "timing": {
            "encode_corpus_s":  round(encode_corpus_s,  2),
            "encode_queries_s": round(encode_queries_s, 2),
            "retrieve_s":       round(bf_retrieve_s, 2),
            "ms_per_query":     round(bf_retrieve_s / len(queries) * 1000, 2),
        },
        "metrics": {k: round(v, 4) for k, v in bf_metrics.items()},
    },
    "plaid": {
        "params": {"n_ivf_probe": N_IVF_PROBE, "n_full_scores": N_FULL_SCORES},
        "timing": {
            "index_build_s":  round(plaid_build_s, 2),
            "retrieve_s":     round(plaid_retrieve_s, 2),
            "ms_per_query":   round(plaid_retrieve_s / len(queries) * 1000, 2),
        },
        "metrics": {k: round(v, 4) for k, v in plaid_metrics.items()},
    },
}

print("\n" + json.dumps(output, indent=2))
out_path = OUT / f"{DATASET}_comparison.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"\nSaved to {out_path}")