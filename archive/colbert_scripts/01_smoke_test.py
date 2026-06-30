"""
Smoke test: verify PyLate ColBERT encode + MaxSim rerank works end-to-end.
Outputs: results/smoke_test.json
"""

import json
import time
from pathlib import Path

from pylate import models
from pylate.rank import rerank

OUT = Path("results")
OUT.mkdir(exist_ok=True)

model = models.ColBERT("lightonai/GTE-ModernColBERT-v1")

queries = ["What is BOND in similarity search?"]
docs = [
    "BOND is a distance oracle that prunes candidates using lower-bound estimates.",
    "PDX accelerates high-dimensional similarity search via columnar vector layout.",
    "ColBERT uses late interaction: MaxSim over per-token embeddings.",
    "Inverted file indexes partition the vector space into Voronoi cells.",
    "k-means clustering assigns vectors to the nearest centroid.",
]
doc_ids = list(range(len(docs)))

t0 = time.perf_counter()
q_embs = model.encode(queries, is_query=True)
d_embs = model.encode(docs, is_query=False)
encode_ms = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
ranked_results = rerank(
    documents_ids=[doc_ids],
    queries_embeddings=q_embs,
    documents_embeddings=[d_embs],
)
rerank_ms = (time.perf_counter() - t1) * 1000

result = {
    "query": queries[0],
    "encode_ms": round(encode_ms, 2),
    "rerank_ms": round(rerank_ms, 2),
    "ranking": [
        {"doc": docs[r["id"]], "score": round(float(r["score"]), 4)}
        for r in ranked_results[0]
    ],
}

print(json.dumps(result, indent=2))
(OUT / "smoke_test.json").write_text(json.dumps(result, indent=2))
print(f"\nSaved to {OUT / 'smoke_test.json'}")
