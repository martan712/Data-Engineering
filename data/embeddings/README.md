# data/embeddings/

Pre-converted ColBERT token embeddings in the BOND-MaxSim token-major format.

## Provenance

Converted from the archive ColBERT cache at
`archive/preliminaries/02_bond_variance/.cache/`.  The caches were produced by
PyLate (a ColBERT-compatible library) using the `GTE-ModernColBERT-v1` model.
Embedding dimension: **D = 128**.  All token embeddings are **L2-unit-normalised**
(verified by `bondmaxsim.oracle.normalization.check_unit_norm` during porting).

## Datasets

| file             | source archive name | num_docs | num_queries |
|------------------|---------------------|----------|-------------|
| scifact.npz      | SciFact             | 5 183    | 200         |
| nfcorpus.npz     | NFCorpus            | 3 633    | 200         |
| arguana.npz      | ArguAna             | 8 674    | 200         |
| scidocs.npz      | SCIDOCS             | 25 657   | 200         |

## NPZ Schema

Each `.npz` file contains four arrays:

| key            | dtype   | shape            | description                                      |
|----------------|---------|------------------|--------------------------------------------------|
| `doc_values`   | float32 | [T, 128]         | all document tokens concatenated (token-major)   |
| `doc_starts`   | int64   | [num_docs]       | start token index of each document in doc_values |
| `query_values` | float32 | [Tq, 128]        | all query tokens concatenated                    |
| `query_starts` | int64   | [num_queries]    | start token index of each query in query_values  |

Document `d` spans `doc_values[doc_starts[d] : doc_starts[d+1]]`
(last document runs to `T`).  Same convention for queries.

## Stage 5 sidecars

The `.npz` archives store embeddings only.  Stage 5 (qrels metrics) needs the
BEIR string IDs and test qrels:

- `data/beir_ids/<ds>_ids.json` — row-order corpus/query `_id` lists from
  HuggingFace `BeIR/<ds>` (the source and order the archive cache was encoded
  in; row order verified empirically by
  `bondmaxsim.data.beir_ids.verify_row_order`, 2026-07-09, cosine 1.0).
- `data/qrels/<ds>.tsv` — test-split qrels
  (`query-id \t corpus-id \t score`, with header).
- `<ds>_test_queries.npz` (scifact, nfcorpus only) — the main archives encode
  the FIRST 200 rows of the queries split, which for scifact/nfcorpus are all
  train queries with **zero** test-qrels overlap.  These extra archives hold
  the encoded test-split queries (`query_values`, `query_starts`, `query_ids`)
  produced by `bondmaxsim.data.encode_test_queries` with the same model and
  settings.  arguana/scidocs need none: their 200 encoded queries are all in
  the test qrels.

## Regeneration

The `.npz` blobs and sidecars are not committed (see `.gitignore`).
Regenerate them with:

```bash
uv run python -m bondmaxsim.data.port_embeddings      # <ds>.npz
uv run python -m bondmaxsim.data.beir_ids             # ids + qrels sidecars
uv run python -m bondmaxsim.data.encode_test_queries  # <ds>_test_queries.npz
```
