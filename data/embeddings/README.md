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

## Regeneration

The `.npz` blobs are not committed (see `.gitignore`).  Regenerate them with:

```bash
.venv/bin/python -m bondmaxsim.data.port_embeddings
```
