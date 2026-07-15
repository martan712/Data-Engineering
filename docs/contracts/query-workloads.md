# Query-workload contract

Status: Accepted
Version: 1

## Stable IDs

Stages 3–4 use:

```text
beir-<dataset>-mechanism-seed42-n50-v1
```

These are the deterministic seed-42 samples of 50 queries from the mechanism
embedding archives. Stage 5 uses:

```text
beir-<dataset>-qrels-test-v1
```

These contain the complete qrels-evaluable test-query workload. A workload ID
changes when query membership, order, encoding configuration, or ID mapping
changes.

## Required metadata

Every workload record contains dataset, regime, ordered query-ID SHA-256,
embedding/configuration SHA-256, sample size, token-count mean, median, minimum,
maximum, and the complete per-query token-count vector or a checksummed
sidecar. It also records how the population was selected and the source
revision/split. Statistics are computed from offsets after validation, not from
text-length estimates.

Stage 3–4 mechanism results and Stage 5 IR results are not merged unless their
workload IDs match. Cross-stage discussion explicitly notes that NFCorpus's
Stage 5 queries are materially shorter (observed means 8.56 versus 12.20),
while SciFact changes identities without a material length shift and ArguAna
and SciDocs remain similar. ArguAna's 48-token workload is stable in both
regimes.
