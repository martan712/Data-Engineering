# Candidate-work contract

Status: Accepted
Version: 1

Candidate controls and observed work are never represented by one overloaded
`budget` field. Every per-query work field is a count observation:

```text
value: non-negative integer or null
quality: exact | estimated | upper_bound | unavailable
source: stable instrumentation description
```

The public fields are:

```text
configured_candidate_cap
configured_full_score_cap
unique_candidates_generated
documents_admitted_to_scoring
documents_fully_scored
documents_probed
partitions_probed
token_hits_inspected
```

Configured caps may be null and do not imply any actual count. Per-query
observations are retained; summaries contain count, min, median, mean, max, and
the number unavailable.

Every comparison declares one scope:

- `system_cap`: complete systems use the same configured cap, but actual work
  may differ. Recall, latency, and actual work are interpreted together.
- `equal_work_reranking`: every compared scorer receives the same explicitly
  identified candidate ID list and exact candidate count.
- `probe_frontier`: partition/list probing is plotted against exact observed
  documents and partitions probed, without claiming equal candidate work.
- `uncapped_reference`: exhaustive dense/exact-safe references.

FAISS and PDX candidate output length is recorded as exact generated/admitted
work; `documents_fully_scored` is exact only when the downstream reranker is
known to score that list once. Partitioned scan records exact partition and
document counts from offsets. PLAID records exact full-score work only if the
pinned implementation exposes and tests it; otherwise the value is null with
`quality = "unavailable"`, PLAID remains `system_cap`, and equal-work wording is
prohibited.
