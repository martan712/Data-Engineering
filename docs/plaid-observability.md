# PLAID compatibility and observability decision

Status: Accepted  
Pinned PyLate version: 1.6.0

PyLate 1.6.0 exposes `n_ivf_probe` and `n_full_scores` as public `PLAID`
constructor parameters. It does not expose a public search-time mutator, so the
adapter applies changed settings by reattaching the existing on-disk index with
the public constructor outside the timed search scope. No private `_index`
access is used.

The public result contains only ranked document IDs and scores. It does not
contain the actual number of candidate documents generated, admitted, or fully
scored for an individual query. Those actual counts therefore have
`value: null` and `quality: unavailable`; the configured `n_full_scores` value
is retained separately as an exact configured full-score cap.

PLAID comparisons are consequently labeled `system_cap`. They must report
latency, recall, and the configured cap together and must not use equal-work
wording.
