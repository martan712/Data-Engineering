# Result Artifacts

This directory separates historical exploratory evidence from final controlled
measurements.

- `legacy/` preserves compact JSON files produced before the controlled
  benchmark protocol was adopted. They are useful for correctness, candidate
  coverage, and mechanism analysis. Their wall-clock values are not final
  cross-method comparisons.
- `controlled/` is reserved for results that satisfy
  `docs/benchmark_protocol.md`. Final tables and speedup claims must use only
  this directory.

Large embeddings and indexes remain outside Git. Controlled result JSON files
must contain immutable input identifiers or SHA-256 hashes so they can be tied
back to the exact inputs used.

