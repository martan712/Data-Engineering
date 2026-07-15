# Terminology contract

Status: Accepted
Version: 1

Use these terms in current code, schemas, documentation, and the paper:

- **BOND-style document-level early termination over embedding dimensions**:
  the final method. It lifts BOND's elimination idea to a complete MaxSim
  document and avoids remaining dimensions of all its tokens.
- **BOND 2002 vector elimination**: the original dimension-incremental scan of
  live candidate vectors. “Dimension pruning” describes avoided remaining
  work; dimensions are not globally deleted.
- **wide token-pruning mechanism**: the breadth-first survivor-compaction
  instrument closer to original BOND. It is not the final production kernel.
- **final fused document kernel**: the MaxSim-specific document-pruning design.
- **configured candidate cap** or **configured full-score cap**: a control that
  may be underfilled. Use **actual work** for observed generated, admitted,
  scored, probed, partition, and token-hit counts.
- **strict top-k ID-set equality**: identical unique ID sets.
- **verified boundary-tie equivalence**: a differing set validated with
  independent exact scores at the k-th boundary.
- **recall versus one oracle tie-breaking choice**: set recall; never a synonym
  for strict or tie-verified exactness.
- **standard IR evaluation with CoRECT-backed metric cross-validation**: Stage
  5's use of CoRECT `evaluate_results` for ordinary qrels metrics.
- **Relevance Composition (RC)**: the separate CoRECT evaluation requiring
  controlled relevant/distractor/random pools. This project does not compute
  RC.
- **PDX-inspired** or **PDX-compatible layout**: local code borrowing layout or
  algorithm ideas without calling the PDX implementation. **Direct PDX
  integration** is reserved for an adapter actually executing pinned PDX code.

Current APIs and fields use `compute_corect_standard_metrics`,
`corect_metric_crosscheck`, and `corect_standard_metrics`. The historical names
`compute_rc_metrics`, `corect_smoke_test`, and `CoRECT_RC_metrics` may appear
only in compatibility readers, deprecation aliases, or explicitly historical
artifacts.

Claims about the wide token mechanism state that it was rejected for the
production path due to domination bookkeeping, late pruning, weak synchronized
threshold dynamics, and measured wall-clock cost. They also state that fused
token pruning was not exhaustively retested at every final late checkpoint
(`{96}` and `{112}`) across all datasets.
