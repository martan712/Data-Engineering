# Provisional paper-methodology corrections

Status: draft for integration after the final artifact rerun. Numeric claims and
dataset-specific conclusions remain provisional. The paragraphs below define
terminology and methodology only; the evidence manifest must supply any final
numbers, tables, and figures.

## BOND 2002 and the final MaxSim method

Suggested replacement for the Background and contribution wording:

> BOND 2002 eliminates individual candidate vectors while scanning their
> dimensions incrementally. Our final method borrows that elimination idea but
> applies it to a complete multi-vector MaxSim document. A fused panel kernel
> accumulates all query-token/document-token pairs and terminates a document
> when a MaxSim upper bound cannot reach the current top-k threshold. We call
> this BOND-style document-level early termination, not an implementation of the
> original single-vector BOND algorithm. The local panel layout is
> PDX-compatible/PDX-inspired; “direct PDX integration” is reserved for arms
> that execute the pinned PDX library.

This distinction belongs in the abstract, Background, kernel description, and
contribution list. “BOND” alone should be qualified whenever the original
algorithm and the final MaxSim kernel could be confused.

## Wide token pruning and final document pruning

Suggested replacement for the mechanism discussion and limitations:

> The wide accounting kernel is a mechanism instrument. It scans breadth-first
> across token blocks, compacts live token survivors, and exposes detailed
> document/token survival curves. That organization is closer to the original
> live-candidate view of BOND, but it is not the production path: domination
> bookkeeping, synchronized threshold dynamics, late pruning, and measured
> wall-clock cost made it unsuitable as the final throughput kernel. The final
> fused kernel instead performs document-level early termination in a
> MaxSim-specific register-tiled scan. Fused token pruning was not exhaustively
> retested at every final late checkpoint on every dataset, so the paper must
> not generalize the wide-kernel token-survival result into a claim about all
> final fused schedules.

## Two query regimes

Suggested replacement for the Data and Experiments setup:

> The project uses two deliberately distinct query workloads. Stages 3--4 use
> a stable seed-42 mechanism sample from the archived encoded-query population;
> these experiments study bounds, pruning, and systems behavior and do not
> require qrels. Stage 5 uses the full encoded qrels-test query set for standard
> IR evaluation. Results record a stable workload ID, ordered-query-ID hash,
> selection rule, and token-length distribution. Comparisons are valid only
> within a workload ID; the paper reports query-length summaries for both
> regimes because MaxSim cost and pruning depend directly on query-token count.

The final generated dataset table should contain the count and token-length
summary for each dataset/regime instead of one project-wide average.

## Timing estimator and counterbalancing

Suggested replacement for the Measurement Protocol paragraph:

> Setup, corpus validation, packing, index construction, oracle construction,
> result validation, and serialization occur outside measured scopes. After
> explicit warm-up rounds, every timing session retains all raw observations.
> Arm order is cyclically counterbalanced across rounds, so each arm occupies
> every position before the cycle repeats. Reported arm summaries use the
> predeclared robust estimator over retained observations, while comparisons
> with the dense baseline use same-round paired margins. Independently selected
> minima are neither reported nor added across components. Separate sessions
> are required for single-threaded and all-core modes.

The exact preset, estimator, session ID, environment, and observation rows are
serialized and must be cited by the final timing tables.

## Candidate caps and actual work

Suggested replacement for Fixed Candidate Budgets:

> A configured candidate or full-score cap is a control, not proof of work
> performed. Candidate-producing arms therefore record, per query and where
> observable, unique candidates generated, documents admitted to scoring,
> documents fully scored, probed documents/partitions, and inspected token
> hits. Strict equal-work reranking comparisons require an exact observable
> actual-work field. Complete systems whose backend exposes only a configured
> cap are labeled system-cap comparisons and are interpreted jointly by
> latency, retrieval quality, and available work counters. In particular,
> PLAID's configured `n_full_scores` is not described as an exact actual count
> when the pinned public API cannot expose that count.

## Exactness and CoRECT scope

Suggested replacement for the correctness and evaluation wording:

> The pruning proof is an exact-arithmetic result under unit-normalized token
> embeddings. Implementation claims are empirical gates in fp32: returned IDs
> must be structurally valid and must either match the oracle top-k ID set or
> differ only by independently scored documents verified at the k-th-score tie
> boundary. Recall against one oracle tie-breaking choice is reported
> separately and is not an exactness certificate. Stage 5 computes ordinary
> qrels metrics and cross-validates them through CoRECT's standard evaluator.
> It does not construct the controlled relevant/distractor/random pools needed
> for CoRECT Relevance Composition (RC), and therefore makes no RC claim.

The final terminology audit must remove unconditional “bit-identical,” plain
set-equality, and “CoRECT RC” wording unless a cited final artifact directly
supports it.

## Evidence lifecycle

Existing result files and paper assets are provisional inputs to renderer and
catalog development. A numeric claim becomes current only when the artifact
catalog classifies its source as current, the evidence manifest selects exact
validated JSON fields, and the generated paper output is fresh. Historical,
diagnostic, invalid, or field-partially-superseded artifacts remain readable
for provenance but cannot be selected silently as headline evidence. In
particular, historical standalone e08/e09 latency fields are superseded by the
counterbalanced R12c timing evidence while independently valid accounting
fields may remain eligible.

## Integration checklist

- Abstract and contribution list: distinguish BOND 2002 from the final method.
- Background and kernels: distinguish the wide instrument, fused document
  kernel, PDX-inspired layout, and direct pinned-PDX arms.
- Methodology: insert the two-workload, timing, work-accounting, exactness, and
  CoRECT paragraphs above.
- Results: replace manually selected numeric text with generated macros/tables;
  carry evidence qualifications into captions.
- Limitations: retain the fused-token coverage limitation and the absence of a
  Relevance Composition evaluation.
- Release audit: require current catalog/evidence entries before removing this
  document's provisional status.
