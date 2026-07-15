# Stage 3: Mechanism Experiments

## Current artifact drivers

E01, E02, E03, E08, R12b, and R12c use the frozen shared-driver pattern. Their
command modules select configuration only; preparation, workload identity,
correctness, accounting/timing, provenance, and atomic serialization live
under `bondmaxsim.experiments`. E01/E02 are accounting-only and make no
latency claim. E03/E08/R12b/R12c retain raw counterbalanced observations. Dedicated
`render_*` commands reopen the validated artifact before plotting.

All current drivers provide `--fixture`. Fixture results are diagnostic and
must not be used as performance evidence.

E04--E07 and E09 are retained as historical/diagnostic scripts. Their
stored artifacts remain readable through the registered compatibility
readers; they are not final-evidence producers. R12b remains explicitly
diagnostic, while its migrated driver records raw timing observations; neither
its historical nor migrated probe fields may be promoted over R12c.

Planned experiments for Stage 3 (see `docs/project_b_analysis_and_research_plan.md`,
Stage 3 section, and `docs/stage1_bond_maxsim_formalization.md` §7 empirical context).

Datasets: SciFact, NFCorpus, ArguAna, SCIDOCS (debugging scale); at least one
100k–1M corpus for scale. All drivers import from `bondmaxsim`.

## Planned Experiments

### `e01_bound_slack.py`
Plot UB_d(k) / score(q,d) over dimension prefixes k.  Measures how tight the
Cauchy-Schwarz bound is and how quickly it falls toward the threshold.  This is
the primary "feasibility diagnostic" from the research plan (bound-slack vs score
dispersion). Per-dataset, per dimension-order arm.

### `e02_pruning_rate.py`
Plot the **two-level survival curves** — fraction still live vs dimensions
scanned — at both granularities the Section 2.3 bound operates on, for each
threshold policy (self_bound / oracle / seed) and dimension order:

- **Document survival**: fraction of documents still live (the original "live
  curve"). Measures dims-to-prune-50/90/99%. Identifies whether score compression
  prevents early *document* pruning — the Option-B weakness (Stage 1 §5.2), and
  the thing that makes the per-document oracle look weak on its own.
- **Token survival**: fraction of document tokens still live in the per-document
  live set. This is the lever the **wide kernel** actually relies on — token
  pruning fires from the first block even while the document bound is still loose
  (Stage 1 §5.3). Document survival alone understates the mechanism; measuring
  token survival is what makes the per-document accounting predictive of the wide
  kernel rather than a verdict on Option B.

Also report the **early-token-pruning rate**: fraction of tokens pruned before
`D/4` dimensions (exp-10 skips bound evaluation before `D/4`, so tokens surviving
past there set the warmup cost, Stage 1 §6).

Instrumentation note: both curves need per-dimension-block live counts, not just
final totals. The accounting kernel already maintains the per-document live token
set (`n_live`) and emits a final `tokens_pruned` (stats[2], currently unused by
the Runner). This experiment adds a per-block accounting hook that logs the summed
token-live count and the document-live count at each block boundary, and surfaces
`tokens_pruned` through `ResultRecord`.

### `e03_order_ablation.py`
Compare natural / bond_dtm / bond_q2 / bond_q2_var / pca_rotation on
cells_scanned_pct and ms_per_query.  Confirms that order affects efficiency
but not correctness (shrink=1 exact-agreement regression, Stage 1 §8 item 5).

### `e04_exact_safe_pruning.py`
Full cells/latency sweep for shrink=1 across datasets and candidate-set sizes.
Tests whether more candidates (larger candidate_budget) improve document pruning
(exp-11 probe: downward cells% slope means small samples understated pruning).

### `e05_approximate_recall_sweep.py`
Sweep shrink in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]; report cells_scanned_pct vs
recall_vs_exact@10 frontier per dataset and per dimension order. The approximate
arm is kept strictly separate from the exact arm (Stage 1 §3).

### `e06_threshold_policy_ablation.py`
Compare self_bound / oracle / seed threshold policies on cells_scanned_pct and
pruning rate. Oracle policy isolates maximum pruning potential; seed policy is
the realistic first-class option for wide-block scan (Stage 1 §4.4 / §8 item 6).

### `e07_cache_layout_sensitivity.py`
Measure gather / SIMD penalty from per-query dimension reordering vs natural
order. Quantifies the reorder cost that the research plan flags must be measured,
not assumed (M3 in methodology and Stage 1 §4.5).

## Output
All results to `results/json/stage3_mechanism_<experiment>_<dataset>.json`.
Decision gate: if no exact-safe arm produces repeatable wall-clock win over brute
force, report negative result with mechanism evidence (research plan Stage 3
stopping condition).
