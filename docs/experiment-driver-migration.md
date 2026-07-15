# Experiment driver migration pattern

Status: Frozen after the Stage 4 serial e08 pilot.

The migrated `e08_checkpoint_ablation` is the reference shape for lanes
4A--4C. A driver selects a dataset, thread mode, fixture/full configuration,
session ID, and output directory. It does not implement timing, correctness,
summaries, serialization, or plotting.

The shared preparation callback performs data validation, query selection,
workload hashing, corpus/index construction, exact-oracle construction, and arm
preparation inside the excluded `experiment-setup` scope. Native arms expose a
single workload pass; they must not contain warmups, repetitions, clocks, or
best-of selection. `TimingProtocol` owns warmups and counterbalanced measured
rounds. Validators and metadata extractors run after each timed operation in
explicit excluded scopes.

Every migrated driver follows this flow:

1. Build typed workload metadata and provenance hashes.
2. Declare `ArmSpec` objects with stable IDs, all method parameters,
   exactness, comparison scope, validator, and metadata extractor.
3. Run through `execute_prepared_timing_experiment`.
4. Atomically write the validated versioned envelope using its deterministic
   experiment/dataset/workload name.
5. Render only by reopening the saved artifact through `load_result`.

Candidate arms retain a per-query `CandidateWorkObservation`. Mechanism arms
retain per-query pruning/accounting rows. PLAID remains `system_cap` because
its actual full-score work is unavailable. Exact-safe arms use the shared
agreement result and fail the session if strict or verified boundary-tie
equivalence does not hold.

Fixture mode is required and is diagnostic only. It exercises native
execution, raw timing observations, validation, metadata, atomic IO, and
artifact-driven rendering without producing performance evidence. The pilot
command is:

```bash
uv run python -m experiments.stage3_mechanism.e08_checkpoint_ablation --fixture
```

The full SciFact configuration is checked against the frozen historical e08
methodology before execution. Only scientific controls are compared; legacy
standalone/best-of timing values are not carried forward as new evidence.
