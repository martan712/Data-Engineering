# Stage 4 integration experiments

These drivers use the shared declarative experiment framework. Dataset loading,
corpus validation, exact-oracle construction, and index building happen in the
excluded `experiment-setup` scope. `TimingProtocol` owns warmups,
counterbalanced measured rounds, validation, summaries, environment capture,
and atomic envelope serialization. Drivers never select a best observation.

Fixture runs are diagnostic only. They exercise native execution, raw timing
observations, metadata, validation, atomic IO, and artifact-driven rendering;
they are not performance evidence.

## e01 fixed candidate arms

e01 writes two distinct artifacts:

- `equal_work_reranking`: FAISS candidate generation plus exact reranking. Every
  query retains exact generated, admitted, and fully-scored document counts.
- `system_cap`: PLAID under configured `n_full_scores` caps. The pinned public
  API does not expose actual fully-scored counts, so those counts remain
  `unavailable` and this output cannot make equal-work claims.

```bash
uv run python -m experiments.stage4_integration.e01_fixed_candidate_arms --fixture
uv run python -m experiments.stage4_integration.e01_fixed_candidate_arms --dataset scifact --threads 1
```

## e02 seeded tau recovery

e02 retains separate direct observations for seed-only and kernel-only arms,
plus a directly timed realizable end-to-end seed-and-kernel arm. It never sums
independently selected component minima. Self-bound, partial seed, and oracle
kernel controls retain the frozen tight/natural/C={112}/shrink=1 methodology.

```bash
uv run python -m experiments.stage4_integration.e02_seeded_tau_recovery --fixture
uv run python -m experiments.stage4_integration.e02_seeded_tau_recovery --dataset scifact --threads 1
```

## e03 partition frontier

e03 compares brute and BOND scanners across a partition-probe frontier. Each
query retains exact documents-probed and partitions-probed counts from the
selected partition offsets and probe-list length. Partial probes are
approximate; the full-probe control is exact-safe.

```bash
uv run python -m experiments.stage4_integration.e03_partitioned_fused_scan --fixture
uv run python -m experiments.stage4_integration.e03_partitioned_fused_scan --dataset scifact --threads 1
```

All result names are deterministic and include experiment, dataset, and
workload IDs. Figures are produced separately by the renderers in
`bondmaxsim.render.stage4`, which reopen and validate saved envelopes.
