# Benchmark machine preparation

Use this checklist before creating the Stage 7 release-candidate record and
before every accepted Stage 8 timing session. The release tooling records the
declarations and observable host state; it does not mutate power, process, or
CPU settings automatically.

## Required preparation

1. Connect AC power and confirm the battery is not the active power source.
2. Record the CPU frequency governor. Use the same governor for every accepted
   session; do not mix governor states in one comparison.
3. Close unrelated interactive applications, scheduled compute jobs, IDE
   indexing, backup/synchronization clients, and package or system updates.
4. Stop other repository processes. In particular, do not run encoding,
   indexing, compilation, plotting, tests, or another benchmark concurrently.
5. Let the machine reach a stable idle temperature before the first session.
6. Use the paper-native binaries recorded by the release candidate. Do not
   rebuild or edit code between accepted sessions.
7. Set and record the declared thread environment and CPU affinity. Keep it
   identical for all arms within a comparison session.
8. Confirm the production data freeze and release-candidate validation commands
   pass immediately before timing.

## Operator declarations

The release-candidate record requires explicit confirmation of all of the
following:

- `unrelated_applications_closed`
- `background_jobs_stopped`
- `machine_quiescent`
- `power_and_governor_recorded`
- `thermal_state_stable`

A missing or false declaration prevents production authorization. These are
operator statements, not values inferred from an unreliable platform API. The
release candidate separately captures observable AC-power state, governor,
thread environment, and CPU affinity from the host.

## Session discipline

- Start each required fresh session in a new process.
- Preserve counterbalanced arm order, equal warm-ups, and equal repetitions.
- Preserve disturbed observations; never delete outliers after viewing them.
- If a round fails partway through, discard that incomplete round and rerun the
  whole comparison. Do not splice arms from different rounds.
- Write results atomically after each complete session.
- Record any disturbance or protocol deviation in the session result.

## Abort and invalidate

Stop the active session and do not accept its timing results if any of these
conditions occurs:

- power source or governor changes;
- thermal throttling or a material temperature excursion is observed;
- an unrelated resource-heavy process starts;
- thread, affinity, binary, code, data, index, or environment hashes drift;
- a correctness, schema, candidate-accounting, or workload check fails;
- an arm or repetition is missing.

Any relevant code, data, index, toolchain, or final-run-manifest change returns
the project to the Stage 7 freeze procedure before further timings can be
accepted.
