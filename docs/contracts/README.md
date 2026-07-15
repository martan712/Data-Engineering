# Finalization contracts

Status: Accepted
Version: 1
Accepted: 2026-07-15

These records freeze the meanings and public names used by the finalization
work. Implementations may add fields compatibly, but changing a defined
meaning requires a new contract version and a migration. Historical JSON is
never edited in place.

The seven accepted records are:

1. [Agreement](agreement.md)
2. [Timing](timing.md)
3. [Candidate work](candidate-work.md)
4. [Query workloads](query-workloads.md)
5. [Results and artifacts](results-and-artifacts.md)
6. [Data and reproducibility](data-and-reproducibility.md)
7. [Terminology](terminology.md)

## Implementation layout

```text
src/bondmaxsim/
  experiments/
    arms.py
    timing.py
    environment.py
    execution.py
    io.py
    workloads.py
  results/
    models.py
    validation.py
    migrations.py
    catalog.py
    evidence.py
  data/
    config.py
    generate.py
    fixture.py
    manifest.py
artifacts/
  schemas/
  catalog.json
  paper_evidence.json
  validation/
report/generated/
```

JSON is the interchange format for the catalog and evidence manifest. This
keeps validation in the Python standard-library path; it deliberately replaces
the provisional `.yaml` names in the implementation plan without changing
their responsibilities. RFC 6901 JSON Pointer is the only field-selection
syntax used by evidence entries.

## Public compatibility policy

- New outputs use `bondmaxsim.result-envelope` version `1.0.0`.
- Historical files remain byte-for-byte unchanged and are read through named
  migrations.
- A materialized normalized historical record receives a new artifact ID and
  records `derived_from` plus the migration name.
- Unknown historical shapes fail validation. Adding one requires a fixture,
  named reader, and schema-registry entry.
- A renamed field may have a read-only compatibility alias at the migration
  boundary; current code and generated artifacts use only the accepted name.
