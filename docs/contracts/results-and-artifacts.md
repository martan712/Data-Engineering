# Results and artifact contract

Status: Accepted
Version: 1

## Result envelope

Every new result uses schema name `bondmaxsim.result-envelope` and semantic
version `1.0.0` with these top-level fields:

```text
schema_name, schema_version, artifact_id, experiment_id, created_at_utc,
dataset_id, workload_id, method_configuration, protocol, environment,
provenance, payload_kind, payload
```

`artifact_id` is a lowercase stable identifier matching
`[a-z0-9][a-z0-9._-]*`; it identifies scientific content and does not encode a
mutable status. `experiment_id`, dataset/workload IDs, method configuration,
and protocol are mandatory even when repeated by the payload. Provenance
contains Git revision and dirty state, lockfile/config/data/index/native-binary
SHA-256 values where applicable, producing command, and input artifact IDs.

Accepted payload kinds are:

```text
timing_comparison
pruning_accounting
candidate_frontier
ir_quality
paired_significance
validation_audit
```

Each kind has a typed schema and semantic validator. Metrics must be finite and
within their declared domains; repetitions, query counts, work counts, and
QPS/latency derivations must be internally consistent. Merges require matching
schema major version, workload/data/index hashes, thread mode, method
configuration, and protocol. Overrides create a new derived artifact and record
the incompatibility; they never silently coerce inputs.

Serialization is deterministic UTF-8 JSON with sorted keys, two-space indent,
one trailing newline, and atomic replace from a sibling temporary file.

## Registry and migration

The schema registry names every current schema and every historical reader.
Historical shapes are identified by stable structural predicates and the
fixtures under `artifacts/baseline/fixtures/`. A compatibility reader returns a
normalized in-memory envelope while retaining source path/checksum, historical
schema name, and migration name. Historical source JSON is never overwritten.

## Catalog status and supersession

The catalog status vocabulary is exactly:

```text
current | superseded | historical | diagnostic | invalid
```

Catalog entries contain artifact ID, path, checksum, schema, producing command,
inputs, workload/method/thread identities, outputs/consumers, and regeneration
cost class. Supersession edges are acyclic and include reason, date, successor,
and scope. Whole-artifact scope is `/`; partial scope is an array of RFC 6901
JSON Pointers such as `/payload/timing/median_ms`. A current paper claim cannot
select a superseded or invalid pointer.

The known standalone e08/e09 wall-clock fields are partially superseded by the
R12c interleaved evidence; their mechanism/accounting fields may remain
diagnostic or current after validation.

## Paper evidence

Every table, figure, generated macro, and material numeric claim has an evidence
ID with paper location, artifact IDs, exact JSON Pointers, filters/grouping,
named derivation, units/rounding, renderer command, generated output, status,
and qualifications. Derivations are named tested functions, not prose-only
formulas. Publication assets live under `report/generated/`; regeneration must
be deterministic and stale output fails the artifact gate.
