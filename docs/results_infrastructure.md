# Versioned result infrastructure

All new experiment evidence uses `bondmaxsim.result-envelope` version `1.0.0`.
The envelope records stable experiment, dataset, workload, method, protocol,
environment, and provenance identities around one semantically validated
payload. Supported payloads cover timing comparisons, pruning/accounting,
candidate frontiers, IR quality, paired significance, and validation audits.

`bondmaxsim.results.load_result` reads current envelopes and every frozen
historical shape in `artifacts/baseline/fixtures/`. Historical JSON is returned
as a normalized in-memory envelope carrying the source checksum, source path,
historical schema name, and migration name; source files are never rewritten.

Serialization uses sorted UTF-8 JSON, two-space indentation, one trailing
newline, a sibling temporary file, `fsync`, and atomic replacement. Repeated
timing sessions merge only when schema major version, workload, dataset,
payload, method configuration, protocol, thread mode, and data/index hashes
match. An explicit override creates a derived artifact and records every
incompatibility.

The catalog files use JSON syntax inside `.yaml` files. JSON is a YAML subset,
which keeps the governance files dependency-free in the core environment.
Catalog validation checks IDs, paths, status vocabulary, supersession scopes,
successor relationships, cycles, and evidence references.

Run the focused infrastructure tests with:

```sh
UV_CACHE_DIR=/tmp/bondmaxsim-uv-cache uv run pytest -q tests/test_result_infrastructure.py
```
