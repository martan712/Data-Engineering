# Data and reproducibility contract

Status: Accepted
Version: 1

## Supported environment

The artifact support target is Linux x86-64. The locked interpreter is
CPython 3.12.13 and the lock is generated/synchronized with uv 0.11.7. `uv.lock` covers
core, dev, retrieval, FAISS, rendering, and test dependencies and is the paper
environment; package lower bounds are not a reproducibility claim.

`portable` native builds use the documented x86-64 baseline without
`-march=native`; `paper-native` uses `-march=native` and is the only profile
eligible for paper latency on the recorded host; `sanitize` uses debug symbols,
ASan, UBSan, and frame pointers. All profiles record compiler path/version,
flags, detected ISA, OpenMP, and binary SHA-256.

## Public sources and configuration

The full configuration covers `scifact`, `nfcorpus`, `arguana`, and `scidocs`.
It identifies immutable revisions—not branches or unpinned dataset names—for:

- `BeIR/<dataset>` corpus and queries splits;
- `BeIR/<dataset>-qrels` test data, or the explicitly selected
  `mteb/<dataset>` fallback;
- `lightonai/GTE-ModernColBERT-v1` model and tokenizer.

Unknown original revisions/defaults are recorded as unknown during recovery;
they are not guessed. Stage 6 cannot freeze or benchmark until every final
source revision and effective setting is concrete.

Document construction is `row["text"]` only; titles are excluded to preserve
the original scientific workload. Query and document ordering follow source
row order except the two explicitly defined workload selectors. Configuration
must state tokenizer/model revisions, special-token behavior, maximum lengths,
padding, truncation, normalization, `is_query`, batch size, precision, device,
output dtype, and every encoding argument rather than relying on library
defaults.

## Outputs and validation

Generation is atomic and resumable per dataset/shard. It produces embedding
NPZ files, ordered ID sidecars, qrels, query-regime files, index manifests, and
a data manifest containing configuration/source hashes, counts, token-length
statistics, environment, and SHA-256 values. Generated embeddings and indexes
remain ignored and are not published.

Before use, validation checks shapes/dtypes, finite values, monotone terminal
offsets, complete token coverage, exact ordered IDs, qrels overlap, norms over
all tokens, and checksums. FAISS/PLAID index manifests record source embedding
hash, parameters, seeds, dependency versions, and output checksums. Results
refuse mismatched data or index hashes.

## Reproduction levels and fixture

- Artifact verification validates tracked evidence, regenerates publication
  assets, runs gates, and compiles the paper.
- Functional reproduction generates a deterministic small fixture and runs
  representative mechanism/system paths on a supported CPU.
- Full performance replication regenerates all public data/indexes and reruns
  the final serial suite on comparable recorded hardware.

The offline fixture uses a fixed seed and deterministic synthetic normalized
token vectors plus IDs/qrels. Its manifest freezes counts, hashes, and expected
exactness. A separate optional public-download fixture may exercise encoding;
CI must not depend on an unpublished cache.

The original and regenerated data are considered equivalent only on exact
ordered IDs, token counts/shapes/dtypes, and bitwise/checksum-equal embeddings.
If not bitwise equal, every data-dependent paper result is rerun; similar
rankings or metrics do not weaken this rule.
