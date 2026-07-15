# Timing contract

Status: Accepted
Version: 1

## Protocol classes

| Class | Warm-up rounds | Measured rounds | Intended use |
|---|---:|---:|---|
| `fixture` | 1 | 3 | deterministic smoke and integration tests |
| `mechanism` | 2 | 9 | Stage 2/3 kernel and accounting comparisons |
| `full_system` | 1 | 5 | expensive Stage 4/5 complete-workload runs |
| `confirmation` | 1 | 9 | single-digit headline margins |

Every compared arm receives the same policy. A claimed margin below 10% needs
at least three fresh process/session IDs using `confirmation`, unless it is
reported as a tie. An accepted win needs a positive median paired margin in
every session and a pooled median paired margin greater than 3%. Differences
in `[-3%, +3%]` are the equivalence band and are reported as ties.

## Ordering and observations

For arms `[a0, ..., an-1]`, round `r` uses the cyclic order beginning at
`(session_offset + r) mod n`. `session_offset` is derived deterministically
from the stable session ID. For a multiple of `n` rounds, every arm occupies
every position equally often. The exact sequence is serialized.

Every observation retains:

```text
session_id, round_index, phase, arm_id, arm_position,
started_at_utc, elapsed_ns, timer_scope_id, completed
```

Warm-ups use `phase = "warmup"`; they are retained but excluded from summaries.
Incomplete rounds remain diagnostic and cannot be serialized as a complete
comparison.

## Timer boundary

The measured scope includes the operation represented by the arm: for a kernel
arm, scoring the already validated/packed workload; for a system arm, query
candidate generation plus all scoring included in the system claim. It excludes
process setup, input loading, corpus validation, packing/index construction,
warm-up, result validation, quality metrics, serialization, rendering, and
plotting. Setup and validation events are recorded with explicit scope names so
their exclusion can be tested.

## Summaries and paired comparison

Latency is stored in integer nanoseconds and presented in milliseconds. Each arm
reports all measured values plus `best_observed`, median, minimum, maximum, and
IQR. Best observed is the minimum warmed measured observation and is never
called typical latency. Quartiles use the linear percentile definition at 25%
and 75%; IQR is `q75 - q25`.

For candidate arm `c` and baseline `b` in the same session/round:

```text
paired_ratio = elapsed_c / elapsed_b
paired_margin_pct = 100 * (elapsed_b - elapsed_c) / elapsed_b
```

Positive margin means the candidate is faster. The median paired per-round
margin is the primary comparison; independently selected best-observed values
are secondary. Raw disturbed observations are retained.

Each session records thread settings, affinity if available, AC/battery and
power/governor state if readable, process ID, host/CPU, software/toolchain, code
and binary hashes, and whether the operator declared the machine quiescent.
