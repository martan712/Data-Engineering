# Shared experiment timing

Experiments declare `ArmSpec` objects and an `ExperimentSpec`; drivers do not
own timing loops. `TimingProtocol` implements the accepted fixture, mechanism,
full-system, and confirmation round counts. A stable fresh session ID selects
the initial arm offset, and every following round rotates cyclically. Raw
warm-up and measured observations retain session, round, phase, arm position,
UTC start, integer nanoseconds, timer scope, completion, and any error.

The measured context contains only the declared arm operation. Setup and
post-operation correctness validation have explicit unmeasured scope events.
Attempting to enter an excluded scope while a measured scope is active raises
`TimingError`. A failed operation or validator leaves a diagnostic observation,
marks the session incomplete, and prevents summaries or paired comparisons
from being emitted as complete evidence.

Complete sessions retain every measured value and calculate best observed,
median, min/max, linearly interpolated quartiles, and IQR. Paired comparisons
join candidate and baseline within the same session and round; positive margin
means the candidate was faster. Independently selected minima are never used
as the primary paired margin.

Run the fake-clock specification tests with:

```sh
UV_CACHE_DIR=/tmp/bondmaxsim-uv-cache uv run pytest -q tests/test_experiment_timing.py
```
