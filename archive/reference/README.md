# Archive Reference Files

On-branch copies of key files from other branches, brought here so that
`final-research-implementation` is self-contained and can be reviewed without
checking out other branches.

## Contents

### `05_maxsim_bond_instrumentation.py`

**Origin:** `Mikel` branch, `experiments/pipeline/05_maxsim_bond_instrumentation.py`.

**Brought on-branch from:** `Mikel` (via `git show Mikel:...`).

**Purpose:** NumPy reference implementation of document-level MaxSim branch-and-
bound with three threshold policies — `bound` (self-contained BOND, k-th largest
running lower bound), `oracle` (true k-th exact score, upper bound on potential),
and `seed` (k-th best exact score among a cheap first-checkpoint partial-score
seed).  This is the threshold-policy reference for Stage 2–3.

**Reimplementation target:** The threshold logic in this file is to be
reimplemented as clean typed functions/classes in
`src/bondmaxsim/threshold/policies.py`.  The Runner testbed in
`src/bondmaxsim/testbed/runner.py` will expose these policies via `RunConfig`.

**Do not import this file directly** — it imports from a `_paths` / `utils_colbert`
shim that is Mikel-branch-local.  Use `src/bondmaxsim/threshold/` instead.

**Stage 1 reference:** `docs/stage1_bond_maxsim_formalization.md` §4.4 (threshold
policy safety and the three regimes), §8 item 6 (seeded threshold is first-class).
