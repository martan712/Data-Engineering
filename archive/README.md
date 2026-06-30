# Archive — Non-Decisive Exploratory History

This directory holds exploratory history kept for provenance only.
**No final research claims cite anything in `archive/`.**

Final claims cite `src/`, `cpp/`, `experiments/`, `results/`, and `docs/`
exclusively.  See `docs/project_structure.md` Convention 8 ("Archive is
non-decisive").

## Contents

### `preliminaries/`

Was `research/preliminaries/` — experiments 01–11 from the `martan's-experiment`
workstream.  Preliminary exploration of BOND-style MaxSim pruning with custom
C/C++ kernels.  Moved here by `git mv` on branch `final-research-implementation`.

Reusable code from these experiments has been factored into `src/bondmaxsim/`
(oracle, ordering, threshold, kernels) and ported to `cpp/per_document_oracle/`.
The `.so` build artifacts that were accidentally tracked have been removed from
the index (see `.gitignore` policy in `docs/project_structure.md`).

### `colbert_scripts/`

Was `research/colbert/` — ColBERT pipeline scripts from the `Mikel` workstream
including exact MaxSim oracle, PLAID and FAISS-IVF wrappers, and BEIR evaluation
utilities.  Moved here by `git mv`.

Reusable logic has been factored into `src/bondmaxsim/oracle/`, `baselines/`,
and `eval/`.

### `reference/`

On-branch copies of key reference files from other branches, brought here so
this branch is self-contained.  See `archive/reference/README.md`.
