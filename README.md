# BOND-MaxSim: Dimension Pruning for ColBERT MaxSim on a PDX Layout

Research project (Data Engineering, Project B). **Question:** can a
PDX-compatible implementation of BOND-style dimension pruning accelerate ColBERT
MaxSim retrieval at matched retrieval quality?

ColBERT scores a query and document by late interaction,
`score(q, d) = sum_i max_j <q_i, d_j>`. Exact MaxSim is expensive. We adapt the
SIGMOD-2002 BOND branch-and-bound over embedding dimensions to MaxSim, implement
it on the PDX columnar layout, and evaluate it against exact MaxSim, IVF + exact
rerank, and PLAID using BEIR-style datasets and the CoRECT framework.

## Layout

See **`docs/project_structure.md`** for the canonical map and conventions. In short:

- `docs/` — the written project: plan + Stage 0–5 artifacts.
- `extern/` — pinned dependencies as submodules: PDX, PDX-sigmod, CoRECT.
- `src/bondmaxsim/` — the research Python package (split by responsibility).
- `cpp/` — our C++ contribution (exact-safe oracle kernels + the wide-block MaxSim BOND).
- `experiments/` — thin reproducible drivers, one folder per stage.
- `results/` — tracked JSON results, figures, and preserved external baselines.
- `archive/` — exploratory history (non-decisive).

## Status

Stages 0 (references) and 1 (BOND-MaxSim formalization + exact-safe proof) are
complete in `docs/`. Stage 2 (mechanism testbed) is under construction. See the
status table in `docs/project_structure.md`.

## Getting Started

The project uses [uv](https://docs.astral.sh/uv/) as its standard Python
toolchain (venv + dependency management). `./setup.sh` wraps the full bootstrap:

```bash
./setup.sh        # init submodules, create .venv (uv), install bondmaxsim, build C++ kernels
uv run pytest     # Stage 2 blocking checks: unit-norm guard + shrink=1 exact-agreement
```

Or run the steps manually:

```bash
uv venv --python 3.12          # create .venv
uv pip install -e ".[dev]"     # editable install (NumPy-only core + pytest)
make -C cpp/per_document_oracle # build the per-document-oracle kernel
uv run pytest                  # run the gate
```

The core install is NumPy-only (the mechanism testbed). The heavy retrieval
stack (torch/pylate/ranx, Stages 3–5) is an optional extra:
`uv pip install -e ".[dev,retrieval,faiss]"`.

## Key Results (Stage 1)

The `shrink = 1` BOND-MaxSim kernel is **exact-safe by proof** (unit-normalized
tokens, exact arithmetic, set-equality top-k); the token-pruning survival
invariant is proven. `shrink < 1` is approximate with no recall guarantee and is
reported only on a quality–work frontier. The contribution is the **multi-vector
MaxSim extension** of PDX's existing single-vector BOND. See
`docs/stage1_bond_maxsim_formalization.md`.
