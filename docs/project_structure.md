# Project Structure And Conventions

Updated 2026-07-15 for branch `final-research-implementation`.

This is the **canonical map** of the research project. It is the single source of
truth for where things live and how new code/results are added. The project
implements and evaluates **BOND-style dimension pruning for ColBERT MaxSim on a
PDX-compatible layout** (see `docs/project_b_analysis_and_research_plan.md`).

Goal of the layout: a **clear, start-to-finish, self-contained research project
on this branch**. Everything explicitly needed to reproduce the research lives on
`final-research-implementation` and is referenced from the stage documents.

## Top-Level Tree

```text
.
├── README.md                       # entry point: what / how to reproduce / status
├── pyproject.toml                  # installs the `bondmaxsim` package (editable)
├── artifacts/                      # schemas, catalog, evidence manifest, baseline inventory
├── setup.sh                        # init submodules at pinned commits + build kernels
├── .gitignore                      # ignores scratch/build/data; TRACKS results + figures
├── .gitmodules                     # pinned external dependencies
│
├── docs/                           # research documents (the written project)
│   ├── project_b_analysis_and_research_plan.md   # original research-stage plan
│   ├── provisional-paper-corrections.md          # nonnumeric wording awaiting final evidence
│   ├── stage0_references_and_baselines.md        # Stage 0 artifact
│   ├── stage1_bond_maxsim_formalization.md       # Stage 1 artifact (proof + audit)
│   ├── stage4_comparison_methodology.md          # Stage 4 fairness controls + Mikel-chart dissection (R8)
│   ├── project_structure.md                      # THIS FILE
│   └── sources/                                  # primary sources brought on-branch
│       └── bond_maxsim_methodology.md            # formula source (was gitignored)
│
├── extern/                         # pinned external dependencies (git submodules)
│   ├── PDX/                         # cwida/PDX @ 93531b9  (evolved layout, ADSampling only)
│   ├── PDX-sigmod/                  # cwida/PDX @ fdc62f2  (SIGMOD snapshot, ships BOND/BSA/ADSampling)
│   └── CoRECT/                      # padas-lab-de/CoRECT @ fedf8bb2  (IR evaluation framework)
│
├── src/bondmaxsim/                 # the research Python package (import `bondmaxsim`)
│   ├── __init__.py
│   ├── config.py                   # paths, pinned commits, constants (D=128, default K, …)
│   ├── schema.py                   # legacy flat-result compatibility only
│   ├── results/                    # versioned envelopes, schemas, migration, catalog validation
│   ├── experiments/                # shared arms, timing, execution, accounting, stage orchestration
│   ├── render/                     # artifact-only figures, tables, macros, and plot data
│   ├── data/                       # embedding export, token packing, doc offsets, dataset loaders
│   ├── oracle/                     # exact MaxSim, normalization guard, exact-agreement, checkpoint simulator (R2)
│   ├── ordering/                   # dimension-order signals: natural / bond / pca (rotation)
│   ├── threshold/                  # threshold policies: self_bound / oracle / seed
│   ├── kernels/                    # ctypes bindings to the C++ kernels + build helpers
│   ├── testbed/                    # packing/config compatibility and mechanism utilities
│   ├── baselines/                  # faiss_ivf / pdx_ivf / plaid wrappers
│   └── eval/                       # qrels metrics + CoRECT standard-metric cross-check
│
├── cpp/                            # OUR C++ contribution
│   ├── per_document_oracle/        # exact-safe oracle kernels (ported exp-09 accounting, exp-10 throughput)
│   ├── wide_block_maxsim_bond/     # Stage 2 deliverable: MaxSim extension of PDX-sigmod BOND (wide token block)
│   ├── fused_panel_maxsim/          # final dense and BOND-style fused document/token scanners
│   └── README.md                   # build instructions; maps kernels to Stage 1 sections
│
├── experiments/                    # thin reproducible drivers — import bondmaxsim, never duplicate logic
│   ├── stage2_testbed/             # normalization guard, exact-agreement, two-mode smoke
│   ├── stage3_mechanism/           # bound slack, pruning-rate, order/checkpoint ablations, shrink sweep
│   ├── stage4_integration/         # method arms vs fixed candidate sets
│   └── stage5_corect/              # CoRECT IR evaluation
│
├── results/                        # tracked outputs (this is research evidence, keep on-branch)
│   ├── json/                       # versioned envelopes plus readable historical results
│   ├── figures/                    # final figures used in the paper
│   └── external_baselines/         # preserved external results we cite
│       └── zen5-martan/            # PDX-sigmod BOND-vs-ADSampling-vs-BSA results (preserved from local checkout)
│
└── archive/                        # exploratory history — kept, clearly marked non-decisive
    ├── preliminaries/              # was research/preliminaries (exp 01–11)
    └── colbert_scripts/            # was research/colbert (to be refactored into src/bondmaxsim)
```

## Where Each Used Artifact Lives

The "things we end up using" from earlier exploration, and their home in this
structure. When a stage document references one of these, it cites the path
below (on-branch), not a branch-local or gitignored path.

| Used artifact | Origin | Home on this branch |
|---|---|---|
| Exact MaxSim oracle | `archive/colbert_scripts/02b_corect_bruteforce.py` | `src/bondmaxsim/oracle/exact_maxsim.py` |
| Exact-safe per-document kernel (accounting) | preliminaries exp-09 `maxsim_kernels.cpp` | `cpp/per_document_oracle/` + `src/bondmaxsim/kernels/` |
| Throughput kernel (dense-warmup) | preliminaries exp-10 `maxsim_kernels.cpp` | `cpp/per_document_oracle/` |
| Threshold policies (self-bound / oracle / seed) | `Mikel:experiments/pipeline/05_maxsim_bond_instrumentation.py` | `src/bondmaxsim/threshold/` (logic) + `experiments/stage3_mechanism/` |
| Dimension-order signals | preliminaries exp-09/11 (`bond_order`) | `src/bondmaxsim/ordering/` |
| MaxSim bound / shrink formula source | `docs/sources/bond_maxsim_methodology.md` (was gitignored) | `docs/sources/bond_maxsim_methodology.md` |
| PLAID / BEIR baselines | `archive/colbert_scripts/02_corect_quick.py`, `03_beir_comparison.py` | `src/bondmaxsim/baselines/`, `src/bondmaxsim/eval/` |
| BOND-in-PDX substrate (single-vector L2) | `extern/PDX-sigmod/include/pdx/bond.hpp` | `extern/PDX-sigmod/` (pinned), extended by `cpp/wide_block_maxsim_bond/` |
| External BOND benchmark results we cite | `extern/PDX-sigmod/benchmarks/results/ZEN5-Martan/` (local-only) | `results/external_baselines/zen5-martan/` |
| CoRECT evaluation framework | `extern/CoRECT/` | `extern/CoRECT/` (pinned) |

## Conventions

1. **One responsibility per module file.** Split distinct logic into separate
   files. Use classes for stateful components (e.g. `TopK`, the testbed
   `Runner`, kernel bindings); plain functions for pure transforms (orders,
   bounds, metrics).
2. **Experiments are thin.** Everything in `experiments/` imports from
   `bondmaxsim` and only wires inputs → run → write results. No algorithm logic
   lives in a driver. If two drivers need the same code, it belongs in the
   package.
3. **One current result envelope.** New experiments write the versioned
   `bondmaxsim.result-envelope` schema atomically. Historical flat
   `ResultRecord` files remain readable only through compatibility readers.
   Artifacts are selected through the catalog and evidence manifest, never by
   filename convention alone.
4. **Exact and approximate never mix.** `shrink = 1` (exact-safe) and
   `shrink < 1` (approximate) results are reported on separate arms; the schema's
   `method`/`threshold_policy` fields must distinguish them (Stage 1 §3).
5. **Accounting vs timing are separate.** Algorithmic-work counters and
   wall-clock observations retain distinct fields and scope semantics. Timing
   sessions preserve every counterbalanced observation and use paired margins;
   independently selected minima are not combined.
6. **External code is pinned and referenced by path+commit.** Cite
   `extern/<repo>/<path>` with the pinned commit. Never copy upstream source into
   the package; build our C++ against their headers via the submodule.
7. **Mechanism vs candidate-generation stay separate.** Dimension-pruning claims
   run exhaustive-but-pruned (no IVF); any candidate-generation arm is reported
   apart so an IVF/PLAID win is never read as a BOND win (Stage 0 decision 2).
8. **Archive is non-decisive.** `archive/` holds exploratory history for
   provenance only. Final claims cite `src/`, `cpp/`, `experiments/`, `results/`.

## .gitignore Policy

- **Ignore:** `__pycache__/`, `*.pyc`, `*.so`, `*.o`, build dirs, virtualenvs,
  large/regenerable data (`*.npy`, `*.pt`, `*.pkl`), downloaded datasets,
  top-level planning `Notes/`.
- **Track:** everything under `results/` (JSON, final figures, preserved
  external baselines), `docs/`, `src/`, `cpp/` source, `experiments/`,
  `pyproject.toml`, `setup.sh`, `.gitmodules`. Results and figures are research
  evidence and must stay on-branch.

## Reproduction

Standard toolchain: [uv](https://docs.astral.sh/uv/) manages the venv (`.venv`)
and dependencies. `./setup.sh` runs the full bootstrap; commands then run under
`uv run` (no manual `source .venv/bin/activate` or `PYTHONPATH=src` needed —
`src/` is on the path via `pyproject.toml`'s `pytest.pythonpath` and the editable
install).

```text
./setup.sh                 # uv venv + editable install + build C++ kernels + init submodules
uv run pytest              # exact-agreement + normalization guards (Stage 2 blocking checks)
uv run python -m experiments.stageN_*.<driver>   # reproduce a stage's results into results/json/
```

Dependency extras: core install is NumPy-only (mechanism testbed); the retrieval
stack for Stages 3–5 is `uv pip install -e ".[dev,retrieval,faiss]"`.

## Stage Status

This table records implementation history, not current publication evidence.
Every numeric statement below is historical/provisional until its source and
field are classified as current by `artifacts/catalog.yaml`, selected by
`artifacts/paper_evidence.yaml`, and regenerated after the final-run gate.

| Stage | Artifact | Status |
|---|---|---|
| 0 References & baselines | `docs/stage0_references_and_baselines.md` | Done |
| 1 BOND-MaxSim formalization | `docs/stage1_bond_maxsim_formalization.md` | Done (incl. §10 addendum, 2026-07-03: proof transfer to the fused panel kernel) |
| Structure & self-containment | this file + scaffold + submodules | In progress |
| 2 Mechanism testbed (instruments v1) | `src/bondmaxsim/`, `cpp/`, `experiments/stage2_testbed/` | Done. Post-3b instrument roles: wide-block ACCOUNTING kernel = algorithmic-work microscope (active); wide-block THROUGHPUT kernel = retired from experiments (Stage 2 record + gate tests only); wide-block dense scan = removed |
| 3b Fused panel kernels (instruments v2) | `docs/stage3b_fused_panel_maxsim_kernel.md`, `cpp/fused_panel_maxsim/` | Done (K1–K5; brute = decision-gate dense baseline, bond doc/token = wall-clock mechanism instruments; overhead revision + parameterized checkpoints 2026-07-03, §6.1.1; cheap query-only bound arm `_bond_cheap` added 2026-07-03, `docs/bond2002_bound_cost_analysis.md`; gates green; scifact measured) |
| 3 Mechanism experiments | `experiments/stage3_mechanism/` | In progress. e01–e04 + e06 done on all 4 datasets (e03/e04: exact-safe BOND at default C={32,64} never beats dense; e06: policy choice moves cells only a few points); e05 done on scifact+nfcorpus (no G2 point at default C); e07 done on scifact/nfcorpus/arguana (reorder cost negligible; scidocs deferred — memory issue); e08 done on all 4 — LATE checkpoints (C={96}/{112}) prune 88–98% exact-safe (e08's standalone-baseline "beats dense on 3/4" was later shown inflated — see R12c); e09 done on all 4. R12a (2026-07-04 overnight, e09 at late sets {112}/{64,112}/{32,64,96,112}) → adopt the TIGHT bound: the cheap query-only bound prunes 0% at every late checkpoint (reverses BOND-2002). R12b (2026-07-04): the e08 "wall-clock exceeds cells" / arguana −19% exact-safe win is a MEASUREMENT ARTIFACT of a standalone dense baseline — interleaved back-to-back it corrects to +8.7% MT. R12c (2026-07-04, driver `r12c_interleaved_exact_safe.py`) re-measured all 4 datasets interleaved (1T + all-cores): corrected all-cores margins arguana +8.7%, scidocs +0.9%, scifact +2.3%, nfcorpus +0.9% — only arguana wins, so **G1 is NOT met at the honest all-cores baseline** (single-dataset, not ≥2). R12 CLOSED (a/b/c done). Scope-final 2026-07-08: RQ4/e05 remainder (R5) and scale (R7) deferred to future work; only e07-scidocs remains (memory issue, back of queue) |
| 4 Candidate-kernel integration | `experiments/stage4_integration/`, `docs/stage4_comparison_methodology.md` | Done (R8, 2026-07-09). e01/e02/e03 on all 4 datasets, MT+1T, interleaved: at matched candidate budgets every token-level pipeline (faiss_ivf/plaid/pdx_ivf) loses to the exhaustive fused dense scan except on scidocs (25.7k docs — faiss/pdx +56/+62% at B=100, recall 0.92); a strong IVF tau-seed recovers ~99% of ORACLE pruning but seed cost > kernel saving everywhere (self_bound stays the free production policy); the partitioned fused scan is a genuine APPROXIMATE frontier (2–5x at recall 0.87–0.95 on arguana/scidocs) whose full-probe exact control is slower than the monolithic scan. Verdict: at this scale the exhaustive fused dense kernel IS the system. Next: R10 write-up |
| 5 Standard IR evaluation | `experiments/stage5_corect/`, `src/bondmaxsim/eval/` | Evaluate the best Stage 3–4 methods with qrels metrics (nDCG@10, recall@100, MRR@10, recall versus exact@10) and standard-metric cross-validation through pinned CoRECT `evaluate_results`. Relevance Composition is not computed. Arms include dense fused, exact-safe BOND, partitioned frontier points, and tuned FAISS-IVF/PLAID system-cap references under the same interleaved controls. |

The working plan (research questions RQ1–RQ5, instrument table, R-items,
decision gates G1/G2) lives in `docs/project_b_analysis_and_research_plan.md`
("Status And Research Plan"). Update both when a stage advances.
