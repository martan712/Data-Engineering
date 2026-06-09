# Accelerating ColBERT Multi-Vector Search with PDX

Data Engineering project. **Primary goal:** test whether **PDX-BOND**
(branch-and-bound on vertically decomposed vectors, SIGMOD 2002) can speed up
**ColBERT-style multi-vector retrieval** using the PDX library, in an information
retrieval setting (BEIR benchmarks; CoRECT-scale evaluation as a follow-up).

ColBERT represents every document as a *variable-length set of token vectors*
and scores with **MaxSim**:

```text
score(query, document) = sum over query tokens q_i of  max over document tokens d_j of  <q_i, d_j>
```

This README is the single source of truth for the repo: what we tested, what
failed, what worked as a fallback, and how to reproduce it.

---

## TL;DR — main result: PDX-BOND did not speed up ColBERT search

**We tested the proposed idea thoroughly. In our ColBERT setup it does not
deliver a meaningful speedup.**

1. **Flat PDX-BOND (`IndexPDXBONDFlat`) was slower than exact NumPy MaxSim.**
   Each query token triggers a near-exhaustive scan over all document token
   vectors (~230k–1.2M). BOND prunes *dimensions*, not documents/clusters, and
   at d=128 ColBERT embeddings spread energy nearly uniformly — so the bound
   stays loose and almost nothing gets skipped.
2. **We also instrumented MaxSim-aware, document-level BOND** (script 33).
   Provably exact (0 true top-k wrongly pruned), but the ceiling is only
   **~1.1x** on raw embeddings (~1.2x realistic with PCA rotation). This is not
   a useful speedup for the project goal.
3. **Ruled-out alternatives confirm the diagnosis:** batching PDX Python calls
   (~1.0x), shared scan without pruning (0.26x — slower). The bottleneck is the
   search kernel doing too much work, not call overhead.

**What we tested and ruled out (BOND as the speedup mechanism):**

![BOND pruning](docs/figures/fig3_bond_pruning.png)

---

### Secondary finding: what actually worked

While pursuing the BOND hypothesis we built a full ColBERT-on-PDX pipeline and
found a **practical fallback** that does speed up search — but the speedup comes
from **IVF cluster pruning**, not from BOND dimension pruning:

- **IVF candidate generation + exact MaxSim re-rank** over the flat token index.
- ~30x on full SciFact, replicates on NFCorpus; FAISS-IVF gets the same pattern
  → the win is the *recipe*, not PDX-specific magic.
- Quality is preserved (exact re-rank); tunable via re-rank budget `C`.

![IVF scaling](docs/figures/fig1_ivf_scaling.png)
![Baselines](docs/figures/fig2_baselines.png)

---

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── experiments/
│   ├── _paths.py                 # shared project-root / import helper
│   ├── utils_colbert.py          # MaxSim, packed embeddings, metrics
│   ├── pipeline/                 # ← run these (7 scripts, in order)
│   │   ├── 01_prepare_beir_benchmark.py
│   │   ├── 02_export_embeddings.py
│   │   ├── 03_beir_ivf_benchmark.py      [WSL / PDX]
│   │   ├── 04_baselines.py
│   │   ├── 05_maxsim_bond_instrumentation.py   ★ primary hypothesis test
│   │   ├── 06_make_figures.py
│   │   └── 07_selector_gap_sweep.py
│   └── archive/                  # earlier iterations, grouped by phase
│       ├── phase01_setup/        # 01–03  PyLate + PDX smoke test
│       ├── phase02_toy/          # 04–05  toy corpus, token PDX
│       ├── phase03_local/        # 06–10  174-doc local corpus
│       ├── phase04_realish/      # 11–18  600-doc corpus + selector studies
│       ├── phase05_scifact_flat_bond/  # 19–24  SciFact subset, flat BOND (slow)
│       ├── phase06_kernel_prototypes/  # 25–26  batch / shared-scan (ruled out)
│       └── phase07_ivf_discovery/      # 27–28  IVF breakthrough (superseded by pipeline/03)
├── docs/figures/
├── artifacts/
│   ├── <name>_benchmark/
│   ├── <name>_benchmark_embeddings/
│   └── results/
└── external/PDX/
```

### Experiment map (what to run vs what is history)

| Folder | Purpose | Run it? |
| --- | --- | --- |
| `experiments/pipeline/` | Current reproducible workflow | **Yes** — steps 01→07 |
| `experiments/archive/phase01–04` | Early corpora + selector debugging | No — background only |
| `experiments/archive/phase05` | SciFact @ 1k docs, **flat BOND slower than exact** | No — shows the failure |
| `experiments/archive/phase06` | Ruled out batching / shared scan | No — negative results |
| `experiments/archive/phase07` | First IVF results (SciFact-only scripts) | No — use `pipeline/03` instead |

Old script numbers (e.g. `29_…`, `33_…`) map to `pipeline/01_…`, `pipeline/05_…`, etc.

---

## Environment setup

Two environments are required because **PDX does not build on Windows** (its
`setup.py` raises `Windows not yet implemented`). Everything except the PDX
search itself runs in a normal Windows venv; the PDX step runs in WSL2/Linux.

### 1. Windows venv (PyLate encoding, FAISS, PLAID, figures)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -r requirements.txt
```

This covers: `pipeline/01–02` (prepare/encode), `04` (baselines), `05` (BOND
study), `06` (figures), and `07` (selector sweep).

### 2. WSL2 / Linux PDX build (only needed for `pipeline/03`)

Build PDX in **native WSL storage** (not `/mnt/c/...`; mounted paths caused venv
and git-submodule permission errors on this machine). Tested on Ubuntu WSL2,
Python 3.14, clang.

```bash
sudo apt update
sudo apt install -y git build-essential clang cmake python3 python3-venv python3-pip libomp-dev libopenblas-dev

WORK="$HOME/data-engineering-pdx"
mkdir -p "$WORK"; cd "$WORK"
python3 -m venv .venv-pdx
source "$WORK/.venv-pdx/bin/activate"
python -m pip install --upgrade pip setuptools wheel

mkdir -p external
git clone https://github.com/cwida/PDX external/PDX
cd external/PDX
git checkout sigmod
git submodule update --init --recursive

export CXX=clang++
# PDX pins its own numpy/sklearn/setuptools/pybind11; h5py needs a wheel on py3.14;
# faiss-cpu is required by PDX examples but missing from its requirements.txt.
python -m pip install numpy==2.2.6 scikit-learn==1.6.1 setuptools==75.5.0 pybind11==2.13.6 h5py==3.16.0 faiss-cpu
python -m pip install .
python examples/pdxearch_simple.py   # smoke test
```

Scripts that need PDX are run through this interpreter, e.g.:

```powershell
wsl -e /bin/bash -lc "cd '<repo path>' && /home/<user>/data-engineering-pdx/.venv-pdx/bin/python experiments/pipeline/03_beir_ivf_benchmark.py --corpus-dir ... --embeddings-dir ... --output ..."
```

**Key API note:** the PDX Python API exposes the `l2sq` metric only. ColBERT
token embeddings are already L2-normalized, so for unit vectors
`squared_l2 = 2 - 2·cos`, i.e. nearest-by-l2sq == largest cosine/inner product.
We convert back with `cosine = 1 - l2sq/2`. l2/IP/cosine orderings all coincide.

---

## Reproduce the pipeline

Scripts live in `experiments/pipeline/`. **Step 05 is the primary hypothesis test**
(BOND pruning); steps 03–04 and 07 are secondary IVF/rerank work. Example on full
SciFact (5183 docs, 50 queries):

```powershell
# 01 — Prepare benchmark
.\.venv\Scripts\python experiments/pipeline/01_prepare_beir_benchmark.py --dataset scifact --max-documents 0 --max-queries 50 --name scifact_full

# 02 — Encode with ColBERT (~23 min CPU for 5183 docs)
.\.venv\Scripts\python experiments/pipeline/02_export_embeddings.py --corpus-dir artifacts/scifact_full_benchmark --output-dir artifacts/scifact_full_benchmark_embeddings --batch-size 16

# 03 — Exact MaxSim vs IVF-on-PDX  (RUN IN WSL — needs PDX)
wsl -e /bin/bash -lc "cd '<repo path>' && /home/<user>/data-engineering-pdx/.venv-pdx/bin/python experiments/pipeline/03_beir_ivf_benchmark.py --corpus-dir artifacts/scifact_full_benchmark --embeddings-dir artifacts/scifact_full_benchmark_embeddings --output artifacts/results/scifact_full_ivf_benchmark.json"

# 04 — Baselines: PLAID + FAISS-IVF
.\.venv\Scripts\python experiments/pipeline/04_baselines.py --corpus-dir artifacts/scifact_full_benchmark --embeddings-dir artifacts/scifact_full_benchmark_embeddings --pdx-result artifacts/results/scifact_full_ivf_benchmark.json --output artifacts/results/scifact_full_baselines.json

# 05 — ★ Primary: MaxSim multi-vector BOND pruning study
.\.venv\Scripts\python experiments/pipeline/05_maxsim_bond_instrumentation.py --embeddings-dir artifacts/scifact_full_benchmark_embeddings --k 10 --output artifacts/results/maxsim_bond_instrumentation_scifact_full.json

# 07 — Selector-gap sweep (re-rank budget C)
.\.venv\Scripts\python experiments/pipeline/07_selector_gap_sweep.py --corpus-dir artifacts/scifact_full_benchmark --embeddings-dir artifacts/scifact_full_benchmark_embeddings --output artifacts/results/scifact_full_selector_gap_sweep.json

# 06 — Regenerate figures
.\.venv\Scripts\python experiments/pipeline/06_make_figures.py
```

All numbers come from the JSON files in `artifacts/results/`.

---

## Results

### 1. Primary — PDX-BOND does not speed up ColBERT (Figure 3)

We tested BOND at three levels; none gave a useful speedup for ColBERT MaxSim
search in our setup (`GTE-ModernColBERT-v1`, d=128, BEIR SciFact up to 5183 docs).

**Flat PDX-BOND (the direct re-implementation path)**

| corpus | exact MaxSim | flat PDX-BOND candidate gen | outcome |
| ---: | ---: | ---: | --- |
| 1000 docs | ~1.6–2.5 s | ~3.6 s (`L=50`) | **slower than exact** |
| 5183 docs | ~7.8 s | not competitive | motivated switch to IVF |

Flat BOND scans essentially all token vectors per query token. Dimension pruning
barely reduces work because ColBERT energy is spread across all 128 dimensions.

**MaxSim-aware document-level BOND instrumentation** (script 33; provably exact,
0 true top-k wrongly pruned):

| setting | ceiling speedup = 1/work-ratio |
| --- | ---: |
| raw ColBERT, oracle threshold | 1.10x |
| PCA-rotated, realistic threshold | 1.22x |
| PCA-rotated, oracle threshold | 1.68x |

**Why it fails:** BOND's Cauchy-Schwarz bound needs early dimensions to carry
enough signal to eliminate documents. ColBERT's L2-normalized 128-dim embeddings
do not concentrate energy that way. PCA rotation partially helps but remains far
below what IVF delivers.

**Conclusion for the project proposal:** re-implementing BOND for ColBERT
multi-vector search, as originally scoped, **did not achieve the speedup goal**.
The negative result is well-supported and reproducible (see script 33 and
iterations 11–14, 19–20 in the history below).

---

### 2. Secondary — IVF + exact rerank (what worked instead)

After flat BOND failed, we switched to **`IndexPDXBONDIVFFlat`** (IVF cluster
pruning, then BOND inside probed clusters). The speedup comes from **IVF
skipping whole clusters**, not from BOND dimension pruning. FAISS-IVF reproduces
the same recall/speedup pattern on the same data.

**IVF scaling on SciFact** (Figure 1; `nprobe=8, L=100`):

| corpus size | exact (s) | IVF gen (s) | speedup | pool recall@10 | reranked qrels recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250  | 0.40 | 0.11 | 3.5x  | 1.00 | 0.938 (= exact) |
| 500  | 0.75 | 0.13 | 5.5x  | 1.00 | 0.938 |
| 1000 | 1.45 | 0.16 | 8.9x  | 1.00 | 0.938 |
| 5183 | 7.78 | 0.26 | 29.8x | 1.00 | 0.938 |

**Baselines on full SciFact** (Figure 2; exact/FAISS/PLAID same Windows machine):

| method | time (50 q) | qrels recall@10 |
| --- | ---: | ---: |
| Exact NumPy MaxSim | 35.1 s | 0.938 |
| **FAISS-IVF + exact rerank** | **0.90 s** | **0.938** |
| PLAID (ColBERT SOTA, CPU, nbits=4) | 56.0 s | 0.794 |
| PDX-IVF + exact rerank (WSL) | 0.29 s | 0.938 |

IVF is standard ANN practice, not a novel contribution. Its value here is as a
**validated fallback pipeline** for fast ColBERT retrieval on PDX, discovered
after the BOND hypothesis failed.

### 2b. Generality — second dataset (NFCorpus: 3633 docs, 50 queries)

To confirm the recipe is not SciFact-specific we re-ran the same pipeline on
BEIR **NFCorpus**, a denser-relevance dataset (1739 qrels labels, ~865k document
token vectors). Exact/FAISS/PLAID measured on the same Windows machine:

| method | time (50 q) | qrels recall@10 | exact agreement@10 |
| --- | ---: | ---: | ---: |
| Exact NumPy MaxSim | 7.50 s | 0.194 | 1.000 |
| **FAISS-IVF + exact rerank** (`nprobe=8, C=50`) | **0.23 s** | **0.188** | 0.840 |
| PLAID (CPU, nbits=4) | 26.8 s | 0.154 | 0.460 |

The two-stage IVF + exact-rerank recipe again gives a **~32x speedup at ~97% of
exact qrels recall** with 0.94 pool coverage, while CPU-PLAID is again slower
than exact and lower quality. The **selector gap reappears** (agreement@10
plateaus at ~0.84 despite 0.94 coverage, and raising `nprobe` 8→16→32 did not
help) — the same signal as SciFact that the fixed `C=50` re-rank budget, not
candidate coverage, is the remaining quality limiter. Note the PDX path (script
31, WSL) has not yet been run on NFCorpus; FAISS stands in for the IVF engine here.

### 2c. Selector-gap sweep (Figure 4) — the C=50 plateau is a budget artifact

The sweep in `35_selector_gap_sweep.py` fixes candidate generation
(`nprobe=8, L=100`) and grows the re-rank budget `C` (selection policy turns out
to barely matter). Agreement with the exact top-10 rises smoothly with `C` and
reaches the pool-coverage limit when the **entire pool** (~400–600 docs) is
re-ranked — at a latency still far below exact:

| dataset | C=50 | C=100 | C=200 | C=pool | pool coverage@10 | time @ C=pool | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact-full | 0.862 | 0.932 | 0.968 | **1.000** | 1.000 | 1.78 s | 8.0x |
| NFCorpus | 0.840 | 0.916 | 0.960 | **0.978** | 0.940 | 0.89 s | 8.9x |

Practical sweet spots: `C=100` keeps ~30x at agreement ≥0.92; `C=200` keeps
~16–20x at ~0.96. On SciFact, re-ranking the full pool gives **perfect (1.000)
agreement with exact MaxSim at 8x speedup**. On NFCorpus the full-pool residual
(0.978) is exactly the pool-coverage limit; widening candidate generation to
`nprobe=32` with `C=200` recovers **qrels recall@10 = 0.1942 = exact** at 0.98 s
(7.4x). So the earlier 0.84 plateau was not structural: quality vs speed is a
smooth, tunable budget curve, and exact-quality operating points exist at ~7–8x.

![Selector gap](docs/figures/fig4_selector_gap.png)

---

## Conclusions and next steps

**Primary (answers the project proposal):**

- We implemented and tested PDX-BOND for ColBERT multi-vector search as
  proposed. **It did not deliver a meaningful speedup** in our experiments:
  flat BOND was slower than exact search; MaxSim-aware dimension BOND caps at
  ~1.1x (raw) / ~1.2x (PCA). We have a clear explanation (uniform energy across
  128 dims → loose pruning bounds) and reproducible instrumentation (script 33).
- This is the main empirical contribution relative to the SIGMOD 2002 idea
  applied to modern ColBERT IR.

**Secondary (practical outcome discovered along the way):**

- **IVF cluster pruning + exact MaxSim re-rank** does speed up ColBERT search
  (~30x on SciFact, replicates on NFCorpus). FAISS-IVF matches the pattern → IVF
  is the mechanism, not BOND. Quality is tunable via re-rank budget `C`.
- IVF is not novel; it is the fallback that worked after BOND failed.

**Open follow-ups:** (1) CoRECT / larger BEIR scale-up; (2) PDX path on NFCorpus
in WSL; (3) fairer PLAID comparison; (4) only if revisiting BOND: IVF-seeded
thresholds + document-level MaxSim bounds (modest ceiling from instrumentation).

---

## Iteration history

Condensed log of how the project got here. Scripts 1–14 referenced below live in
`experiments/archive/`; 15–20 use the active scripts. Result JSONs are in
`artifacts/results/`.

| # | What | Outcome |
| ---: | --- | --- |
| 1–2 | PyLate baseline; build PDX in WSL | ColBERT encodes (toy PLAID index); PDX builds on `sigmod` branch; BOND smoke test matches NumPy exact L2. |
| 3 | Toy ColBERT → flat PDX token index | Token-only PDX scoring ≠ exact MaxSim ranking; need candidate-gen + rerank. PDX exposes only `l2sq` → L2-normalize + cosine conversion. |
| 4 | Local 174-doc corpus | Candidate generation + exact rerank recovers exact top-1/3/5 at small `L,C`. Token-only scoring unreliable. |
| 5 | "Realish" 600-doc corpus | Recipe still works; exact top-5/10 recovery no longer perfect → two failure modes appear. |
| 6 | Candidate-selection study | Split **pool coverage** vs **selector loss**. At `L=50` the pool contains exact top-10 for all queries → remaining misses are selector failures. |
| 7–9 | Coverage-preserving + MaxSim-aware selectors | `union_approx_and_count` best at `C=50`; heuristic selectors only marginally help. Misses are selector, not coverage, failures. |
| 10 | First real benchmark: SciFact subset (1000 docs) | Recipe transfers to BEIR. At this size PDX candidate-gen is *slower* than exact NumPy → needs scaling/timing analysis. |
| 11 | Timing breakdown | PDX `search` is ~97–99% of candidate-gen time; selection/rerank negligible. |
| 12 | Scaling curve + API check | No batch-search in PDX API (one call per query token). Still slower than exact at ≤1000 docs; gap narrows with size. |
| 13 | PDX batch-search prototype | Batching the Python/C++ boundary gives ~1.0x → call overhead is **not** the bottleneck; it is the search kernel. |
| 14 | Shared-scan kernel prototype | Sharing the block scan *without* pruning is 0.26x (slower). BOND pruning is essential → negative result. |
| 15 | **IVF-on-PDX** candidate generation | Switching to `IndexPDXBONDIVFFlat` (cluster pruning) gives the first real crossover: 20–40x with preserved recall. **The key fix.** |
| 16 | IVF scaling curve | Speedup grows with corpus size: 3.5x → 8.9x (250 → 1000 docs). |
| 17 | Full SciFact scale-up (5183 docs) | ~30x at `nprobe=8, L=100`, qrels recall@10 = 0.938 (= exact). Generalized scripts 29–31 added. |
| 18 | Baselines (PLAID + FAISS-IVF) | FAISS reproduces 0.938 at ~39x (win is architectural); PLAID slower + 0.794 on CPU. |
| 19 | MaxSim-BOND instrumentation | Dimension BOND dead on raw ColBERT (≤1.10x even with oracle threshold). Provably exact. |
| 20 | PCA rotation revival | Partially revives pruning (1.22x realistic / 1.68x oracle); secondary optimization. |
| 21 | Generality check on NFCorpus (2nd dataset) | Recipe transfers: FAISS-IVF + rerank ~32x at 0.188 vs 0.194 exact qrels recall@10 (~97%); CPU-PLAID again slower + lower quality. Selector gap recurs (agreement@10 ~0.84). |
| 22 | Selector-gap sweep (script 35) | The C=50 plateau is a budget artifact. Agreement rises smoothly with C; full-pool re-rank = 1.000 agreement at 8x (SciFact); `nprobe=32, C=200` = exact qrels recall at 7.4x (NFCorpus). Policy choice barely matters. |

### Caveats

- Component timings on the current Python/WSL implementation are **not** a
  controlled end-to-end speedup claim; relative comparisons within one machine
  are valid, cross-machine absolute times are not.
- qrels are small/incomplete; exact-MaxSim ranking recovery is the primary
  reference, qrels recall@k the IR metric.
- The PDX kernel prototypes in iterations 13–14 patched the vendored PDX
  (`external/PDX`) in the WSL checkout; those patches are not required for the
  current IVF pipeline.
