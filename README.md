# Accelerating ColBERT Multi-Vector Search with PDX

Data Engineering project. Goal: make **ColBERT-style multi-vector retrieval**
faster using the **PDX** vector library and the **BOND** branch-and-bound idea
(SIGMOD 2002), and evaluate it honestly against established baselines.

ColBERT represents every document as a *variable-length set of token vectors*
and scores with **MaxSim**:

```text
score(query, document) = sum over query tokens q_i of  max over document tokens d_j of  <q_i, d_j>
```

This is the single source of truth for the repo. It covers setup, how to
reproduce everything, the results, and the full iteration history.

---

## TL;DR — what we found

1. **The fast, quality-preserving recipe is two-stage:** IVF (cluster) candidate
   generation over the flat token-vector index, then **exact MaxSim re-rank** on
   the small candidate pool. On full SciFact this gives a **~30–40x speedup that
   grows with corpus size, with no loss in qrels recall@10 (0.938 = exact)**, and
   it **replicates on a second dataset (NFCorpus): ~32x at ~97% of exact recall.**
2. **The win is architectural, not PDX-specific.** FAISS-IVF reproduces the
   *identical* recall at a similar speedup, so the result is the *recipe*, and
   PDX is a valid engine for it. PLAID (the ColBERT SOTA) was slower and lower
   quality on CPU because its quantized scoring trades recall; our exact re-rank
   preserves it.
3. **The professor's dimension-level BOND does not transfer to ColBERT MaxSim at
   d=128** — ColBERT spreads energy nearly uniformly over the 128 dims, so the
   pruning bound stays loose (≤1.1x even with a perfect threshold). An
   energy-concentrating **PCA rotation** partially revives it (~1.2–1.7x), but
   that is a secondary optimization, not the headline.

![IVF scaling](docs/figures/fig1_ivf_scaling.png)
![Baselines](docs/figures/fig2_baselines.png)
![BOND pruning](docs/figures/fig3_bond_pruning.png)

---

## Repository layout

```text
.
├── README.md                     # this file (the definitive documentation)
├── requirements.txt              # Windows venv deps (PyLate, FAISS, matplotlib, ...)
├── experiments/                  # the active, reproducible pipeline
│   ├── utils_colbert.py          # shared helpers (packed embeddings, MaxSim, metrics)
│   ├── 29_prepare_beir_benchmark.py        # download + subset any BEIR dataset
│   ├── 30_export_beir_pylate_embeddings.py # encode corpus/queries with ColBERT
│   ├── 31_beir_ivf_benchmark.py            # exact MaxSim vs IVF-on-PDX (+ rerank)  [WSL]
│   ├── 32_baselines_full_scifact.py        # PLAID + FAISS-IVF baselines
│   ├── 33_maxsim_bond_instrumentation.py   # multi-vector MaxSim-BOND pruning study
│   ├── 34_make_writeup_figures.py          # regenerate the figures below
│   └── archive/                  # 28 earlier exploratory scripts (iterations 1–14)
├── docs/figures/                 # PNG figures used in this README
├── artifacts/                    # data + embeddings + result JSONs (git-ignored bulk)
│   ├── <name>_benchmark/             # documents.jsonl, queries.jsonl, qrels.json
│   ├── <name>_benchmark_embeddings/  # documents_packed.npz, queries_packed.npz
│   └── results/                      # all measured result JSONs
└── external/PDX/                 # vendored PDX library (do not edit; built in WSL)
```

> The `experiments/archive/` folder holds the superseded exploratory scripts
> (toy corpus, "realish" corpus, SciFact-subset candidate/selector studies, and
> the PDX batch / shared-scan kernel prototypes). They are kept for provenance
> and are summarized in the [Iteration history](#iteration-history) below. You do
> not need them to run the current pipeline.

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

This covers: preparing/encoding datasets (29, 30), the FAISS-IVF + PLAID
baselines (32), the MaxSim-BOND instrumentation (33), and the figures (34).

### 2. WSL2 / Linux PDX build (only needed for script 31's PDX path)

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
wsl -e /bin/bash -lc "cd '<repo path>' && /home/<user>/data-engineering-pdx/.venv-pdx/bin/python experiments/31_beir_ivf_benchmark.py --corpus-dir ... --embeddings-dir ... --output ..."
```

**Key API note:** the PDX Python API exposes the `l2sq` metric only. ColBERT
token embeddings are already L2-normalized, so for unit vectors
`squared_l2 = 2 - 2·cos`, i.e. nearest-by-l2sq == largest cosine/inner product.
We convert back with `cosine = 1 - l2sq/2`. l2/IP/cosine orderings all coincide.

---

## Reproduce the pipeline

The current pipeline is dataset-agnostic (any BEIR dataset; `--max-documents 0`
= full corpus). Example used for the headline results (full SciFact: 5183 docs,
50 queries):

```powershell
# 1. Prepare the benchmark (downloads BEIR SciFact, writes documents/queries/qrels)
.\.venv\Scripts\python experiments\29_prepare_beir_benchmark.py --dataset scifact --max-documents 0 --max-queries 50 --name scifact_full

# 2. Encode with ColBERT (lightonai/GTE-ModernColBERT-v1, CPU ~23 min for 5183 docs)
.\.venv\Scripts\python experiments\30_export_beir_pylate_embeddings.py --corpus-dir artifacts/scifact_full_benchmark --output-dir artifacts/scifact_full_benchmark_embeddings --batch-size 16

# 3. Exact MaxSim vs IVF-on-PDX candidate gen + exact rerank  (RUN IN WSL — needs PDX)
wsl -e /bin/bash -lc "cd '<repo path>' && /home/<user>/data-engineering-pdx/.venv-pdx/bin/python experiments/31_beir_ivf_benchmark.py --corpus-dir artifacts/scifact_full_benchmark --embeddings-dir artifacts/scifact_full_benchmark_embeddings --output artifacts/results/scifact_full_ivf_benchmark.json"

# 4. Baselines: PLAID + FAISS-IVF on the same corpus  (Windows venv)
.\.venv\Scripts\python experiments\32_baselines_full_scifact.py --corpus-dir artifacts/scifact_full_benchmark --embeddings-dir artifacts/scifact_full_benchmark_embeddings --pdx-result artifacts/results/scifact_full_ivf_benchmark.json --output artifacts/results/scifact_full_baselines.json

# 5. MaxSim multi-vector BOND pruning study  (Windows venv; --rotation pca to test the revival)
.\.venv\Scripts\python experiments\33_maxsim_bond_instrumentation.py --embeddings-dir artifacts/scifact_full_benchmark_embeddings --k 10 --output artifacts/results/maxsim_bond_instrumentation_scifact_full.json

# 6. Regenerate the figures in docs/figures/
.\.venv\Scripts\python experiments\34_make_writeup_figures.py
```

All numbers come from the JSON files in `artifacts/results/`.

---

## Results (full SciFact: 5183 docs, 50 queries)

### IVF candidate generation scales (Figure 1)

`nprobe=8, L=100`, speedup = exact MaxSim time / IVF candidate-gen time:

| corpus size | exact (s) | IVF gen (s) | speedup | pool recall@10 | reranked qrels recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250  | 0.40 | 0.11 | 3.5x  | 1.00 | 0.938 (= exact) |
| 500  | 0.75 | 0.13 | 5.5x  | 1.00 | 0.938 |
| 1000 | 1.45 | 0.16 | 8.9x  | 1.00 | 0.938 |
| 5183 | 7.78 | 0.26 | 29.8x | 1.00 | 0.938 |

Candidate generation stays nearly flat while exact MaxSim grows linearly, so the
speedup grows with corpus size. Quality is preserved because the exact re-rank on
the candidate pool fixes the final order.

### Baselines (Figure 2; exact/FAISS/PLAID measured on the same Windows machine)

| method | time (50 q) | qrels recall@10 |
| --- | ---: | ---: |
| Exact NumPy MaxSim | 35.1 s | 0.938 |
| **FAISS-IVF + exact rerank** | **0.90 s** | **0.938** |
| PLAID (ColBERT SOTA, CPU, nbits=4) | 56.0 s | 0.794 |
| PDX-IVF + exact rerank (WSL) | 0.29 s | 0.938 |

FAISS-IVF reproduces the exact recall at ~39x → the win is the two-stage recipe.
PLAID's quantization loses recall on CPU; exact re-rank keeps it. (PDX time is
from a different machine/env (WSL), so its *absolute* value is not directly
comparable to the same-machine FAISS/PLAID times — only the relative picture is.)

### MaxSim-BOND dimension pruning (Figure 3; provably exact, 0 true top-k pruned)

| setting | ceiling speedup = 1/work-ratio |
| --- | ---: |
| raw ColBERT, oracle threshold | 1.10x |
| PCA-rotated, realistic threshold | 1.22x |
| PCA-rotated, oracle threshold | 1.68x |

Dimension-level branch-and-bound is effectively dead on raw ColBERT (the bound
stays loose until ~all dims are read). A PCA rotation concentrates energy and
partially revives it, but it remains a secondary optimization.

### Generality check — second dataset (NFCorpus: 3633 docs, 50 queries)

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

---

## Conclusions and next steps

- **Headline contribution:** IVF candidate generation + exact MaxSim re-rank is
  a large, growing, quality-preserving speedup, triangulated three ways (vs
  NumPy, FAISS, PLAID).
- **Useful negative result:** classic dimension-decomposition BOND does not
  transfer to ColBERT MaxSim at d=128, with a clear explanation (uniform energy);
  PCA rotation only partially revives it.
- **Open follow-ups (rough priority):** (1) scale to more/larger BEIR or a CoRECT
  MS-MARCO subset (scripts 29–31 already support it); (2) a fairer PLAID ceiling
  (GPU and/or higher `nbits`/`n_full_scores`); (3) tune the rerank budget `C` to
  close the selector gap; (4) if pursuing a kernel, a combined PCA + IVF-seeded
  threshold + document-MaxSim-bound scan that stacks the two pruning effects.

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

### Caveats

- Component timings on the current Python/WSL implementation are **not** a
  controlled end-to-end speedup claim; relative comparisons within one machine
  are valid, cross-machine absolute times are not.
- qrels are small/incomplete; exact-MaxSim ranking recovery is the primary
  reference, qrels recall@k the IR metric.
- The PDX kernel prototypes in iterations 13–14 patched the vendored PDX
  (`external/PDX`) in the WSL checkout; those patches are not required for the
  current IVF pipeline.
