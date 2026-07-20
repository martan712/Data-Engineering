# Accelerating ColBERT Multi-Vector Search with PDX/BOND

This repository contains a Data Engineering research project on ColBERT style
multi-vector retrieval. The main question is whether the PDX data layout and
BOND dimension pruning can reduce the cost of MaxSim search.

The complete methodology, results, and limitations are in the
[LaTeX report](report/main.tex). This README is only a guide to the repository
and the commands needed to run it.

## Repository structure

`cpp/`
: C++17 implementations of the compiled exact MaxSim baseline and the
  exact-safe BOND MaxSim kernel.

`experiments/pipeline/`
: Current preparation, encoding, benchmark, and figure-generation scripts.
  Scripts `08` and `10` are the main controlled benchmark runners.

`experiments/archive/`
: Earlier experiments kept for traceability. They are not the source of the
  final performance claims.

`results/`
: JSON result files. `results/final/` contains the accepted release results,
  while the other folders contain validation runs, pilots, and older work.

`docs/`
: Benchmark protocol, implementation notes, final result summary, and figures.

`report/`
: LaTeX source for the project report and its bibliography.

`tests/`
: Unit tests for the kernels, candidate pipeline, result manifests, and report
  consistency.

Generated embeddings, indexes, and downloaded datasets are stored under
`artifacts/`. They are not committed because they are large. The PDX checkout
is expected at `external/PDX` and is also kept outside Git.

## Environment

The PDX package does not support Windows. Run the benchmarks in Linux or WSL2,
preferably from native Linux storage rather than a mounted Windows directory.
The controlled release used Python 3.11.9, Clang, OpenMP, and CPU execution.

Install the system dependencies on Ubuntu:

```bash
sudo apt update
sudo apt install -y git build-essential clang cmake python3 python3-venv \
  python3-pip libomp-dev libopenblas-dev
```

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
python -m pip install -r requirements.txt -r requirements-pdx.txt
```

Clone and install the audited PDX revision:

```bash
mkdir -p external
git clone https://github.com/cwida/PDX external/PDX
git -C external/PDX checkout --detach fdc62f2d22b3793060abf633cb5407438c7f739b
git -C external/PDX submodule update --init --recursive
CXX=clang++ python -m pip install ./external/PDX
```

Build the two project kernels and run the tests:

```bash
USE_OPENMP=1 bash cpp/exact_maxsim/build_wsl.sh
USE_OPENMP=1 bash cpp/bond_maxsim/build_wsl.sh
python -m unittest discover -s tests -v
```

## Small reproducibility run

The following commands prepare a small SciFact subset and encode it with
`lightonai/GTE-ModernColBERT-v1`. The preparation script downloads the public
BEIR archive once and then reuses the local copy.

```bash
python experiments/pipeline/01_prepare_beir_benchmark.py \
  --dataset scifact --max-documents 200 --max-queries 10 \
  --name scifact_smoke

python experiments/pipeline/02_export_embeddings.py \
  --corpus-dir artifacts/scifact_smoke_benchmark \
  --output-dir artifacts/scifact_smoke_benchmark_embeddings \
  --batch-size 16
```

Run a short exact, FAISS-IVF, and PDX-IVF comparison:

```bash
OMP_NUM_THREADS=4 python experiments/pipeline/08_controlled_ivf_pilot.py \
  --corpus-dir artifacts/scifact_smoke_benchmark \
  --embeddings-dir artifacts/scifact_smoke_benchmark_embeddings \
  --output artifacts/results/scifact_smoke_ivf.json \
  --max-documents 200 --max-queries 5 \
  --top-l 50 --nprobe 8 --c-values 20 50 \
  --threads 4 --warmup-runs 1 --measured-runs 1 \
  --pdx-source external/PDX
```

Run a short exact versus BOND comparison on the same embeddings:

```bash
OMP_NUM_THREADS=4 python experiments/pipeline/10_controlled_bond_pilot.py \
  --corpus-dir artifacts/scifact_smoke_benchmark \
  --embeddings-dir artifacts/scifact_smoke_benchmark_embeddings \
  --output artifacts/results/scifact_smoke_bond.json \
  --max-documents 200 --max-queries 5 \
  --k 10 --seed-counts 10 50 --threads 4 \
  --warmup-runs 1 --measured-runs 1
```

These commands are smoke tests. The final measurements use pinned CPU
affinity, fixed thread settings, held-out queries, and five interleaved runs.
The exact measurement contract is documented in
[`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).

## Results and figures

The accepted JSON files are already available in `results/final/`. The report
figures can be regenerated directly from them:

```bash
python experiments/pipeline/09_make_controlled_figures.py
python experiments/pipeline/11_make_bond_figure.py
python experiments/pipeline/12_make_transfer_figure.py
python experiments/pipeline/14_make_plaid_figure.py
```

For a short description of the released evidence, see
[`docs/final_results.md`](docs/final_results.md). Earlier exploratory results
remain available for audit purposes but should not be used for final timing
claims.

## Compile the report

From the `report/` directory:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The report uses the figures in `docs/figures/`. More LaTeX details are provided
in [`report/README.md`](report/README.md).
