# Stage 5 IR evaluation

Stage 5 evaluates the selected retrieval-system arms on the stable
`beir-<dataset>-qrels-test-v1` workloads. It reports ordinary qrels metrics
(nDCG@10, recall@100, and MRR@10), recall against the exact MaxSim ranking,
actual candidate-work observations where available, and counterbalanced raw
timing observations.

CoRECT is used only to cross-validate those ordinary metrics through the
pinned `corect.utils.evaluate_results` function. The dense arm must agree with
the independent local evaluator before the artifact is accepted. No separate
CoRECT evaluation target is computed or claimed.

## Drivers

`e01_ir_evaluation.py` is declarative orchestration. Data/index preparation,
single-pass arm execution, correctness checks, metrics, timing, provenance,
and atomic serialization live under `bondmaxsim.experiments.stage5`. Full
vector arms are:

- `dense-fused`, `openblas-exact`, and `bond-exact-safe` as exact-safe scans;
- `partitioned-nprobe-*` as the probe frontier;
- `faiss-cap-*` as exact-rerank arms with observed full-score counts;
- `plaid-cap-*` as system-cap references because actual full-score work is
  unavailable from the pinned public API.

Run a diagnostic fixture or a full workload with:

```bash
uv run python -m experiments.stage5_corect.e01_ir_evaluation --fixture
uv run python -m experiments.stage5_corect.e01_ir_evaluation --dataset scifact --threads 1
```

`e02_paired_significance.py` consumes the saved, validated e01 artifact. It
uses retained per-query nDCG@10 values and does not rebuild the corpus or rerun
retrieval:

```bash
uv run python -m experiments.stage5_corect.e02_paired_significance --input results/json/<e01-artifact>.json
```

`e03_bm25_baseline.py` is a lexical quality reference. Its index build is
excluded and query tokenization remains inside its timer. Its artifact marks
timing as standalone and explicitly forbids latency margins against e01 vector
arms:

```bash
uv run python -m experiments.stage5_corect.e03_bm25_baseline --fixture
HF_DATASETS_OFFLINE=1 uv run python -m experiments.stage5_corect.e03_bm25_baseline --dataset scifact
```

Rendering always reopens a saved artifact:

```bash
uv run python -m experiments.stage5_corect.render_ir_table results/json/<artifact>.json
uv run python -m experiments.stage5_corect.render_ir_table results/json/<artifact>.json --figure results/figures/stage5.png
```

Fixture artifacts are diagnostic only and are never performance evidence.
