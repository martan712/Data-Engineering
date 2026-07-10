# Stage 5: CoRECT IR Evaluation (R9)

Rescoped 2026-07-09 (plan doc R9): one driver, not the earlier e01–e05 list.
Stage 5 is the paper's FINAL SECTION — take the best method per family from
the Stage 3–4 verdicts and evaluate them as retrieval systems in the proper IR
framework: qrels metrics (nDCG@10, recall@100, MRR@10), CoRECT RC metrics,
recall-vs-exact, and interleaved wall-clock latency, with index build cost and
memory reported separately.

Fairness controls (all satisfied by the driver): one machine, one OS, fixed
thread count per run, interleaved round-robin timing (dense never timed
standalone — the R12b lesson), repeated runs with dispersion recorded, full
quality-latency frontier, tuned external baselines.

## `e01_ir_evaluation.py`

Method arms (one-stack controls as Stage 4 e01, retrieval depth k=100):

| arm | what it is | why it is here |
|---|---|---|
| `dense_fused` | fused dense MaxSim over the full corpus | the exact production baseline (RQ2 kernel) |
| `bond_exact_safe` | fused BOND, TIGHT bound, natural order, C={112}, self_bound tau | the free exact-safe policy (R8b): recall 1.0 by construction — exactness costs nothing in IR quality |
| `partitioned@{16,32}` | partitioned fused scan (bond scanner) at two e03 frontier points | the genuine approximate frontier: nprobe 16/32 land at recall_vs_exact@10 ≈0.9/0.95 on every dataset |
| `faiss_ivf@B`, `plaid@B` | tuned external references at matched budgets B ∈ {100, 1000, 5000} | candidate-generation pipelines the qrels metrics must judge |

Quality is computed over ALL evaluable test queries (scifact 300, nfcorpus
323, arguana 200, scidocs 200 — see `data/embeddings/README.md` for the
test-query sidecars); doc indices map to BEIR IDs via `data/beir_ids/`.

Usage:

```bash
uv run python -m experiments.stage5_corect.e01_ir_evaluation 0   # all cores
uv run python -m experiments.stage5_corect.e01_ir_evaluation 1   # 1 thread
```

## How extern/CoRECT is used (and what is deliberately not used)

We execute the actual pinned checkout (`extern/CoRECT` @ `fedf8bb2`), never a
copy: `bondmaxsim.eval.corect` puts `extern/CoRECT/src` on `sys.path` and
calls CoRECT's own `corect.utils.evaluate_results` (pytrec_eval-based
NDCG/MAP/Recall/P/MRR at cutoffs).  Every `CoRECT_RC_metrics` value in the
Stage 5 results is produced by CoRECT code.  Their package has a real circular
import (`corect.utils` ↔ `corect.model_wrappers`); the adapter imports
`model_wrappers` first to break it (documented in `_import_evaluate_results`).

We deliberately do NOT route retrieval through CoRECT's evaluation pipeline
(`corect.cli.evaluate` → `eval_utils`), for three structural reasons:

1. **Single-vector by construction.** Their `AbstractModelWrapper` contract is
   one embedding per query/document scored with `cos_sim` inside their batched
   loop.  ColBERT MaxSim is multi-vector; our fused panel kernels, partitioned
   scan, faiss-rerank and PLAID arms cannot execute inside that loop — wiring
   a "ColBERT wrapper" into it would only benchmark their brute-force cos_sim
   path, not our methods.
2. **CUDA assumption.** `eval_utils._get_top_k` calls `scores.cuda()`; this
   project is scoped single-node CPU.
3. **Different purpose.** The pipeline exists to compare embedding
   *compression* methods (PQ/PCA/LSH registry) — orthogonal to our question.

The plan doc anticipated this split: Stage 5 lists CoRECT as reusable "after
adding or specifying a ColBERT/MaxSim wrapper".  The adapter IS that wrapper,
placed at the interface where the systems genuinely meet: our arms produce the
run (`query_id → doc_id → score`), CoRECT's evaluator judges it — which is
also how their own pipeline ends.

Gate: `bondmaxsim.eval.corect.corect_smoke_test` verifies CoRECT's metrics
agree with our independent ranx metrics (to CoRECT's own `round(…, 5)`) on a
synthetic fixture (unit test) AND on the real `dense_fused` run inside the
driver, before any RC metric is reported.

Not used and deferred with R7 (future work): CoRECT's CoRE corpus pools — the
controlled 100k–1M+ scale axis.  At BEIR scale their dataset utilities wrap
the same `BeIR/*` HuggingFace datasets we already load with verified IDs.

## Output

- `results/json/stage5_corect_e01_ir_evaluation_<dataset>.json` (runs merged
  across `mt`/`1t` tags)
- `results/figures/stage5_corect/e01_ir_evaluation_<dataset>.png`
  (quality-latency frontier)
