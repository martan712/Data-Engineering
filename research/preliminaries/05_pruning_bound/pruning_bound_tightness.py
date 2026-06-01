"""
Experiment 3: Pruning bound tightness

For a sample of queries, compute the ratio UB(k) / Score(Q, D) at
k = 16, 32, 64, 128 for both relevant and irrelevant documents.

UB_ij(k) = q_i[:k]·d_j[:k]  +  ||q_i[k:]|| · ||d_j[k:]||   (Cauchy-Schwarz)
UB_d(k)  = Σ_i  max_j  UB_ij(k)   (document-level MaxSim upper bound)

Reuses cached embeddings from 02_bond_variance/.cache/.
Qrels loaded via ir_datasets (beir/<dataset>/test).

Decision gate: median UB(k)/Score < 1.2 at k=64 for irrelevant documents.
               If so, document-level pruning will be effective.

Run from anywhere:
    source "$HOME/Data Engineering/research/.venv/bin/activate"
    python "$HOME/Data Engineering/research/preliminaries/05_pruning_bound/pruning_bound_tightness.py"

Requires: numpy, scipy, matplotlib, ir_datasets
"""
import pathlib, gc
import numpy as np
import matplotlib.pyplot as plt
import ir_datasets

HERE        = pathlib.Path(__file__).parent
FIGURES_DIR = HERE / "figures"; FIGURES_DIR.mkdir(exist_ok=True)

CACHE_02 = HERE.parent / "02_bond_variance" / ".cache"

K_BUDGETS    = [16, 32, 64, 128]
MAX_QUERIES  = 100   # qrels-matched queries per dataset
MAX_DOCS     = 2_000 # docs to score per query (random sample from corpus)

# Datasets available in 02_bond_variance cache that also have BEIR qrels via ir_datasets
DATASETS = {
    "NFCorpus": "beir/nfcorpus/test",
    "SciFact":  "beir/scifact/test",
    "SCIDOCS":  "beir/scidocs/test",
    "FiQA":     "beir/fiqa/test",
}


def load_qrels(ir_name: str) -> dict[str, dict[str, int]]:
    """Returns {query_id: {doc_id: relevance}} from ir_datasets."""
    ds    = ir_datasets.load(ir_name)
    qrels = {}
    for q in ds.qrels_iter():
        qrels.setdefault(q.query_id, {})[q.doc_id] = q.relevance
    return qrels


def load_cache(ds_name: str):
    doc_path = CACHE_02 / f"{ds_name}_doc_embs.npy"
    qry_path = CACHE_02 / f"{ds_name}_qry_embs.npy"
    if not doc_path.exists() or not qry_path.exists():
        print(f"  [{ds_name}] Cache not found — run 02_bond_variance first.")
        return None, None
    doc_embs = np.load(doc_path, allow_pickle=True)
    qry_embs = np.load(qry_path, allow_pickle=True)
    return doc_embs, qry_embs


def maxsim_full(q_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """True MaxSim score: Σ_i max_j (q_i · d_j)."""
    return float((q_emb @ doc_emb.T).max(axis=1).sum())


def ub_maxsim(q_emb: np.ndarray, doc_emb: np.ndarray, k: int) -> float:
    """
    Document-level MaxSim upper bound at dimension budget k.
    UB_ij(k) = q_i[:k]·d_j[:k] + ||q_i[k:]|| · ||d_j[k:]||
    UB_d(k)  = Σ_i max_j UB_ij(k)
    """
    partial = q_emb[:, :k] @ doc_emb[:, :k].T   # (Q_tok, D_tok)
    resid_q = np.sqrt(np.maximum(0.0, 1.0 - (q_emb[:, :k] ** 2).sum(axis=1)))  # (Q_tok,)
    resid_d = np.sqrt(np.maximum(0.0, 1.0 - (doc_emb[:, :k] ** 2).sum(axis=1)))  # (D_tok,)
    upper   = partial + resid_q[:, None] * resid_d[None, :]  # (Q_tok, D_tok)
    return float(upper.max(axis=1).sum())


def analyze_dataset(ds_name: str, ir_name: str) -> dict | None:
    print(f"\n[{ds_name}]")
    doc_embs, qry_embs = load_cache(ds_name)
    if doc_embs is None:
        return None

    try:
        qrels = load_qrels(ir_name)
    except Exception as e:
        print(f"  Could not load qrels via ir_datasets: {e}")
        return None

    # Map string IDs to integer indices
    # ir_datasets uses string IDs; our cache uses HF row order (0, 1, 2, …)
    # We load query IDs from ir_datasets to align with cached query embeddings
    ir_ds      = ir_datasets.load(ir_name)
    ir_qids    = [q.query_id for q in ir_ds.queries_iter()]
    ir_docids  = [d.doc_id   for d in ir_ds.docs_iter()]
    qid_to_idx = {qid: i for i, qid in enumerate(ir_qids)}
    did_to_idx = {did: i for i, did in enumerate(ir_docids)}

    # Sample queries that have qrels and cached embeddings
    valid_qids = [qid for qid in list(qrels.keys())[:MAX_QUERIES]
                  if qid in qid_to_idx and qid_to_idx[qid] < len(qry_embs)]
    if not valid_qids:
        print(f"  No matching queries found.")
        return None
    print(f"  {len(valid_qids)} queries with qrels")

    rng = np.random.default_rng(42)
    n_docs = min(len(doc_embs), MAX_DOCS)
    doc_indices = rng.choice(len(doc_embs), n_docs, replace=False)

    ratios = {k: {"relevant": [], "irrelevant": []} for k in K_BUDGETS}

    for qid in valid_qids:
        q_idx = qid_to_idx[qid]
        if q_idx >= len(qry_embs):
            continue
        q_emb   = qry_embs[q_idx]
        rel_ids = set(qrels[qid].keys())

        for d_local_idx, d_global_idx in enumerate(doc_indices):
            d_emb = doc_embs[d_global_idx]
            # Map global doc index back to ir_datasets doc_id
            if d_global_idx < len(ir_docids):
                did = ir_docids[d_global_idx]
                is_relevant = did in rel_ids
            else:
                is_relevant = False

            true_score = maxsim_full(q_emb, d_emb)
            if true_score < 1e-6:
                continue  # avoid div-by-zero for degenerate docs

            for k in K_BUDGETS:
                ub    = ub_maxsim(q_emb, d_emb, k)
                ratio = ub / true_score
                bucket = "relevant" if is_relevant else "irrelevant"
                ratios[k][bucket].append(ratio)

    # Summary
    for k in K_BUDGETS:
        rel_med   = np.median(ratios[k]["relevant"])   if ratios[k]["relevant"]   else float("nan")
        irrel_med = np.median(ratios[k]["irrelevant"]) if ratios[k]["irrelevant"] else float("nan")
        n_rel     = len(ratios[k]["relevant"])
        n_irrel   = len(ratios[k]["irrelevant"])
        gate = "✓" if irrel_med < 1.2 else "✗"
        print(f"  k={k:>3}  rel med={rel_med:.3f} (n={n_rel})  "
              f"irrel med={irrel_med:.3f} (n={n_irrel})  gate@1.2: {gate}")

    return ratios


def print_gate(all_ratios: dict):
    print("\n" + "=" * 70)
    print("DECISION GATE: median UB/Score at k=64 for irrelevant docs")
    print("Gate passed if < 1.2 (pruning effective)")
    print("=" * 70)
    for ds_name, ratios in all_ratios.items():
        if ratios is None:
            continue
        k64 = ratios[64]["irrelevant"]
        if not k64:
            print(f"  {ds_name}: no data")
            continue
        med = np.median(k64)
        verdict = "PASS" if med < 1.2 else ("MARGINAL" if med < 1.5 else "FAIL")
        print(f"  {ds_name}: median UB/Score at k=64 = {med:.4f} → {verdict}")


def plot(all_ratios: dict):
    valid = {k: v for k, v in all_ratios.items() if v is not None}
    if not valid:
        print("No valid results to plot.")
        return

    n_ds  = len(valid)
    n_k   = len(K_BUDGETS)
    fig, axes = plt.subplots(n_ds, n_k, figsize=(3.2 * n_k, 3.5 * n_ds), sharey="row")
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Pruning bound tightness: UB(k) / Score(Q,D) for relevant vs irrelevant docs\n"
        "Cauchy-Schwarz MaxSim upper bound at each dimension budget k",
        fontsize=10, fontweight="bold", y=1.01,
    )

    colors = {"relevant": "#009988", "irrelevant": "#EE7733"}

    for row, (ds_name, ratios) in enumerate(valid.items()):
        for col, k in enumerate(K_BUDGETS):
            ax = axes[row, col]
            data_rel   = np.array(ratios[k]["relevant"])
            data_irrel = np.array(ratios[k]["irrelevant"])

            parts = ax.violinplot(
                [d for d in [data_rel, data_irrel] if len(d)],
                positions=range(sum([len(data_rel) > 0, len(data_irrel) > 0])),
                showmedians=True, showextrema=False,
            )
            labels_used = []
            for i, (d, label) in enumerate(
                [(data_rel, "relevant"), (data_irrel, "irrelevant")]
            ):
                if len(d) == 0:
                    continue
                idx = labels_used.__len__()
                parts["bodies"][idx].set_facecolor(colors[label])
                parts["bodies"][idx].set_alpha(0.7)
                labels_used.append(label)

            ax.axhline(1.0, ls="-",  lw=0.8, color="#888", alpha=0.5)
            ax.axhline(1.2, ls="--", lw=1.0, color="#CC3311", alpha=0.7,
                       label="Gate 1.2" if row == 0 and col == 0 else "")

            ax.set_xticks(range(len(labels_used)))
            ax.set_xticklabels(labels_used, fontsize=7)
            ax.set_title(f"{ds_name}  k={k}", fontsize=8, fontweight="bold")
            ax.set_ylabel("UB / Score" if col == 0 else "")
            ax.set_ylim(0.95, min(3.0, max(
                np.percentile(data_rel, 95) if len(data_rel) else 1.5,
                np.percentile(data_irrel, 95) if len(data_irrel) else 1.5,
            ) * 1.1))
            ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "pruning_bound_tightness.png"
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")


def main():
    all_ratios = {}
    for ds_name, ir_name in DATASETS.items():
        all_ratios[ds_name] = analyze_dataset(ds_name, ir_name)
        gc.collect()

    print_gate(all_ratios)
    plot(all_ratios)


if __name__ == "__main__":
    main()
