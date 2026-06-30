"""
Experiment 0: Does ColBERT's variance concentrate fast enough after dimension
reordering for BOND-style pruning to be effective?

Changes from v1:
  - Multiple BEIR datasets (FiQA, SCIDOCS, NFCorpus, ArguAna, SciFact)
    to test generalization across domains. Encodes one at a time to fit 12 GB.
  - Normalization check on the embeddings.
  - Compares two dimension orderings:
        (a) global variance ordering  (sort dims by corpus variance)
        (b) query-adaptive ordering   (sort dims by q_i^2 per query, BOND-style)
  - Faster MaxSim via padded batch matmul (PyTorch).
  - Subsamples queries and docs for rank correlation to stay tractable.

Run from anywhere:
    source "$HOME/Data Engineering/research/.venv/bin/activate"
    python "$HOME/Data Engineering/research/preliminaries/02_bond_variance/bond_variance_analysis.py"

Requires: pylate, datasets, numpy, scipy, matplotlib, torch
"""
import pathlib, gc, time
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
HERE       = pathlib.Path(__file__).parent
FIGURES_DIR = HERE / "figures";  FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR   = HERE / ".cache";   CACHE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"

# BEIR datasets ordered by corpus size (ascending). All fit in 12 GB one at a time.
# Adjust this list depending on how much time/memory you have.
DATASETS = {
    # name       : (beir_hf_name,   max_docs, max_queries_for_rankcorr)
    "NFCorpus"   : ("BeIR/nfcorpus", None,     200),
    "SciFact"    : ("BeIR/scifact",  None,     200),
    "ArguAna"    : ("BeIR/arguana",  None,     200),
    "SCIDOCS"    : ("BeIR/scidocs",  None,     200),
    "FiQA"       : ("BeIR/fiqa",     None,     200),
}

# For rank correlation we cap the number of docs to keep runtime reasonable.
MAX_DOCS_RANKCORR = 10_000

BUDGETS    = [1, 2, 4, 8, 16, 32, 64, 128]
BATCH_SIZE = 64   # encoding batch size – keep low for 12 GB RAM

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_beir_dataset(hf_name: str, max_docs: int | None, max_queries: int | None):
    """Load a BEIR dataset from HuggingFace. Returns (docs, queries) as lists of str."""
    from datasets import load_dataset
    corpus_ds  = load_dataset(hf_name, "corpus",  split="corpus")
    queries_ds = load_dataset(hf_name, "queries", split="queries")

    n_docs = len(corpus_ds) if max_docs is None else min(max_docs, len(corpus_ds))
    n_q    = len(queries_ds) if max_queries is None else min(max_queries, len(queries_ds))

    docs    = [row["text"] for row in corpus_ds.select(range(n_docs))]
    queries = [row["text"] for row in queries_ds.select(range(n_q))]
    return docs, queries


def encode_or_load(name: str, docs: list, queries: list):
    """Encode with the ColBERT model, or load from cache."""
    doc_cache = CACHE_DIR / f"{name}_doc_embs.npy"
    qry_cache = CACHE_DIR / f"{name}_qry_embs.npy"

    if doc_cache.exists() and qry_cache.exists():
        print(f"  Loading cached embeddings for {name} …")
        doc_embs = np.load(doc_cache, allow_pickle=True)
        qry_embs = np.load(qry_cache, allow_pickle=True)
        return doc_embs, qry_embs

    from pylate import models
    print(f"  Encoding {name} with {MODEL_NAME} ({len(docs)} docs, {len(queries)} queries) …")
    model = models.ColBERT(MODEL_NAME)

    # Encode in batches to limit peak memory
    doc_embs = model.encode(
        docs, is_query=False, batch_size=BATCH_SIZE, show_progress_bar=True,
    )
    qry_embs = model.encode(
        queries, is_query=True, batch_size=BATCH_SIZE, show_progress_bar=True,
    )

    np.save(doc_cache, np.array(doc_embs, dtype=object))
    np.save(qry_cache, np.array(qry_embs, dtype=object))
    print(f"  Cached to {doc_cache}")

    # Free the model to reclaim memory before next dataset
    del model; gc.collect()
    return doc_embs, qry_embs


def check_normalization(all_vecs: np.ndarray, name: str):
    """Verify that embeddings are L2-normalized (needed for Cauchy-Schwarz bounds)."""
    norms = np.linalg.norm(all_vecs, axis=1)
    print(f"  [{name}] Embedding norms: mean={norms.mean():.4f}, "
          f"std={norms.std():.6f}, min={norms.min():.4f}, max={norms.max():.4f}")
    if norms.std() > 0.01 or abs(norms.mean() - 1.0) > 0.05:
        print(f"  ⚠  WARNING: embeddings do NOT appear L2-normalized! "
              f"Cauchy-Schwarz bounds will not apply directly.")
        return False
    return True


def maxsim_partial_batch(q_embs_k, doc_embs_k, max_docs: int):
    """
    MaxSim scores using only the first k (already-sliced) dimensions.
    Uses numpy batch matmul for speed.

    q_embs_k   : list of (n_qtokens, k) arrays
    doc_embs_k : list of (n_dtokens, k) arrays  [only first max_docs]

    Returns: (n_queries, n_docs) score matrix
    """
    n_docs = min(len(doc_embs_k), max_docs)
    scores = np.empty((len(q_embs_k), n_docs), dtype=np.float32)
    for d_idx in range(n_docs):
        d = doc_embs_k[d_idx]  # (D_tokens, k)
        for q_idx, q in enumerate(q_embs_k):
            # q: (Q_tokens, k),  d: (D_tokens, k)
            sim = q @ d.T          # (Q_tokens, D_tokens)
            scores[q_idx, d_idx] = sim.max(axis=1).sum()
    return scores

# ── Main analysis per dataset ─────────────────────────────────────────────────

def analyze_dataset(name: str, doc_embs, qry_embs, max_docs_rc: int, max_queries_rc: int):
    """
    Returns a dict with:
      - cum_var          : (128,) cumulative explained variance (global ordering)
      - global_corrs     : (n_queries, n_budgets) Spearman rho (global ordering)
      - adaptive_corrs   : (n_queries, n_budgets) Spearman rho (query-adaptive ordering)
      - is_normalized    : bool
    """
    DIM = doc_embs[0].shape[1]
    assert DIM == 128, f"Expected 128 dims, got {DIM}"

    # Stack all token vectors for variance analysis
    all_vecs = np.vstack(doc_embs)
    is_norm = check_normalization(all_vecs, name)

    # ── Per-dimension variance (global ordering) ──────────────────────────────
    dim_var    = all_vecs.var(axis=0)                      # (128,)
    global_idx = np.argsort(dim_var)[::-1]                 # high→low variance
    cum_var    = np.cumsum(dim_var[global_idx]) / dim_var.sum()

    # Free the big matrix
    del all_vecs; gc.collect()

    # ── Rank correlation at partial-dim budgets ───────────────────────────────
    n_q  = min(len(qry_embs), max_queries_rc)
    n_d  = min(len(doc_embs), max_docs_rc)

    # Pre-slice doc embeddings for global ordering
    doc_global = [e[:, global_idx] for e in doc_embs[:n_d]]

    # --- Global ordering ---
    print(f"  [{name}] Rank correlations (global variance ordering) …")
    qry_global = [e[:, global_idx] for e in qry_embs[:n_q]]
    full_scores = maxsim_partial_batch(
        [q[:, :DIM] for q in qry_global], doc_global, n_d
    )

    global_corrs = np.zeros((n_q, len(BUDGETS)))
    for b_idx, k in enumerate(BUDGETS):
        partial_scores = maxsim_partial_batch(
            [q[:, :k] for q in qry_global], [d[:, :k] for d in doc_global], n_d
        )
        for q_idx in range(n_q):
            rho, _ = spearmanr(partial_scores[q_idx], full_scores[q_idx])
            global_corrs[q_idx, b_idx] = rho
        print(f"    k={k:3d}  mean ρ={global_corrs[:, b_idx].mean():.4f}")

    # --- Query-adaptive ordering (BOND-style: sort dims by q_i^2 per query) ---
    print(f"  [{name}] Rank correlations (query-adaptive q² ordering) …")

    # Full 128D scores are the same regardless of ordering (just reordered dims)
    # But we need to recompute with query-specific ordering
    adaptive_corrs = np.zeros((n_q, len(BUDGETS)))
    for q_idx in range(n_q):
        q_raw = qry_embs[q_idx]                            # (Q_tokens, 128)
        # Mean q_i^2 across query tokens → importance of each dimension
        q_importance = (q_raw ** 2).mean(axis=0)            # (128,)
        q_order = np.argsort(q_importance)[::-1]            # dims sorted by query importance

        q_reordered = q_raw[:, q_order]
        docs_reordered = [e[:, q_order] for e in doc_embs[:n_d]]

        # Full score (all 128 dims)
        full_q = maxsim_partial_batch([q_reordered], docs_reordered, n_d)[0]

        for b_idx, k in enumerate(BUDGETS):
            partial_q = maxsim_partial_batch(
                [q_reordered[:, :k]], [d[:, :k] for d in docs_reordered], n_d
            )[0]
            rho, _ = spearmanr(partial_q, full_q)
            adaptive_corrs[q_idx, b_idx] = rho

        if (q_idx + 1) % 50 == 0 or q_idx == n_q - 1:
            print(f"    query {q_idx+1}/{n_q} done")

    for b_idx, k in enumerate(BUDGETS):
        print(f"    k={k:3d}  mean ρ={adaptive_corrs[:, b_idx].mean():.4f}")

    return {
        "cum_var":        cum_var,
        "global_corrs":   global_corrs,
        "adaptive_corrs": adaptive_corrs,
        "is_normalized":  is_norm,
        "dim":            DIM,
        "n_docs":         len(doc_embs),
        "n_queries":      n_q,
        "n_docs_rc":      n_d,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict):
    """One row per dataset: [cumulative variance | global vs adaptive rank corr]."""
    n_ds = len(results)
    fig, axes = plt.subplots(n_ds, 2, figsize=(13, 3.5 * n_ds))
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "ColBERT dimension variance & MaxSim rank fidelity at partial-dim budgets\n"
        f"Model: {MODEL_NAME}  ·  Orderings: global-variance vs query-adaptive (q²)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for row, (ds_name, res) in enumerate(results.items()):
        ax1, ax2 = axes[row]
        cum_var = res["cum_var"]
        g_corr  = res["global_corrs"]
        a_corr  = res["adaptive_corrs"]

        # --- Panel 1: cumulative variance ---
        dims = np.arange(1, 129)
        ax1.plot(dims, cum_var * 100, color="#AA3377", linewidth=2)
        for mark in [8, 16, 32, 64]:
            ax1.axvline(mark, ls="--", lw=0.8, color="#888", alpha=0.6)
            ax1.text(mark + 1, 5, f"{mark}d\n{cum_var[mark-1]*100:.0f}%",
                     fontsize=7, color="#555", va="bottom")
        ax1.set_xlabel("Dims (sorted by variance)")
        ax1.set_ylabel("Cum. explained var (%)")
        ax1.set_title(f"{ds_name}  ({res['n_docs']} docs)", fontsize=9, fontweight="bold")
        ax1.set_xlim(1, 128); ax1.set_ylim(0, 105)
        ax1.grid(ls="--", lw=0.4, alpha=0.5); ax1.set_axisbelow(True)
        ax1.spines[["top", "right"]].set_visible(False)

        # --- Panel 2: rank correlation comparison ---
        g_mean = g_corr.mean(axis=0)
        a_mean = a_corr.mean(axis=0)
        g_p25  = np.percentile(g_corr, 25, axis=0)
        g_p75  = np.percentile(g_corr, 75, axis=0)
        a_p25  = np.percentile(a_corr, 25, axis=0)
        a_p75  = np.percentile(a_corr, 75, axis=0)

        ax2.plot(BUDGETS, g_mean, "o-", lw=2, ms=4, color="#009988",  label="Global variance order")
        ax2.fill_between(BUDGETS, g_p25, g_p75, alpha=0.15, color="#009988")
        ax2.plot(BUDGETS, a_mean, "s-", lw=2, ms=4, color="#EE7733",  label="Query-adaptive q² order")
        ax2.fill_between(BUDGETS, a_p25, a_p75, alpha=0.15, color="#EE7733")

        for k_val, g_rho, a_rho in zip(BUDGETS, g_mean, a_mean):
            ax2.annotate(f"{g_rho:.2f}", (k_val, g_rho), textcoords="offset points",
                         xytext=(-12, 6), fontsize=6.5, color="#009988")
            ax2.annotate(f"{a_rho:.2f}", (k_val, a_rho), textcoords="offset points",
                         xytext=(4, -10), fontsize=6.5, color="#EE7733")

        ax2.axhline(0.95, ls=":", lw=1, color="#CC3311", alpha=0.7, label="ρ = 0.95")
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(BUDGETS)
        ax2.set_xticklabels([str(k) for k in BUDGETS])
        ax2.set_xlabel("Dimensions used")
        ax2.set_ylabel("Spearman ρ vs full 128-dim")
        ax2.set_title(
            f"{ds_name}  (rank corr: {res['n_queries']} queries × {res['n_docs_rc']} docs)",
            fontsize=9, fontweight="bold",
        )
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=7, framealpha=0.8, loc="lower right")
        ax2.grid(ls="--", lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
        ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "bond_variance_analysis_multi.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}

    for ds_name, (hf_name, max_docs, max_q) in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        t0 = time.time()
        docs, queries = load_beir_dataset(hf_name, max_docs, max_q)
        print(f"  {len(docs)} docs, {len(queries)} queries")

        doc_embs, qry_embs = encode_or_load(ds_name, docs, queries)
        del docs, queries; gc.collect()

        res = analyze_dataset(
            ds_name, doc_embs, qry_embs,
            max_docs_rc=MAX_DOCS_RANKCORR,
            max_queries_rc=max_q,
        )
        results[ds_name] = res

        elapsed = time.time() - t0
        print(f"  [{ds_name}] Done in {elapsed:.0f}s")

        # Free memory before next dataset
        del doc_embs, qry_embs; gc.collect()

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Dataset':>12} {'Norm?':>6}"
    for k in BUDGETS:
        header += f"  {k:>3}d_G  {k:>3}d_Q"
    print(header)
    print("-" * len(header))

    for ds_name, res in results.items():
        row = f"{ds_name:>12} {'  ✓' if res['is_normalized'] else '  ✗':>6}"
        g_mean = res["global_corrs"].mean(axis=0)
        a_mean = res["adaptive_corrs"].mean(axis=0)
        for b_idx in range(len(BUDGETS)):
            row += f"  {g_mean[b_idx]:.3f}  {a_mean[b_idx]:.3f}"
        print(row)

    print(f"\nG = global variance ordering, Q = query-adaptive q² ordering")
    print(f"Values are mean Spearman ρ of MaxSim ranking vs full 128-dim ground truth.\n")

    # ── Key decision numbers ──────────────────────────────────────────────
    for ds_name, res in results.items():
        cum_var = res["cum_var"]
        g_mean  = res["global_corrs"].mean(axis=0)
        a_mean  = res["adaptive_corrs"].mean(axis=0)
        print(f"[{ds_name}]")
        for b_idx, k in enumerate(BUDGETS):
            pct = cum_var[min(k, 128) - 1] * 100
            print(f"  {k:3d} dims — cum.var {pct:5.1f}%  |  "
                  f"ρ_global={g_mean[b_idx]:.4f}  ρ_adaptive={a_mean[b_idx]:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_results(results)


if __name__ == "__main__":
    main()
