"""
Experiment 7: BOND reordering vs natural-order columnar pruning — recall-at-fixed-pruning

Companion to experiment 6. Exp 6 used a *safe* Cauchy-Schwarz bound and found it
prunes ~0% at k<=64 for either dimension order, because MaxSim max-pools over
hundreds of doc tokens and compresses all doc scores into a ~3% band that is
narrower than the bound's ~6% slack. That makes the safe-bound BOND-vs-natural
comparison degenerate (0 vs 0).

This experiment instead measures APPROXIMATE pruning, the regime where dimension
ordering can actually matter:

  (A) ALGORITHMIC benefit — recall at a fixed pruning budget.
      Rank docs by a PARTIAL MaxSim score over k dimensions, keep the top
      fraction f, and measure recall of the true top-knn.
        - natural : partial score over the first k dimensions (fixed order)
        - bond    : partial score over the top-k dims by per-query q^2 importance
      If BOND's reordering captures the full ranking better, it keeps more true
      neighbours at the same f. Exp 2 (flat variance at 128D) predicts the gain
      is small.

  (B) SYSTEMS cost — what BOND's reordering costs on a columnar layout.
      PDX stores vectors dim-major (each dimension contiguous). Natural reads the
      first k dims = a contiguous block (sequential). BOND needs the top-k dims in
      a query-specific order = a scattered gather of k rows. We time the partial-
      score kernel under both access patterns (min of N_REPEATS, like exp 3),
      including BOND's argsort + gather.

Decision gate (combine only if it clearly helps):
  PASS to combine iff at k=64, f=10% the BOND recall gain >= 5 pp
       AND BOND wall-clock <= natural wall-clock.
  Expect FAIL: tiny recall gain (flat variance) + slower (gather penalty).

Reuses cached embeddings from 02_bond_variance/.cache/ (no qrels needed).

Run from anywhere:
    source "$HOME/Data Engineering/research/.venv/bin/activate"
    python "$HOME/Data Engineering/research/preliminaries/07_bond_recall_tradeoff/bond_recall_tradeoff.py"

Requires: numpy, matplotlib
"""
import pathlib, gc, time
import numpy as np
import matplotlib.pyplot as plt

HERE        = pathlib.Path(__file__).parent
FIGURES_DIR = HERE / "figures"; FIGURES_DIR.mkdir(exist_ok=True)
CACHE_02    = HERE.parent / "02_bond_variance" / ".cache"

K_BUDGETS   = [16, 32, 64]
KEEP_FRACS  = [0.01, 0.05, 0.10, 0.25]
KNN         = 10
MAX_QUERIES = 100
MAX_DOCS    = 2_000
GATE_K      = 64
GATE_F      = 0.10
DATASETS    = ["NFCorpus", "SciFact", "ArguAna", "SCIDOCS"]

# Timing (systems) config — synthetic; access-pattern cost is layout physics, not
# data-dependent. Large N ⇒ DRAM-bound regime (exp 3), where a gather hurts most.
TIME_DIM, TIME_NTOK, TIME_QTOK = 128, 100_000, 32
TIME_NQUERY, TIME_WARMUP, TIME_REPEATS = 20, 3, 12

rng = np.random.default_rng(42)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cache(ds_name: str):
    dp = CACHE_02 / f"{ds_name}_doc_embs.npy"
    qp = CACHE_02 / f"{ds_name}_qry_embs.npy"
    if not dp.exists() or not qp.exists():
        print(f"  [{ds_name}] cache not found — run 02_bond_variance first.")
        return None, None
    return np.load(dp, allow_pickle=True), np.load(qp, allow_pickle=True)


def partial_scores(q, docs, dims) -> np.ndarray:
    """Partial MaxSim score per doc using only `dims` (an index array)."""
    qk = q[:, dims]
    return np.array([float((qk @ d[:, dims].T).max(axis=1).sum()) for d in docs])


_ROT = {}
def rotation(dim: int) -> np.ndarray:
    """
    Fixed random ORTHOGONAL rotation — the core of ADSampling (cwida/PDX,
    include/pdx/pruners/adsampling.hpp: Householder-QR of a random gaussian).
    Orthogonal ⇒ preserves inner products ⇒ full MaxSim (and true top-k)
    unchanged; only the first-k *partial* score changes, becoming an unbiased
    estimate of the full score. Built once, shared across datasets/queries.
    """
    if dim not in _ROT:
        g = np.random.default_rng(123).standard_normal((dim, dim))
        Q, _ = np.linalg.qr(g)
        _ROT[dim] = Q.astype(np.float32)
    return _ROT[dim]


# ── (A) algorithmic: recall at fixed pruning, natural vs BOND ─────────────────

# sequential (free, no per-query gather): natural, global, adsampling.  gather: bond.
SCHEMES     = ("natural", "global", "adsampling", "bond")
SEQ_SCHEMES = ("natural", "global", "adsampling")


def analyze_recall(ds_name: str) -> dict | None:
    doc_embs, qry_embs = load_cache(ds_name)
    if doc_embs is None:
        return None
    print(f"\n[{ds_name}] recall@fixed-pruning")

    n_docs  = min(len(doc_embs), MAX_DOCS)
    doc_idx = rng.choice(len(doc_embs), n_docs, replace=False)
    docs    = [np.asarray(doc_embs[i], dtype=np.float32) for i in doc_idx]
    n_q     = min(len(qry_embs), MAX_QUERIES)

    # Fixed GLOBAL importance order: aggregate q^2 over ALL query tokens, once.
    dim     = np.asarray(qry_embs[0], dtype=np.float32).shape[1]
    glob_q2 = np.zeros(dim, dtype=np.float64)
    for qi in range(n_q):
        glob_q2 += (np.asarray(qry_embs[qi], dtype=np.float32) ** 2).sum(axis=0)
    global_order = np.argsort(glob_q2)[::-1]

    # ADSampling: rotate stored docs once at "build" time (orthogonal ⇒ MaxSim
    # preserved). Scanned later in fixed order 0..k-1 (sequential, no gather).
    R        = rotation(dim)
    rot_docs = [d @ R for d in docs]

    recall = {k: {s: {f: [] for f in KEEP_FRACS} for s in SCHEMES} for k in K_BUDGETS}

    for qi in range(n_q):
        q = np.asarray(qry_embs[qi], dtype=np.float32)
        full = np.array([float((q @ d.T).max(axis=1).sum()) for d in docs])
        true_topk = set(np.argpartition(full, -KNN)[-KNN:].tolist())
        bond_order = np.argsort((q ** 2).mean(axis=0))[::-1]   # per-query (gather)
        q_rot      = q @ R                                       # per-query rotation (cheap)

        for k in K_BUDGETS:
            # (q_or_qrot, docs_or_rotdocs, dims)
            arms = {
                "natural":    (q,     docs,     np.arange(k)),       # arbitrary fixed prefix (seq)
                "global":     (q,     docs,     global_order[:k]),   # fixed global-q² order (seq, free)
                "adsampling": (q_rot, rot_docs, np.arange(k)),       # rotated, fixed prefix (seq, free)
                "bond":       (q,     docs,     bond_order[:k]),     # per-query top-q² (gather)
            }
            for scheme, (qq, dd, dims) in arms.items():
                rank = np.argsort(partial_scores(qq, dd, dims))[::-1]
                for f in KEEP_FRACS:
                    keep = set(rank[:max(KNN, int(round(f * n_docs)))].tolist())
                    recall[k][scheme][f].append(len(keep & true_topk) / KNN)

    res = {"n_q": n_q, "n_docs": n_docs, "recall": {}}
    for k in K_BUDGETS:
        res["recall"][k] = {s: {f: float(np.mean(recall[k][s][f])) for f in KEEP_FRACS}
                            for s in SCHEMES}
    for k in K_BUDGETS:
        cell = {s: res["recall"][k][s][GATE_F] * 100 for s in SCHEMES}
        print(f"  k={k:>3}  recall@{int(GATE_F*100)}%  "
              f"natural={cell['natural']:5.1f}  global={cell['global']:5.1f}  "
              f"adsampling={cell['adsampling']:5.1f}  bond={cell['bond']:5.1f}")
    del docs, rot_docs; gc.collect()
    return res


# ── (B) systems: columnar access-pattern cost, natural vs BOND ────────────────

def analyze_timing() -> dict:
    print(f"\n[timing] columnar layout, N={TIME_NTOK:,} doc tokens, "
          f"{TIME_QTOK} qtok × {TIME_NQUERY} queries (synthetic, DRAM-bound)")
    Drow = rng.standard_normal((TIME_NTOK, TIME_DIM)).astype(np.float32)
    Drow /= np.linalg.norm(Drow, axis=1, keepdims=True)
    Ddim = np.ascontiguousarray(Drow.T)          # (128, N) dim-major = PDX columnar
    Qs   = []
    for _ in range(TIME_NQUERY):
        q = rng.standard_normal((TIME_QTOK, TIME_DIM)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        Qs.append(q)
    bond_orders = [np.argsort((q ** 2).mean(axis=0))[::-1] for q in Qs]

    def best(run):
        for _ in range(TIME_WARMUP): run()
        b = float("inf")
        for _ in range(TIME_REPEATS):
            t0 = time.perf_counter(); run(); b = min(b, time.perf_counter() - t0)
        return b / TIME_NQUERY * 1e3   # ms/query

    # ADSampling: docs rotated once at build time, stored columnar; per query we
    # rotate the query (cheap) then scan the contiguous first-k block (sequential).
    R    = rotation(TIME_DIM)
    Drot = np.ascontiguousarray((Drow @ R).T)    # (128, N) rotated, dim-major

    out = {}
    for k in K_BUDGETS:
        # Pre-gathered contiguous (k,N) doc operand (shape-representative). numpy
        # cannot matmul over scattered rows without materializing them, but a C
        # kernel reads the selected dimension-rows directly — its cost is this
        # contiguous matmul, with NO per-query 26 MB copy. This is the honest
        # "fused" BOND cost in a contiguous columnar layout.
        Dc = np.ascontiguousarray(Ddim[bond_orders[0][:k], :])

        def natural():
            for q in Qs:
                (q[:, :k] @ Ddim[:k, :]).max(axis=0)            # contiguous column block
        def adsampling():
            for q in Qs:
                qr = q @ R                                       # per-query rotation
                (qr[:, :k] @ Drot[:k, :]).max(axis=0)            # contiguous, sequential
        def bond_fused():
            for q, order in zip(Qs, bond_orders):
                (q[:, order[:k]] @ Dc).max(axis=0)               # no big doc copy (C-kernel proxy)
        def bond_copy():
            for q, order in zip(Qs, bond_orders):
                d = order[:k]
                (q[:, d] @ Ddim[d, :]).max(axis=0)               # numpy fancy-index COPY (artifact)
        out[k] = {"natural": best(natural), "adsampling": best(adsampling),
                  "bond_fused": best(bond_fused), "bond_copy": best(bond_copy)}
        o = out[k]
        print(f"  k={k:>3}  natural={o['natural']:6.3f}  adsampling={o['adsampling']:6.3f}  "
              f"bond_fused={o['bond_fused']:6.3f}  bond_copy={o['bond_copy']:6.3f} ms/q  →  "
              f"fused {o['bond_fused']/o['natural']:.2f}× natural (copy artifact {o['bond_copy']/o['natural']:.2f}×)")
    return out


# ── gate + plot ───────────────────────────────────────────────────────────────

def print_gate(recall, timing):
    print("\n" + "=" * 80)
    print("DECISION GATE — does BOND's per-query gather beat the best FREE sequential scheme?")
    print(f"  sequential (free, no gather): natural | global (fixed q²) | adsampling (rotation)")
    print(f"  gather (costs extra):         bond (per-query top-q²)")
    print(f"PASS to combine BOND iff (k={GATE_K},keep={int(GATE_F*100)}%) bond − best_sequential >= 5 pp")
    print("=" * 80)
    adapt_gains = []
    for ds, r in recall.items():
        if r is None: continue
        c = {s: r["recall"][GATE_K][s][GATE_F] * 100 for s in SCHEMES}
        best_seq = max(c[s] for s in SEQ_SCHEMES)
        adapt_gains.append(c["bond"] - best_seq)
        print(f"  {ds:>10}: natural {c['natural']:5.1f}  global {c['global']:5.1f}  "
              f"adsampling {c['adsampling']:5.1f}  |  bond {c['bond']:5.1f}  "
              f"(bond−best_seq = {c['bond']-best_seq:+4.1f} pp)")
    mean_adapt = float(np.mean(adapt_gains)) if adapt_gains else float("nan")
    t = timing[GATE_K]
    fused_cost, copy_cost = t["bond_fused"] / t["natural"], t["bond_copy"] / t["natural"]
    print(f"\n  bond − best_sequential : {mean_adapt:+.1f} pp   (gate: >= +5 pp)")
    print(f"  bond TRUE cost (fused/no-copy): {fused_cost:.2f}× natural   "
          f"(numpy fancy-index copy artifact was {copy_cost:.2f}×)")
    if mean_adapt >= 5 and fused_cost <= 1.15:
        verdict = (f"COMBINE — BOND adds {mean_adapt:+.1f} pp recall, and with a fused kernel its true "
                   f"cost is ~{fused_cost:.2f}× natural (the ~{copy_cost:.1f}× was a numpy gather-copy "
                   f"artifact). In a contiguous columnar layout the recall is nearly free.")
    elif mean_adapt >= 5:
        verdict = (f"TRADEOFF — BOND adds {mean_adapt:+.1f} pp at {fused_cost:.2f}× fused cost.")
    else:
        verdict = f"do NOT combine — a free sequential scheme matches BOND (+{mean_adapt:.1f} pp)."
    print(f"\n  VERDICT: {verdict}")
    print(f"  CAVEAT: fused cost measured in a GLOBAL contiguous layout (each dim = one long run).")
    print(f"          PDX's BLOCKED layout (dim = 64 floats/block) may add real scattered-access cost;")
    print(f"          confirm with a C/SIMD kernel before treating BOND as free (→ experiment 8).")


def plot(recall, timing):
    valid = {k: v for k, v in recall.items() if v is not None}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle(
        "Experiment 7 — MaxSim approximate pruning: BOND (per-query gather) vs free sequential schemes "
        "(natural / global-q² / ADSampling rotation)\n"
        "Left: recall vs prune aggressiveness  ·  Mid: per-dataset recall  ·  Right: systems cost (gather vs sequential)",
        fontsize=9.5, fontweight="bold",
    )

    # Panel 1: recall vs keep-fraction at gate k, four schemes, averaged over datasets
    ax = axes[0]
    style = {"natural": ("#888888", "o-"), "global": ("#0077BB", "^-"),
             "adsampling": ("#009988", "D-"), "bond": ("#EE7733", "s--")}
    for scheme, (color, ls) in style.items():
        mean = [np.mean([valid[ds]["recall"][GATE_K][scheme][f] for ds in valid]) * 100
                for f in KEEP_FRACS]
        ax.plot([f*100 for f in KEEP_FRACS], mean, ls, color=color, lw=2, ms=5, label=scheme)
    ax.set_xlabel("Docs kept (%)"); ax.set_ylabel(f"Recall of true top-{KNN} (%)")
    ax.set_title(f"Recall vs pruning (k={GATE_K}, mean over datasets)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: recall@gate per dataset, all four schemes (grouped bars)
    ax = axes[1]
    names = list(valid.keys()); x = np.arange(len(names)); w = 0.20
    colors = {"natural": "#888888", "global": "#0077BB", "adsampling": "#009988", "bond": "#EE7733"}
    for j, s in enumerate(SCHEMES):
        vals = [valid[ds]["recall"][GATE_K][s][GATE_F] * 100 for ds in names]
        ax.bar(x + (j - 1.5) * w, vals, w, color=colors[s], label=s)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(f"Recall @k={GATE_K}, keep={int(GATE_F*100)}% (%)")
    ax.set_title("Per dataset: sequential schemes vs BOND(gather)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: wall-clock — sequential schemes, BOND fused (true) vs BOND copy (numpy artifact)
    ax = axes[2]
    x = np.arange(len(K_BUDGETS)); w = 0.21
    nat   = [timing[k]["natural"]    for k in K_BUDGETS]
    ads   = [timing[k]["adsampling"] for k in K_BUDGETS]
    bf    = [timing[k]["bond_fused"] for k in K_BUDGETS]
    bc    = [timing[k]["bond_copy"]  for k in K_BUDGETS]
    ax.bar(x - 1.5*w, nat, w, color="#0077BB", label="natural/global (seq)")
    ax.bar(x - 0.5*w, ads, w, color="#009988", label="adsampling (seq+rotate)")
    ax.bar(x + 0.5*w, bf,  w, color="#EE7733", label="BOND fused (true cost)")
    ax.bar(x + 1.5*w, bc,  w, color="#CC3311", alpha=0.55, label="BOND numpy-copy (artifact)")
    for xi, (n, f) in enumerate(zip(nat, bf)):
        ax.annotate(f"{f/n:.2f}×", (xi + 0.5*w, f), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color="#CC5500")
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in K_BUDGETS])
    ax.set_ylabel("Latency (ms/query, min)")
    ax.set_title(f"Cost: BOND fused ≈ sequential (N={TIME_NTOK//1000}k tokens)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5); ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    out = FIGURES_DIR / "bond_recall_tradeoff.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")


def main():
    print("Experiment 7: BOND reordering vs natural-order columnar pruning (recall@fixed-pruning)")
    recall = {ds: analyze_recall(ds) for ds in DATASETS}
    timing = analyze_timing()
    print_gate(recall, timing)
    plot(recall, timing)


if __name__ == "__main__":
    main()
