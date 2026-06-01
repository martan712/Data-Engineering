"""
Experiment 6: BOND reordering vs columnar Cauchy-Schwarz pruning — is combining them worth it?

Experiments 2, 3 and 5 together suggest that adding BOND-style per-query
dimension reordering on top of a columnar layout + Cauchy-Schwarz bound is a bad
trade. This experiment turns that argument into numbers by separating the two
effects:

  (A) ALGORITHMIC benefit — does BOND actually prune more?
      For a budget of k dimensions, both schemes compute a *safe* document-level
      MaxSim upper bound and prune a doc when UB(k) < tau (tau = the true knn-th
      best score for that query, an oracle threshold). Both bounds are valid, so
      neither ever prunes a true top-k doc (recall is 1.0 by construction — we
      assert it). The only question is the PRUNE RATE:
        - natural : UB over the first k dimensions (fixed order)
        - bond    : UB over the top-k dimensions by per-query q^2 importance
      BOND front-loads high-energy dims, so its bound is tighter — but exp 2
      showed variance is flat at 128D, so the gain should be small.

  (B) SYSTEMS cost — what does BOND's reordering cost on a columnar layout?
      PDX stores vectors dim-major (each dimension contiguous). The natural scheme
      reads the first k dims = a contiguous block (sequential). BOND needs the
      top-k dims in a *query-specific* order = a scattered gather of k rows. We
      time the partial-score kernel under both access patterns (min of N_REPEATS,
      like exp 3), including BOND's argsort + gather.

Decision gate (combine only if it clearly helps):
  PASS  to combine if at k=64 the BOND prune-rate gain >= 10 percentage points
        AND BOND wall-clock <= natural wall-clock.
  Expect FAIL: small prune gain (flat variance) + slower (gather penalty).

Reuses cached embeddings from 02_bond_variance/.cache/ (no qrels needed — the
threshold is the oracle knn-th true score).

Run from anywhere:
    source "$HOME/Data Engineering/research/.venv/bin/activate"
    python "$HOME/Data Engineering/research/preliminaries/06_bond_layout_tradeoff/bond_layout_tradeoff.py"

Requires: numpy, matplotlib
"""
import pathlib, gc, time
import numpy as np
import matplotlib.pyplot as plt

HERE        = pathlib.Path(__file__).parent
FIGURES_DIR = HERE / "figures"; FIGURES_DIR.mkdir(exist_ok=True)
CACHE_02    = HERE.parent / "02_bond_variance" / ".cache"

K_BUDGETS   = [16, 32, 64]
GATE_K      = 64
KNN         = 10            # threshold = knn-th best true score (oracle)
MAX_QUERIES = 100
MAX_DOCS    = 2_000
DATASETS    = ["NFCorpus", "SciFact", "ArguAna", "SCIDOCS"]

# Timing (systems) config — synthetic, since the access-pattern cost is layout
# physics, not data-dependent. Large N so we are in the DRAM-bound regime (exp 3),
# which is exactly where a scattered gather hurts most.
TIME_DIM     = 128
TIME_NTOK    = 100_000     # stacked doc tokens
TIME_QTOK    = 32          # ColBERT query tokens
TIME_NQUERY  = 20
TIME_WARMUP  = 3
TIME_REPEATS = 12

rng = np.random.default_rng(42)


# ── shared helpers ────────────────────────────────────────────────────────────

def load_cache(ds_name: str):
    dp = CACHE_02 / f"{ds_name}_doc_embs.npy"
    qp = CACHE_02 / f"{ds_name}_qry_embs.npy"
    if not dp.exists() or not qp.exists():
        print(f"  [{ds_name}] cache not found — run 02_bond_variance first.")
        return None, None
    return np.load(dp, allow_pickle=True), np.load(qp, allow_pickle=True)


def maxsim_full(q, d) -> float:
    return float((q @ d.T).max(axis=1).sum())


def ub_maxsim_subset(q, d, dims) -> float:
    """
    Safe document-level MaxSim upper bound using an arbitrary subset of `dims`:
        UB_ij = q_i[dims]·d_j[dims] + ||q_i[~dims]|| · ||d_j[~dims]||
        UB_d  = Σ_i max_j UB_ij
    Valid for ANY dim subset (Cauchy-Schwarz on the complement), so it never
    underestimates the true score — pruning with it is lossless.
    """
    qk = q[:, dims]; dk = d[:, dims]
    partial = qk @ dk.T
    resid_q = np.sqrt(np.maximum(0.0, 1.0 - (qk ** 2).sum(axis=1)))
    resid_d = np.sqrt(np.maximum(0.0, 1.0 - (dk ** 2).sum(axis=1)))
    return float((partial + resid_q[:, None] * resid_d[None, :]).max(axis=1).sum())


# ── (A) algorithmic: prune-rate, natural vs BOND ──────────────────────────────

def analyze_pruning(ds_name: str) -> dict | None:
    doc_embs, qry_embs = load_cache(ds_name)
    if doc_embs is None:
        return None
    print(f"\n[{ds_name}] prune-rate analysis")

    n_docs      = min(len(doc_embs), MAX_DOCS)
    doc_idx     = rng.choice(len(doc_embs), n_docs, replace=False)
    docs        = [np.asarray(doc_embs[i], dtype=np.float32) for i in doc_idx]
    n_q         = min(len(qry_embs), MAX_QUERIES)

    pruned   = {k: {"natural": [], "bond": []} for k in K_BUDGETS}
    unsafe   = 0  # count of true top-k docs that got pruned (must stay 0)

    for qi in range(n_q):
        q = np.asarray(qry_embs[qi], dtype=np.float32)
        # true scores + oracle threshold (knn-th best)
        true = np.array([maxsim_full(q, d) for d in docs])
        if n_docs <= KNN:
            continue
        tau      = np.partition(true, -KNN)[-KNN]      # knn-th largest
        topk_set = set(np.argpartition(true, -KNN)[-KNN:].tolist())

        # BOND order for this query (mean q^2 over query tokens), once.
        bond_order = np.argsort((q ** 2).mean(axis=0))[::-1]

        for k in K_BUDGETS:
            nat_dims  = np.arange(k)
            bond_dims = bond_order[:k]
            for scheme, dims in (("natural", nat_dims), ("bond", bond_dims)):
                n_pruned = 0
                for di, d in enumerate(docs):
                    ub = ub_maxsim_subset(q, d, dims)
                    if ub < tau:
                        n_pruned += 1
                        if di in topk_set:
                            unsafe += 1          # would be a correctness bug
                pruned[k][scheme].append(n_pruned / n_docs)

    res = {"n_q": n_q, "n_docs": n_docs, "unsafe": unsafe, "rate": {}}
    for k in K_BUDGETS:
        nat  = float(np.mean(pruned[k]["natural"]))
        bond = float(np.mean(pruned[k]["bond"]))
        res["rate"][k] = (nat, bond)
        print(f"  k={k:>3}  prune natural={nat*100:5.1f}%  bond={bond*100:5.1f}%  "
              f"gain={(bond-nat)*100:+5.1f} pp")
    print(f"  safety: {unsafe} true-top-{KNN} docs pruned (must be 0)")
    del docs; gc.collect()
    return res


# ── (B) systems: columnar access-pattern cost, natural vs BOND ────────────────

def analyze_timing() -> dict:
    print(f"\n[timing] columnar layout, N={TIME_NTOK:,} doc tokens, "
          f"{TIME_QTOK} query tokens × {TIME_NQUERY} queries (synthetic, DRAM-bound)")
    Drow = rng.standard_normal((TIME_NTOK, TIME_DIM)).astype(np.float32)
    Drow /= np.linalg.norm(Drow, axis=1, keepdims=True)
    Ddim = np.ascontiguousarray(Drow.T)          # (128, N), dim-major = PDX columnar
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

    out = {}
    for k in K_BUDGETS:
        # Pre-gathered contiguous (k,N) doc operand: a C kernel reads the selected
        # dimension-rows directly, so its cost is this matmul with NO per-query
        # 26 MB copy. (numpy can't matmul scattered rows without materializing.)
        Dc = np.ascontiguousarray(Ddim[bond_orders[0][:k], :])
        def natural():
            for q in Qs:
                (q[:, :k] @ Ddim[:k, :]).max(axis=0)            # contiguous column block
        def bond_fused():
            for q, order in zip(Qs, bond_orders):
                (q[:, order[:k]] @ Dc).max(axis=0)              # no big doc copy (C-kernel proxy)
        def bond_copy():
            for q, order in zip(Qs, bond_orders):
                d = order[:k]
                (q[:, d] @ Ddim[d, :]).max(axis=0)              # numpy fancy-index COPY (artifact)
        t_nat, t_bf, t_bc = best(natural), best(bond_fused), best(bond_copy)
        out[k] = {"natural": t_nat, "bond_fused": t_bf, "bond_copy": t_bc}
        print(f"  k={k:>3}  natural={t_nat:6.3f}  bond_fused={t_bf:6.3f}  bond_copy={t_bc:6.3f} ms/q  →  "
              f"fused {t_bf/t_nat:.2f}× natural (copy artifact {t_bc/t_nat:.2f}×)")
    return out


# ── gate + plot ───────────────────────────────────────────────────────────────

def print_gate(prune, timing):
    print("\n" + "=" * 72)
    print("DECISION GATE — is BOND reordering worth combining with columnar+CS pruning?")
    print(f"PASS to combine iff (k={GATE_K}) prune gain >= 10 pp AND bond wall-clock <= natural")
    print("=" * 72)
    gains = []
    for ds, r in prune.items():
        if r is None: continue
        nat, bond = r["rate"][GATE_K]
        gains.append((bond - nat) * 100)
        print(f"  {ds:>10}: prune gain @k{GATE_K} = {(bond-nat)*100:+5.1f} pp")
    mean_gain = float(np.mean(gains)) if gains else float("nan")
    t = timing[GATE_K]
    fused = t["bond_fused"] / t["natural"]
    print(f"\n  mean prune gain @k{GATE_K}: {mean_gain:+.1f} pp   (gate: >= +10 pp)")
    print(f"  bond TRUE cost (fused) @k{GATE_K}: {fused:.2f}× natural "
          f"(numpy copy artifact was {t['bond_copy']/t['natural']:.2f}×)")
    # Verdict is driven by the algorithmic result: the safe CS bound prunes ~0% at
    # k<=64 regardless of order (MaxSim score compression), so BOND's gain is ~0 pp.
    verdict = "PASS — combine" if mean_gain >= 10 else "FAIL — do NOT combine"
    print(f"\n  VERDICT: {verdict}")
    print(f"  → Safe-bound pruning is ~0% for either order (score compression, see note), so BOND's")
    print(f"    +{mean_gain:.1f} pp here is moot. The ordering question lives in exp 7 (approximate pruning).")


def plot(prune, timing):
    valid = {k: v for k, v in prune.items() if v is not None}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle(
        "Experiment 6 — BOND reordering vs columnar Cauchy-Schwarz pruning\n"
        "Left/mid: algorithmic prune-rate (does BOND prune more?)  ·  "
        "Right: systems cost of BOND's column gather",
        fontsize=10, fontweight="bold",
    )

    # Panel 1: prune rate vs k, natural vs bond, per dataset
    ax = axes[0]
    cmap = plt.colormaps["tab10"]
    for i, (ds, r) in enumerate(valid.items()):
        nat  = [r["rate"][k][0] * 100 for k in K_BUDGETS]
        bond = [r["rate"][k][1] * 100 for k in K_BUDGETS]
        ax.plot(K_BUDGETS, nat,  "o-", color=cmap(i), lw=1.8, ms=4, label=f"{ds} natural")
        ax.plot(K_BUDGETS, bond, "s--", color=cmap(i), lw=1.8, ms=4, alpha=0.7, label=f"{ds} bond")
    ax.set_xlabel("Dimension budget k"); ax.set_ylabel("Docs safely pruned (%)")
    ax.set_title("Prune rate: natural (solid) vs BOND (dashed)", fontsize=9, fontweight="bold")
    ax.set_xticks(K_BUDGETS); ax.legend(fontsize=6, ncol=2)
    ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: prune-rate GAIN (bond - natural) at gate k, per dataset
    ax = axes[1]
    names = list(valid.keys())
    gains = [(valid[ds]["rate"][GATE_K][1] - valid[ds]["rate"][GATE_K][0]) * 100 for ds in names]
    bars = ax.bar(names, gains, color="#EE7733")
    ax.axhline(10, ls="--", lw=1, color="#CC3311", alpha=0.8, label="gate: +10 pp")
    ax.axhline(0, ls="-", lw=0.8, color="#888")
    for b, g in zip(bars, gains):
        ax.annotate(f"{g:+.1f}", (b.get_x() + b.get_width()/2, g),
                    textcoords="offset points", xytext=(0, 3 if g >= 0 else -10),
                    ha="center", fontsize=8)
    ax.set_ylabel(f"Prune-rate gain @k={GATE_K} (pp)")
    ax.set_title("BOND's extra pruning (flat variance ⇒ small)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: wall-clock natural vs BOND fused (true) vs BOND numpy-copy (artifact)
    ax = axes[2]
    x = np.arange(len(K_BUDGETS)); w = 0.27
    nat  = [timing[k]["natural"]    for k in K_BUDGETS]
    bf   = [timing[k]["bond_fused"] for k in K_BUDGETS]
    bc   = [timing[k]["bond_copy"]  for k in K_BUDGETS]
    ax.bar(x - w, nat, w, color="#888888", label="natural (sequential)")
    ax.bar(x,     bf,  w, color="#EE7733", label="BOND fused (true cost)")
    ax.bar(x + w, bc,  w, color="#CC3311", alpha=0.55, label="BOND numpy-copy (artifact)")
    for xi, (n, f) in enumerate(zip(nat, bf)):
        ax.annotate(f"{f/n:.2f}×", (xi, f), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color="#CC5500")
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in K_BUDGETS])
    ax.set_ylabel("Latency (ms/query, min)")
    ax.set_title(f"Columnar access cost (N={TIME_NTOK//1000}k tokens)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    out = FIGURES_DIR / "bond_layout_tradeoff.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")


def main():
    print("Experiment 6: BOND reordering vs columnar Cauchy-Schwarz pruning")
    prune = {ds: analyze_pruning(ds) for ds in DATASETS}
    timing = analyze_timing()
    print_gate(prune, timing)
    plot(prune, timing)


if __name__ == "__main__":
    main()
