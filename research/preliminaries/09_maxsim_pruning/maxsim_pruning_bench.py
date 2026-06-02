"""
Experiment 9: MaxSim (ColBERT) pruning — BOND vs ADSampling vs natural, PDX-faithful.

The project's actual target is MaxSim, not single-vector kNN (exp 8). The integration
idea that resolves the BOND↔MaxSim tension:

  Single-vector BOND picks ONE per-query dim order, but MaxSim has m query tokens each
  with its own importance. Resolution: prune on the DOC-TOKEN axis and share ONE
  block-aware dim order across the whole query. The per-query reorder is paid once per
  query, then every doc reuses it — so MaxSim is far more BOND-favorable than kNN.

Three nested pruning levels, one synchronized columnar pass (see maxsim_kernels.cpp):
  (1) synchronized multi-token scan (exp 4) — one column pass serves all m query tokens;
  (2) inner-max token pruning — drop a doc-token once its Cauchy-Schwarz upper bound is
      below every query token's running lower bound (live-set compaction, PDX positions);
  (3) outer doc pruning — prune a doc when Σ_i max_j U_ij < T (T = K-th best score so far).

Dimension axis = the exp-8 question lifted to MaxSim:
  natural — first-k dims (fixed order).
  bond    — block-aware: aggregate importance Σ_i (q_i−μ)², DISTANCE_TO_MEANS_IMPROVED
            partition+index-sort (one order for the whole query). top_perc swept.
  ada     — rotate all tokens once by an orthogonal R at build (MaxSim exactly preserved),
            scan sequentially; no per-query reorder.

Recall knob: a single `shrink` ∈ [0,1] scales the Cauchy-Schwarz residual.
  shrink = 1.0 → SAFE bound (recall 1.000, exact top-K — validated vs brute force);
  shrink < 1.0 → approximate (more pruning, recall<1). Reported rule, swept to matched recall.

Real data: cached BEIR ColBERT embeddings in 02_bond_variance/.cache (exp 6/7). Falls
back to synthetic if the cache is missing.

Build:  clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -o maxsim_kernels.so maxsim_kernels.cpp
Run:    python maxsim_pruning_bench.py
"""
import ctypes, time, pathlib, gc
import numpy as np
import matplotlib.pyplot as plt

HERE     = pathlib.Path(__file__).parent
FIGS     = HERE / "figures"; FIGS.mkdir(exist_ok=True)
CACHE_02 = HERE.parent / "02_bond_variance" / ".cache"
lib      = ctypes.CDLL(str(HERE / "maxsim_kernels.so"))

f32p = ctypes.POINTER(ctypes.c_float)
u32p = ctypes.POINTER(ctypes.c_uint32)
u64p = ctypes.POINTER(ctypes.c_uint64)
csz  = ctypes.c_size_t
lib.maxsim_knn.argtypes = [f32p, u64p, csz, f32p, csz, csz, u32p, u32p, csz,
                           f32p, ctypes.c_float, csz, u32p, f32p, u64p]
lib.maxsim_knn.restype  = ctypes.c_uint64
lib.maxsim_full.argtypes = [f32p, u64p, csz, f32p, csz, csz, csz, u32p, f32p]  # brute-force baseline
lib.maxsim_full.restype  = ctypes.c_uint64
def fp(a): return a.ctypes.data_as(f32p)
def up(a): return a.ctypes.data_as(u32p)
def lp(a): return a.ctypes.data_as(u64p)

# ── config ────────────────────────────────────────────────────────────────────
D           = 128
K           = 10
MAX_DOCS    = 800
MAX_QUERIES = 12
DATASETS    = ["NFCorpus", "SciFact", "ArguAna"]   # SCIDOCS cache also present (2.3 GB; add if desired)
TOP_PERCS   = [1 / 8, 1 / 4, 1 / 2]                 # block-aware BOND knob (PDX default 1/4)
BOND_TP     = 1 / 4
SHRINKS     = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5, 0.3]  # recall knob sweep (1.0 = safe)
TARGET_REC  = 0.95                                  # matched-recall operating point for wall-clock
FETCH = np.array([4, 8, 8, 12, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64,
                  128, 128, 128, 128, 256, 256, 512, 1024, 2048, 4096], dtype=np.uint32)
WARMUP, REPEATS = 2, 4
rng = np.random.default_rng(42)

_ROT = None
def rotation():
    global _ROT
    if _ROT is None:
        g = np.random.default_rng(123).standard_normal((D, D))
        Qr, _ = np.linalg.qr(g); _ROT = Qr.astype(np.float32)
    return _ROT

# ── data ──────────────────────────────────────────────────────────────────────
def load_dataset(name):
    dp, qp = CACHE_02 / f"{name}_doc_embs.npy", CACHE_02 / f"{name}_qry_embs.npy"
    if not dp.exists() or not qp.exists():
        return None, None
    docs = np.load(dp, allow_pickle=True)
    qrys = np.load(qp, allow_pickle=True)
    n = min(len(docs), MAX_DOCS)
    idx = rng.choice(len(docs), n, replace=False)
    docs = [np.ascontiguousarray(docs[i], dtype=np.float32) for i in idx]
    nq = min(len(qrys), MAX_QUERIES)
    qrys = [np.ascontiguousarray(qrys[i], dtype=np.float32) for i in range(nq)]
    return docs, qrys

def synthetic():
    print("  [synthetic fallback] cache missing")
    docs = []
    for _ in range(MAX_DOCS):
        nd = int(rng.integers(40, 200))
        d = rng.standard_normal((nd, D)).astype(np.float32); d /= np.linalg.norm(d, axis=1, keepdims=True)
        docs.append(d)
    qrys = []
    for _ in range(MAX_QUERIES):
        q = rng.standard_normal((int(rng.integers(10, 33)), D)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True); qrys.append(q)
    return docs, qrys

# ── layout (per-doc dim-major) + per-query Qcum ───────────────────────────────
def build_layout(docs, R=None):
    offs = np.zeros(len(docs) + 1, dtype=np.uint64)
    chunks = []
    for k, d in enumerate(docs):
        dd = (d @ R) if R is not None else d
        chunks.append(np.ascontiguousarray(dd.T).ravel())   # dim-major (D, n_d)
        offs[k + 1] = offs[k] + d.shape[0]
    flat = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    return flat, offs

def qcum(Q, order):
    qo = (Q[:, order] ** 2)
    c = np.zeros((Q.shape[0], D + 1), dtype=np.float32)
    c[:, 1:] = np.cumsum(qo, axis=1)
    return np.ascontiguousarray(c, dtype=np.float32)

def bond_order(Q, mu, top_perc):
    imp = ((Q - mu) ** 2).sum(axis=0)               # aggregate Σ_i (q_i−μ)^2 over query tokens
    idx = np.argsort(imp)[::-1]
    tp = int(np.floor(D * top_perc))
    return np.concatenate([np.sort(idx[:tp]), np.sort(idx[tp:])]).astype(np.uint32)

# ── kernel wrapper ────────────────────────────────────────────────────────────
IDENT = np.arange(D, dtype=np.uint32)

def run(flat, offs, n_docs, Q, order, Qc, shrink):
    m = Q.shape[0]
    tid = np.empty(K, np.uint32); ts = np.empty(K, np.float32); st = np.zeros(3, np.uint64)
    lib.maxsim_knn(fp(flat), lp(offs), n_docs, fp(np.ascontiguousarray(Q)), m, D,
                   up(order), up(FETCH), FETCH.size, fp(Qc), ctypes.c_float(shrink), K,
                   up(tid), fp(ts), lp(st))
    return tid, st  # st = [cells, docs_pruned, tokens_pruned]

def run_full(flat, offs, n_docs, Q):
    """Brute-force baseline: full MaxSim scan, no pruning (same columnar layout)."""
    m = Q.shape[0]
    tid = np.empty(K, np.uint32); ts = np.empty(K, np.float32)
    lib.maxsim_full(fp(flat), lp(offs), n_docs, fp(np.ascontiguousarray(Q)), m, D, K, up(tid), fp(ts))
    return tid

def best(fn):
    for _ in range(WARMUP): fn()
    b = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t0)
    return b

# ── brute-force ground truth ──────────────────────────────────────────────────
def brute_topk(Q, docs):
    s = np.array([float((Q @ d.T).max(axis=1).sum()) for d in docs])
    return set(np.argpartition(s, -K)[-K:].tolist())

# ── per-dataset analysis ──────────────────────────────────────────────────────
def analyze(name, docs, qrys):
    n_docs = len(docs)
    mu = np.concatenate([d for d in docs], axis=0).mean(axis=0).astype(np.float32)
    total_tokens = sum(d.shape[0] for d in docs)
    R = rotation()

    # build the three arms' layouts once ("build time")
    flat_raw, offs = build_layout(docs)
    flat_rot, _    = build_layout(docs, R)

    gt = [brute_topk(q, docs) for q in qrys]

    # per-query precomputed orders + Qcum
    bond_orders = {tp: [bond_order(q, mu, tp) for q in qrys] for tp in TOP_PERCS}
    arms = {
        "natural": {"flat": flat_raw, "Q": qrys, "order": [IDENT] * len(qrys)},
        "bond":    {"flat": flat_raw, "Q": qrys, "order": bond_orders[BOND_TP]},
        "ada":     {"flat": flat_rot, "Q": [q @ R for q in qrys], "order": [IDENT] * len(qrys)},
    }
    for a in arms.values():
        a["Qc"] = [qcum(a["Q"][i], a["order"][i]) for i in range(len(qrys))]

    def sweep(arm, shrink):
        recs, work, dp, tp = [], [], [], []
        for i in range(len(qrys)):
            tid, st = run(arm["flat"], offs, n_docs, arm["Q"][i], arm["order"][i], arm["Qc"][i], shrink)
            m = arm["Q"][i].shape[0]
            recs.append(len(set(tid.tolist()) & gt[i]) / K)
            work.append(st[0] / (total_tokens * D * m))
            dp.append(st[1] / n_docs); tp.append(st[2] / total_tokens)
        return (float(np.mean(recs)), float(np.mean(work)),
                float(np.mean(dp)), float(np.mean(tp)))

    print(f"\n[{name}] {n_docs} docs ({total_tokens:,} tokens), {len(qrys)} queries")
    res = {a: {"frontier": [], "safe": None} for a in arms}
    for a in arms:
        for s in SHRINKS:
            r, w, dp, tp = sweep(arms[a], s)
            res[a]["frontier"].append({"shrink": s, "recall": r, "work": w, "docprune": dp, "tokprune": tp})
        res[a]["safe"] = res[a]["frontier"][0]
        sf = res[a]["safe"]
        print(f"  {a:>8} safe(shrink=1): recall={sf['recall']:.3f}  work={sf['work']*100:5.1f}%  "
              f"docprune={sf['docprune']*100:4.1f}%  tokprune={sf['tokprune']*100:4.1f}%")

    # block-aware BOND top_perc sweep (safe mode → all recall 1.0; differentiator is work)
    tp_work = {}
    for tp in TOP_PERCS:
        arm = {"flat": flat_raw, "Q": qrys, "order": bond_orders[tp],
               "Qc": [qcum(qrys[i], bond_orders[tp][i]) for i in range(len(qrys))]}
        _, w, _, _ = sweep(arm, 1.0)
        tp_work[tp] = w
    best_tp = min(tp_work, key=tp_work.get)
    print("  bond top_perc (safe): " + "  ".join(f"{tp:.3g}={tp_work[tp]*100:.1f}%" for tp in TOP_PERCS)
          + f"  → best={best_tp:.3g}")
    res["bond_tp"] = {"work": tp_work, "best": best_tp}

    # wall-clock at matched recall (~TARGET_REC) per arm
    def pick_shrink(a):
        feas = [f for f in res[a]["frontier"] if f["recall"] >= TARGET_REC]
        return (min(feas, key=lambda f: f["work"]) if feas
                else max(res[a]["frontier"], key=lambda f: f["recall"]))
    timing = {}
    for a in arms:
        op = pick_shrink(a); s = op["shrink"]
        ms = best(lambda: [run(arms[a]["flat"], offs, n_docs, arms[a]["Q"][i],
                               arms[a]["order"][i], arms[a]["Qc"][i], s)
                           for i in range(len(qrys))]) / len(qrys) * 1e3
        timing[a] = {"ms": ms, "shrink": s, "recall": op["recall"], "work": op["work"]}
    # full-scan reference (no pruning): shrink huge
    ms_full = best(lambda: [run(flat_raw, offs, n_docs, qrys[i], IDENT,
                                arms["natural"]["Qc"][i], 1e6) for i in range(len(qrys))]) / len(qrys) * 1e3
    timing["full"] = {"ms": ms_full, "shrink": None, "recall": 1.0, "work": 1.0}
    print(f"  wall-clock @recall≈{TARGET_REC}:  " +
          "  ".join(f"{a}={timing[a]['ms']:.2f}ms(r={timing[a]['recall']:.2f})" for a in arms) +
          f"  full={ms_full:.2f}ms")
    res["timing"] = timing
    del flat_raw, flat_rot; gc.collect()
    return res

# ── validation ────────────────────────────────────────────────────────────────
def validate(docs, qrys):
    print("Validation — safe arm (shrink=1.0) must reproduce brute-force top-K (recall 1.000)")
    sl_docs = docs[:120]; sl_q = qrys[: min(5, len(qrys))]
    n = len(sl_docs)
    mu = np.concatenate(sl_docs, axis=0).mean(axis=0).astype(np.float32)
    R = rotation()
    flat_raw, offs = build_layout(sl_docs)
    flat_rot, _    = build_layout(sl_docs, R)
    ok = True
    for i, q in enumerate(sl_q):
        gt = brute_topk(q, sl_docs)
        for a, (flat, qq, order) in {
            "natural": (flat_raw, q, IDENT),
            "bond":    (flat_raw, q, bond_order(q, mu, BOND_TP)),
            "ada":     (flat_rot, q @ R, IDENT),
        }.items():
            tid, _ = run(flat, offs, n, qq, order, qcum(qq, order), 1.0)
            r = len(set(tid.tolist()) & gt) / K
            if r != 1.0:
                print(f"  FAIL q{i} {a}: recall={r:.3f}"); ok = False
    if ok:
        print("  OK — natural/bond/ada all exact at shrink=1.0")
    return ok

# ── plot ──────────────────────────────────────────────────────────────────────
def plot(results):
    valid = {k: v for k, v in results.items() if v is not None}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle(
        "Experiment 9 — MaxSim (ColBERT) pruning: BOND vs ADSampling vs natural, PDX-faithful\n"
        "Prune the doc-token axis; share one block-aware dim order across the query. "
        "Recall knob = Cauchy-Schwarz residual shrink (1.0 = safe).",
        fontsize=9.5, fontweight="bold")
    colors = {"natural": "#888888", "bond": "#0077BB", "ada": "#009988"}

    # Panel 1: work-vs-recall frontier (mean over datasets)
    ax = axes[0]
    for a in ("natural", "bond", "ada"):
        pts = []
        for s_idx in range(len(SHRINKS)):
            rs = [valid[ds][a]["frontier"][s_idx]["recall"] for ds in valid]
            ws = [valid[ds][a]["frontier"][s_idx]["work"] for ds in valid]
            pts.append((np.mean(rs) * 100, np.mean(ws) * 100))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colors[a], lw=2, ms=5, label=a)
    ax.set_xlabel(f"Recall of true top-{K} (%)"); ax.set_ylabel("Cells scanned (% of full)")
    ax.set_title("Work–recall frontier (lower = better)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: safe-mode prune rates (token vs doc), mean over datasets
    ax = axes[1]
    arms = ["natural", "bond", "ada"]; x = np.arange(len(arms)); w = 0.38
    tok = [np.mean([valid[ds][a]["safe"]["tokprune"] for ds in valid]) * 100 for a in arms]
    doc = [np.mean([valid[ds][a]["safe"]["docprune"] for ds in valid]) * 100 for a in arms]
    ax.bar(x - w / 2, tok, w, color="#EE7733", label="doc-tokens pruned")
    ax.bar(x + w / 2, doc, w, color="#CC3311", label="docs pruned")
    for xi in range(len(arms)):
        ax.annotate(f"{tok[xi]:.0f}%", (xi - w / 2, tok[xi]), textcoords="offset points",
                    xytext=(0, 2), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_ylabel("pruned (%)"); ax.set_title("Safe mode (recall 1.0): token-pruning dominates",
                                              fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: wall-clock at matched recall (mean over datasets)
    ax = axes[2]
    arms2 = ["full", "natural", "bond", "ada"]
    ms = [np.mean([valid[ds]["timing"][a]["ms"] for ds in valid]) for a in arms2]
    rc = [np.mean([valid[ds]["timing"][a]["recall"] for ds in valid]) for a in arms2]
    cols = ["#bbbbbb", "#888888", "#0077BB", "#009988"]
    ax.bar(arms2, ms, color=cols)
    for i, a in enumerate(arms2):
        ax.annotate(f"{ms[i]:.2f}\nr={rc[i]:.2f}", (i, ms[i]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7)
    ax.set_ylabel("ms/query (min)")
    ax.set_title(f"Wall-clock @recall≈{TARGET_REC} (mean over datasets)", fontsize=9, fontweight="bold")
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    out = FIGS / "maxsim_pruning.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")

# ── exp 9b: amortization sweep (repeat exp 8's BOND-vs-natural-vs-ADA question over m) ──
# Exp 8 (single-vector L2, pessimal BS=64) found BOND SLOWEST despite scanning the fewest
# dims: ~1 multiply-add per loaded value → MEMORY-bound, so BOND's gather penalty (2.5× at
# 256-byte dim-rows) erased its dim savings. Exp 9 (MaxSim) found BOND ~fastest. This sweep
# repeats the comparison over m = number of query tokens sharing ONE reordered scan, on REAL
# docs (so BOND has a genuine cell advantage to cash in), SAFE mode (recall 1.0, all arms exact).
# Two things differ from exp-8's worst case and both favour BOND here:
#   (i) layout — the per-doc dim-row is ~doc-length (100s of floats), i.e. exp-8's LARGE-bucket
#       friendly regime, so the gather penalty is already mild (~1.02×) even at m=1; and
#  (ii) amortization + compute — as m grows the one reorder is reused across m tokens and each
#       loaded column does m multiply-adds (compute-bound), so BOND's ~5-6% fewer-cells edge
#       shows through and it edges below natural. The effect is real but modest (see figure).
AMORT_NDOCS   = 400
NQ_AMORT      = 6                              # longest queries used (fewer → higher m cap)
AMORT_MGRID   = [1, 2, 4, 8, 16, 32, 48]      # per dataset, capped to its real query length
# Fine recall-knob grid for 9c's per-arm tuning. BOND's recall(shrink) curve is steep, so a coarse
# grid lands the "most-aggressive shrink with recall≥target" point back at recall 1.0 (under-pruned);
# dense steps near 1.0 let each arm actually operate near TARGET_REC.
AMORT_SHRINKS = [1.0, 0.97, 0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.82, 0.79, 0.75, 0.70, 0.60, 0.50]

def _amort_setup(ds):
    """Shared real-data setup for the amortization sweeps (9b/9c). Returns None if cache absent.
    Each dataset is swept over its OWN query-token range (first m tokens of its longest queries)."""
    dp, qp = CACHE_02 / f"{ds}_doc_embs.npy", CACHE_02 / f"{ds}_qry_embs.npy"
    if not dp.exists() or not qp.exists():
        return None
    r2 = np.random.default_rng(7)
    docs_all = np.load(dp, allow_pickle=True); qrys_all = np.load(qp, allow_pickle=True)
    n_docs = min(len(docs_all), AMORT_NDOCS)
    idx = r2.choice(len(docs_all), n_docs, replace=False)
    docs = [np.ascontiguousarray(docs_all[i], dtype=np.float32) for i in idx]
    mu = np.concatenate(docs, axis=0).mean(axis=0).astype(np.float32)
    total_tokens = sum(d.shape[0] for d in docs)
    # longest real queries → widest m sweep; use the SAME queries at every m (first m tokens).
    order_by_len = sorted(range(len(qrys_all)), key=lambda i: -qrys_all[i].shape[0])
    qsel = [np.ascontiguousarray(qrys_all[i], dtype=np.float32) for i in order_by_len[:NQ_AMORT]]
    max_m = min(q.shape[0] for q in qsel)
    m_values = [m for m in AMORT_MGRID if m <= max_m]
    if m_values[-1] < max_m:
        m_values.append(int(max_m))           # always include the dataset's full query length
    R = rotation()
    flat_raw, offs = build_layout(docs)
    flat_rot, _    = build_layout(docs, R)
    return dict(docs=docs, qsel=qsel, mu=mu, R=R, n_docs=n_docs, total_tokens=total_tokens,
                m_values=m_values, max_m=max_m, flat_raw=flat_raw, flat_rot=flat_rot, offs=offs)

def _amort_arms(S, m):
    """Build the three arms (natural/bond/ada) for the first-m real tokens of each query."""
    Qs = [q[:m] for q in S["qsel"]]
    arms = {
        "natural": {"flat": S["flat_raw"], "Q": Qs, "order": [IDENT] * len(Qs)},
        "bond":    {"flat": S["flat_raw"], "Q": Qs, "order": [bond_order(q, S["mu"], BOND_TP) for q in Qs]},
        "ada":     {"flat": S["flat_rot"], "Q": [q @ S["R"] for q in Qs], "order": [IDENT] * len(Qs)},
    }
    for a in arms.values():
        a["Qc"] = [qcum(a["Q"][i], a["order"][i]) for i in range(len(Qs))]
    return arms

def _arm_measure(S, arm, shrink, gt=None):
    """Mean (work, recall) over queries for one arm at a fixed shrink (recall vs gt if given)."""
    work, rec = [], []
    for i in range(len(arm["Q"])):
        tid, st = run(arm["flat"], S["offs"], S["n_docs"], arm["Q"][i], arm["order"][i], arm["Qc"][i], shrink)
        work.append(st[0] / (S["total_tokens"] * D * arm["Q"][i].shape[0]))
        if gt is not None:
            rec.append(len(set(tid.tolist()) & gt[i]) / K)
    return float(np.mean(work)), (float(np.mean(rec)) if gt is not None else 1.0)

def _arm_time(S, arm, shrink):
    return best(lambda: [run(arm["flat"], S["offs"], S["n_docs"], arm["Q"][i], arm["order"][i], arm["Qc"][i], shrink)
                         for i in range(len(arm["Q"]))]) / len(arm["Q"]) * 1e3

def _amort_run_dataset(ds, mode):
    """Run the amortization sweep for one dataset. mode='safe' (recall 1.0) or 'recall'
    (each arm tuned per-m to ~TARGET_REC). Returns (m_values, res) or None if cache absent."""
    S = _amort_setup(ds)
    if S is None:
        print(f"  [{ds}] cache missing — skipped"); return None
    m_values = S["m_values"]
    # one safe-mode exactness check per dataset (must reproduce brute-force top-K)
    qv = S["qsel"][0][:m_values[-1]]; bo = bond_order(qv, S["mu"], BOND_TP)
    gt0 = brute_topk(qv, S["docs"])
    tid, _ = run(S["flat_raw"], S["offs"], S["n_docs"], qv, bo, qcum(qv, bo), 1.0)
    print(f"  [{ds}] {S['n_docs']} docs ({S['total_tokens']:,} tok), {len(S['qsel'])} queries, "
          f"m 1..{m_values[-1]}  (safe exactness r={len(set(tid.tolist()) & gt0) / K:.3f})")

    res = {a: {"ms": [], "work": [], "recall": []} for a in ("natural", "bond", "ada", "brute")}
    for m in m_values:
        arms = _amort_arms(S, m)
        gt = [brute_topk(q, S["docs"]) for q in arms["natural"]["Q"]] if mode == "recall" else None
        for a in ("natural", "bond", "ada"):
            if mode == "safe":
                w, _ = _arm_measure(S, arms[a], 1.0); ms = _arm_time(S, arms[a], 1.0); r = 1.0
            else:  # tune shrink to the smallest-work point with recall >= TARGET_REC
                pts = [(s, *_arm_measure(S, arms[a], s, gt)) for s in AMORT_SHRINKS]
                feas = [p for p in pts if p[2] >= TARGET_REC]
                s_op, w, r = min(feas, key=lambda p: p[1]) if feas else max(pts, key=lambda p: p[2])
                ms = _arm_time(S, arms[a], s_op)
            res[a]["ms"].append(ms); res[a]["work"].append(w); res[a]["recall"].append(r)
        # brute-force baseline: full scan, no pruning (work = 100%, recall = 1.0 by construction)
        Qb = arms["natural"]["Q"]
        bms = best(lambda: [run_full(S["flat_raw"], S["offs"], S["n_docs"], Qb[i]) for i in range(len(Qb))]) / len(Qb) * 1e3
        res["brute"]["ms"].append(bms); res["brute"]["work"].append(1.0); res["brute"]["recall"].append(1.0)
        print(f"    m={m:>2}  " + "  ".join(
            f"{a}={res[a]['ms'][-1]:6.2f}ms(w{res[a]['work'][-1]*100:.0f}%"
            + (f" r{res[a]['recall'][-1]:.2f}" if mode == "recall" else "") + ")"
            for a in ("natural", "bond", "ada", "brute"))
            + f"  | bond/nat={res['bond']['ms'][-1] / res['natural']['ms'][-1]:.2f}×"
            + f" bond/brute={res['bond']['ms'][-1] / res['brute']['ms'][-1]:.2f}×")
    return m_values, res

def amortization_experiment():
    print("\n" + "=" * 80)
    print("Exp 9b — amortization sweep over query-token count m, SAFE mode (recall 1.0), per dataset")
    print("  (m=1 ≈ exp-8 single-vector regime; BOND's fewer-cells edge cashes in as m amortizes)")
    print("=" * 80)
    out = {ds: _amort_run_dataset(ds, "safe") for ds in DATASETS}
    plot_amortization_grid(out, "maxsim_amortization.png",
        "Experiment 9b — BOND vs natural vs ADA over query-token count m, SAFE mode (recall 1.0)  "
        "— one row per dataset\n"
        "Left: wall-clock vs m  ·  Middle: wall-clock relative to natural (below 1.0 = BOND/ADA faster)  "
        "·  Right: cells scanned (algorithmic work)")
    return out

def amortization_recall_experiment():
    print("\n" + "=" * 80)
    print(f"Exp 9c — same sweep, each arm tuned per-m to ~{int(TARGET_REC*100)}% recall (APPROXIMATE), per dataset")
    print("  (does the dimension order matter more once recall loss is allowed?)")
    print("=" * 80)
    out = {ds: _amort_run_dataset(ds, "recall") for ds in DATASETS}
    plot_amortization_grid(out, "maxsim_amortization_recall.png",
        f"Experiment 9c — BOND vs natural vs ADA over m, tuned to ~{int(TARGET_REC*100)}% recall "
        "(APPROXIMATE)  — one row per dataset\n"
        "Left: wall-clock vs m  ·  Middle: wall-clock relative to natural (below 1.0 = faster)  "
        "·  Right: cells scanned (recall annotated)", show_recall=True)
    return out

def plot_amortization_grid(out, outfile, suptitle, show_recall=False):
    dsets = [ds for ds in DATASETS if out.get(ds) is not None]
    if not dsets:
        print("  no datasets available — figure skipped"); return
    nrow = len(dsets)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 3.6 * nrow), squeeze=False)
    fig.suptitle(suptitle, fontsize=9.5, fontweight="bold")
    colors = {"natural": "#888888", "bond": "#0077BB", "ada": "#009988", "brute": "#CC3311"}
    styles = {"natural": "o-", "bond": "o-", "ada": "o-", "brute": "s--"}

    for r_idx, ds in enumerate(dsets):
        m_values, res = out[ds]
        bottom = (r_idx == nrow - 1)

        ax = axes[r_idx][0]
        for a in ("brute", "natural", "bond", "ada"):
            ax.plot(m_values, res[a]["ms"], styles[a], color=colors[a], lw=2, ms=4, label=a)
        ax.set_xscale("log", base=2); ax.set_xticks(m_values); ax.set_xticklabels(m_values, fontsize=7)
        ax.set_ylabel(f"{ds}\nms/query (min)", fontsize=8.5, fontweight="bold")
        if r_idx == 0: ax.set_title("Absolute wall-clock vs m", fontsize=9, fontweight="bold")
        if bottom: ax.set_xlabel("query tokens m (shared scan)")
        if r_idx == 0: ax.legend(fontsize=7)
        ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

        ax = axes[r_idx][1]
        for a in ("bond", "ada", "brute"):
            rel = [res[a]["ms"][i] / res["natural"]["ms"][i] for i in range(len(m_values))]
            ax.plot(m_values, rel, styles[a], color=colors[a], lw=2, ms=4, label=f"{a}/natural")
        ax.axhline(1.0, ls="--", lw=1, color="#888")
        ax.set_xscale("log", base=2); ax.set_xticks(m_values); ax.set_xticklabels(m_values, fontsize=7)
        ax.set_ylabel("× natural", fontsize=8)
        if r_idx == 0: ax.set_title("Wall-clock relative to natural (<1 = faster)", fontsize=9, fontweight="bold")
        if bottom: ax.set_xlabel("query tokens m")
        if r_idx == 0: ax.legend(fontsize=7)
        ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

        ax = axes[r_idx][2]
        for a in ("brute", "natural", "bond", "ada"):
            ax.plot(m_values, [w * 100 for w in res[a]["work"]], styles[a], color=colors[a], lw=2, ms=4, label=a)
        if show_recall:
            for i, m in enumerate(m_values):
                ax.annotate(f"{res['bond']['recall'][i]:.2f}", (m, res["bond"]["work"][i] * 100),
                            textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6, color="#0077BB")
        ax.set_xscale("log", base=2); ax.set_xticks(m_values); ax.set_xticklabels(m_values, fontsize=7)
        ax.set_ylabel("cells (% of full)", fontsize=8)
        if r_idx == 0: ax.set_title("Algorithmic work (cells scanned)", fontsize=9, fontweight="bold")
        if bottom: ax.set_xlabel("query tokens m")
        if r_idx == 0: ax.legend(fontsize=7)
        ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

    out_path = FIGS / outfile
    fig.tight_layout(rect=[0, 0, 1, 0.99]); fig.savefig(out_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out_path}")

def main():
    print(f"Experiment 9: MaxSim pruning — D={D}, K={K}, ≤{MAX_DOCS} docs, ≤{MAX_QUERIES} queries/dataset\n")
    loaded = {}
    for ds in DATASETS:
        d, q = load_dataset(ds)
        loaded[ds] = (d, q)
    # validation on the first available dataset (or synthetic)
    first = next((loaded[ds] for ds in DATASETS if loaded[ds][0] is not None), (None, None))
    if first[0] is None:
        first = synthetic(); loaded = {"synthetic": first}; DATASETS[:] = ["synthetic"]
    if not validate(first[0], first[1]):
        print("VALIDATION FAILED — aborting."); return

    results = {}
    for ds in DATASETS:
        d, q = loaded[ds]
        if d is None:
            print(f"\n[{ds}] cache missing — skipped"); results[ds] = None; continue
        results[ds] = analyze(ds, d, q)
    plot(results)

    print("\nTakeaway: in MaxSim the max-then-sum collapse lets inner-max TOKEN pruning remove "
          "most\ndoc-token columns even in SAFE mode (recall 1.0) — the dim order is shared once "
          "per query,\nso block-aware BOND's reorder is amortized. See the work–recall frontier "
          "for the approximate regime.")

def amort():
    """Exp 9b + 9c only (the amortization sweeps) — no 3-dataset main analysis."""
    print("Experiment 9 (amortization sweeps 9b + 9c) — per-dataset, over query-token count m\n")
    amortization_experiment()          # 9b — safe mode (recall 1.0)
    amortization_recall_experiment()   # 9c — tuned to ~TARGET_REC recall

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "amort":     # python maxsim_pruning_bench.py amort  → just 9b + 9c
        amort()
    elif mode == "main":    # python maxsim_pruning_bench.py main   → just the 3-dataset analysis
        main()
    else:                   # default: everything
        amort(); main()
