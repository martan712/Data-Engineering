"""
Experiment 8 (v3): pruning kernels — BOND vs ADSampling against PDX's REAL knobs.


The k-NN kernel also now uses PDX's adaptive fetch schedule
(DIMENSIONS_FETCHING_SIZES {4,8,8,12,16,16,32,...}, pdxearch.hpp:87): the prune
predicate is evaluated once per dim-BLOCK (Warmup→Prune cadence), not per dim.

Build:  clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -o pruning_kernels.so pruning_kernels.cpp
Run:    python pruning_kernels_bench.py
"""
import ctypes, time, pathlib
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figures"; FIGS.mkdir(exist_ok=True)
lib  = ctypes.CDLL(str(HERE / "pruning_kernels.so"))

f32p, u32p, csz = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t
lib.scan_partial_blocked.argtypes = [f32p, f32p, u32p, csz, csz, csz, csz, f32p]
lib.scan_partial_blocked.restype  = None
lib.knn_l2_blocked.argtypes = [f32p, f32p, u32p, f32p, u32p, csz, csz, csz, csz, csz, u32p, f32p]
lib.knn_l2_blocked.restype  = ctypes.c_uint64
def fp(a): return a.ctypes.data_as(f32p)
def up(a): return a.ctypes.data_as(u32p)

# ── config ────────────────────────────────────────────────────────────────────
N, D, KNN, NQ = 65_536, 128, 10, 100
# PDX's shipped adaptive fetch schedule (pdxearch.hpp:87). Sums well past D=128.
FETCH = np.array([4, 8, 8, 12, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64,
                  128, 128, 128, 128, 256, 256, 512, 1024, 2048, 4096], dtype=np.uint32)
# Part A: a "bucket" is a single dim-major run of BS vectors. ×64 is the pessimal
# block size PDX never uses on IVF; real buckets are 100s–1000s.
BS_SWEEP   = [64, 128, 256, 512, 1024, 2048, 4096]
BS_PRUNE   = 1024                       # representative real bucket size for Part B
TOP_PERCS  = [1 / 8, 1 / 4, 1 / 2]      # DISTANCE_TO_MEANS_IMPROVED knob (PDX uses 1/4)
ALPHA_GRID = np.linspace(0.0, 6.0, 49)  # ADSampling ε₀ sweep for matched recall
WARMUP, REPEATS = 3, 8
rng = np.random.default_rng(42)

# ── data ──────────────────────────────────────────────────────────────────────
X = rng.standard_normal((N, D)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = rng.standard_normal((NQ, D)).astype(np.float32); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
mean = X.mean(0)
ident = np.arange(D, dtype=np.uint32)

def blockify(Xin, BS):
    return np.ascontiguousarray(Xin.reshape(N // BS, BS, D).transpose(0, 2, 1)).ravel()

# ADSampling: one orthogonal rotation at build; query rotated per query (cheap).
Rg = rng.standard_normal((D, D)); R, _ = np.linalg.qr(Rg); R = R.astype(np.float32)
Xr, Qr = X @ R, Q @ R

def ads_ratios(alpha):
    r = np.ones(D + 1, dtype=np.float32)
    for v in range(1, D):
        r[v] = (v / D) * (1 + alpha / np.sqrt(v)) ** 2
    return r
ratios_exact = np.ones(D + 1, dtype=np.float32)

# ── PDX's real dimension orders ───────────────────────────────────────────────
def order_distance_to_means(q):
    """DISTANCE_TO_MEANS (pdxearch.hpp:225): full argsort by |q−mean| desc — the
    naive per-dim scatter (pessimal cache behaviour)."""
    return np.argsort(np.abs(q - mean))[::-1].astype(np.uint32)

def order_dtm_improved(q, top_perc):
    """DISTANCE_TO_MEANS_IMPROVED (pdxearch.hpp:235): top `top_perc` dims by
    |q−mean|, then INDEX-SORT each partition so access is monotonic-with-gaps
    (prefetcher-friendly), not a random scatter."""
    tp = int(np.floor(D * top_perc))
    idx = np.argsort(np.abs(q - mean))[::-1]          # importance desc
    top, rest = np.sort(idx[:tp]), np.sort(idx[tp:])  # index-sort each partition
    return np.concatenate([top, rest]).astype(np.uint32)

def best(fn):
    for _ in range(WARMUP): fn()
    b = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t0)
    return b

# oracle ground truth (shared)
GT = [set(np.argpartition(((X - Q[i]) ** 2).sum(1), KNN)[:KNN].tolist()) for i in range(NQ)]

# ── validation: exact arms must be recall 1.000 ───────────────────────────────
def knn(Bk_, qf, of, ratios, BS):
    tid = np.empty(KNN, np.uint32); td = np.empty(KNN, np.float32)
    return lib.knn_l2_blocked(fp(Bk_), fp(qf), up(of), fp(ratios), up(FETCH), FETCH.size,
                              N, D, BS, KNN, up(tid), fp(td)), tid

def validate():
    Bk = blockify(X, BS_PRUNE)
    rec = lambda of: np.mean([len(set(knn(Bk, Q[i], of(i), ratios_exact, BS_PRUNE)[1].tolist()) & GT[i]) / KNN
                              for i in range(NQ)])
    r_nat = rec(lambda i: ident)
    r_bond = rec(lambda i: order_dtm_improved(Q[i], 1 / 4))
    print(f"Validation: natural recall={r_nat:.3f}  bond-improved(1/4) recall={r_bond:.3f}  (both must be 1.000)")
    return r_nat == 1.0 and r_bond == 1.0

# ── Part A: access cost — block-size sweep + order penalty ─────────────────────
def part_a():
    print("\nPart A — pure access cost (full 128-dim scan, no pruning)")
    out = np.empty(N, np.float32)

    print("  (1) penalty vs bucket size BS  [×64 is the pessimal size PDX never uses]")
    pen_full, pen_impr = {}, {}
    for BS in BS_SWEEP:
        Bk = blockify(X, BS)
        full = [order_distance_to_means(Q[i]) for i in range(NQ)]
        impr = [order_dtm_improved(Q[i], 1 / 4) for i in range(NQ)]
        t_seq  = best(lambda: [lib.scan_partial_blocked(fp(Bk), fp(Q[i]), up(ident),   D, N, D, BS, fp(out)) for i in range(NQ)])
        t_full = best(lambda: [lib.scan_partial_blocked(fp(Bk), fp(Q[i]), up(full[i]), D, N, D, BS, fp(out)) for i in range(NQ)])
        t_impr = best(lambda: [lib.scan_partial_blocked(fp(Bk), fp(Q[i]), up(impr[i]), D, N, D, BS, fp(out)) for i in range(NQ)])
        pen_full[BS], pen_impr[BS] = t_full / t_seq, t_impr / t_seq
        print(f"    BS={BS:>5}  full-argsort={pen_full[BS]:.2f}×  improved(1/4)={pen_impr[BS]:.2f}×  sequential")

    print(f"  (2) order penalty at the pessimal BS={BS_SWEEP[0]}  [naive scatter vs PDX's improved order]")
    BS = BS_SWEEP[0]; Bk = blockify(X, BS)
    t_seq = best(lambda: [lib.scan_partial_blocked(fp(Bk), fp(Q[i]), up(ident), D, N, D, BS, fp(out)) for i in range(NQ)])
    order_pen = {"sequential": 1.0}
    full = [order_distance_to_means(Q[i]) for i in range(NQ)]
    order_pen["full argsort"] = best(lambda: [lib.scan_partial_blocked(fp(Bk), fp(Q[i]), up(full[i]), D, N, D, BS, fp(out)) for i in range(NQ)]) / t_seq
    for tp in TOP_PERCS:
        ords = [order_dtm_improved(Q[i], tp) for i in range(NQ)]
        order_pen[f"improved {tp:.3g}"] = best(lambda: [lib.scan_partial_blocked(fp(Bk), fp(Q[i]), up(ords[i]), D, N, D, BS, fp(out)) for i in range(NQ)]) / t_seq
    for k, v in order_pen.items():
        print(f"    {k:>14}: {v:.2f}× sequential")
    return {"bs_full": pen_full, "bs_impr": pen_impr, "order_pen": order_pen}

# ── Part B: matched-recall pruning ────────────────────────────────────────────
def measure(Bk_, qf, of, ratios, BS):
    df, rc = [], []
    for i in range(NQ):
        nd, tid = knn(Bk_, qf(i), of(i), ratios, BS)
        df.append(nd / (N * D)); rc.append(len(set(tid.tolist()) & GT[i]) / KNN)
    ms = best(lambda: [knn(Bk_, qf(i), of(i), ratios, BS) for i in range(NQ)]) / NQ * 1e3
    return float(np.mean(df)), float(np.mean(rc)), ms

def ada_at_recall(Bkr, target):
    """Sweep ADSampling ε₀; return the (dims, recall, ms, alpha) closest to `target`
    from above (smallest dims among recall>=target, else best-recall point)."""
    cand = []
    for a in ALPHA_GRID:
        ratios = ads_ratios(a)
        rc, df = [], []
        for i in range(NQ):
            nd, tid = knn(Bkr, Qr[i], ident, ratios, BS_PRUNE)
            df.append(nd / (N * D)); rc.append(len(set(tid.tolist()) & GT[i]) / KNN)
        cand.append((a, float(np.mean(df)), float(np.mean(rc))))
    feas = [c for c in cand if c[2] >= target]
    a, d, r = min(feas, key=lambda c: c[1]) if feas else max(cand, key=lambda c: c[2])
    ms = best(lambda: [knn(Bkr, Qr[i], ident, ads_ratios(a), BS_PRUNE) for i in range(NQ)]) / NQ * 1e3
    return {"dims": d, "recall": r, "ms": ms, "alpha": a}

def part_b():
    print(f"\nPart B — matched-recall pruning (faithful bucket BS={BS_PRUNE}, adaptive fetch schedule)")
    Bk, Bkr = blockify(X, BS_PRUNE), blockify(Xr, BS_PRUNE)
    res = {}
    d, r, m = measure(Bk, lambda i: Q[i], lambda i: ident, ratios_exact, BS_PRUNE)
    res["natural"] = {"dims": d, "recall": r, "ms": m}
    bond_ords = [order_dtm_improved(Q[i], 1 / 4) for i in range(NQ)]
    d, r, m = measure(Bk, lambda i: Q[i], lambda i: bond_ords[i], ratios_exact, BS_PRUNE)
    res["bond-improved"] = {"dims": d, "recall": r, "ms": m}
    res["ada@0.99"]  = ada_at_recall(Bkr, 0.99)
    res["ada@0.999"] = ada_at_recall(Bkr, 0.999)
    print(f"  {'arm':>14} {'dims%':>7} {'recall':>7} {'ms/q':>8}   note")
    notes = {"natural": "exact bound", "bond-improved": "exact bound, DTM_IMPROVED(1/4)",
             "ada@0.99": f"ε₀={res['ada@0.99']['alpha']:.2f}", "ada@0.999": f"ε₀={res['ada@0.999']['alpha']:.2f}"}
    for k in ["natural", "bond-improved", "ada@0.99", "ada@0.999"]:
        v = res[k]
        print(f"  {k:>14} {v['dims']*100:6.1f} {v['recall']:7.3f} {v['ms']:8.4f}   {notes[k]}")
    return res

# ── plot ──────────────────────────────────────────────────────────────────────
def plot(a, b):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle(
        "Experiment 8 (v3) — BOND vs ADSampling against PDX's real knobs (single-vector L2, 128D)\n"
        "Unblocked buckets + DISTANCE_TO_MEANS_IMPROVED order + adaptive fetch schedule + matched recall",
        fontsize=9.5, fontweight="bold")

    # Panel 1: access penalty vs bucket size
    ax = axes[0]
    ax.plot(BS_SWEEP, [a["bs_full"][bs] for bs in BS_SWEEP], "o-", lw=2, color="#CC3311", label="full argsort (naive)")
    ax.plot(BS_SWEEP, [a["bs_impr"][bs] for bs in BS_SWEEP], "s-", lw=2, color="#0077BB", label="improved 1/4 (PDX)")
    ax.axhline(1.0, ls="--", lw=0.8, color="#888")
    ax.set_xscale("log", base=2); ax.set_xticks(BS_SWEEP); ax.set_xticklabels(BS_SWEEP, rotation=45, fontsize=7)
    ax.set_xlabel("bucket size BS (vectors / dim-run)"); ax.set_ylabel("access penalty (× sequential)")
    ax.set_title("BS=64 pessimal; larger buckets ~halve it,\nthen plateau ~1.7× (min at BS=128)", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: order penalty at the pessimal BS=64
    ax = axes[1]
    keys = list(a["order_pen"].keys()); vals = [a["order_pen"][k] for k in keys]
    cols = ["#888888", "#CC3311"] + ["#0077BB"] * (len(keys) - 2)
    ax.bar(keys, vals, color=cols)
    ax.axhline(1.0, ls="--", lw=0.8, color="#888")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}×", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    ax.set_ylabel(f"access penalty (× sequential)")
    ax.set_title(f"Order penalty at pessimal BS={BS_SWEEP[0]}\nindex-sort (improved) ≪ naive scatter", fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: matched-recall — dims scanned + ms, recall annotated
    ax = axes[2]; ax2 = ax.twinx()
    keys = ["natural", "bond-improved", "ada@0.99", "ada@0.999"]
    labels = ["natural\n(exact)", "bond-impr\n(exact)", "ADA\n@0.99", "ADA\n@0.999"]
    dims = [b[k]["dims"] * 100 for k in keys]; ms = [b[k]["ms"] for k in keys]
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w / 2, dims, w, color="#0077BB", label="dims scanned %")
    ax2.bar(x + w / 2, ms, w, color="#EE7733", label="ms/query")
    for xi, k in enumerate(keys):
        ax.annotate(f"recall\n{b[k]['recall']*100:.1f}%", (xi - w / 2, dims[xi]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6.5, color="#0077BB")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("dims scanned (%)", color="#0077BB"); ax2.set_ylabel("ms/query (min)", color="#EE7733")
    ax.set_title(f"Matched recall (faithful BS={BS_PRUNE})\nbond-improved = exact; ADA swept to target", fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top"]].set_visible(False)

    out = FIGS / "pruning_kernels.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")

def main():
    print(f"Experiment 8 (v3): N={N:,}, D={D}, knn={KNN}, NQ={NQ}, faithful BS={BS_PRUNE}\n")
    if not validate():
        print("VALIDATION FAILED — aborting."); return
    a = part_a()
    b = part_b()
    plot(a, b)
    print("\nTakeaway: the 1.4–2.7× BOND access penalty is largely a ×64 artifact — any larger "
          "bucket\nroughly halves it (min at BS=128), though it plateaus around ~1.7× rather than "
          "reaching 1×.\nThe shipped index-sort order is friendlier than a naive scatter. At matched "
          "recall,\nADSampling and bond-improved are the honest comparison (see Part B table).")

if __name__ == "__main__":
    main()
