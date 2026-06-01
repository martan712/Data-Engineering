"""
Experiment 8: pruning kernels — block-aware BOND vs ADSampling (global vs blocked)

Exp 8 (v1) found: in PDX's blocked (×64) layout, per-query BOND ordering scatters
*individual* dimensions (128 jumps of 256 B), defeating the prefetcher → 1.4–2.7×
access penalty, so BOND scanned fewer dims yet ran slower. (And the v1 wall-clock
under-counted pruning because the kernel didn't compact pruned lanes.)

This version fixes both:
  1. The k-NN kernel now COMPACTS survivors (dense live list) so fewer dims scanned
     actually means fewer iterations — wall-clock reflects pruning.
  2. **Block-aware (chunked) BOND**: reorder dimensions in CHUNKS of consecutive dims
     (each chunk is contiguous in the block → a sequential streaming read), and only
     the coarse *chunk order* is per-query. chunk=1 ≡ full per-dim BOND (max scatter);
     chunk=D ≡ natural (no reorder). A middle chunk size should keep most of BOND's
     pruning while making the block's prefetcher work *for* us.

Strategies (single-vector L2 k-NN, 128D, blocked layout):
    natural      — fixed order, exact monotone bound (recall 1.0)
    bond[chunk]  — chunk-ordered by (q−mean)² importance, exact bound (recall 1.0)
    adsampling   — rotation + PDX ratio bound (α=1.5, approximate)

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
lib.knn_l2_blocked.argtypes = [f32p, f32p, u32p, f32p, csz, csz, csz, csz, u32p, f32p]
lib.knn_l2_blocked.restype  = ctypes.c_uint64
def fp(a): return a.ctypes.data_as(f32p)
def up(a): return a.ctypes.data_as(u32p)

# ── config ────────────────────────────────────────────────────────────────────
N, D, BS, KNN, NQ = 65_536, 128, 64, 10, 100
ALPHA  = 1.5
CHUNKS = [1, 2, 4, 8, 16, 32, 64, 128]   # chunk=1 ≡ full BOND, chunk=128 ≡ natural
WARMUP, REPEATS = 3, 8
rng = np.random.default_rng(42)

# ── data + layouts ────────────────────────────────────────────────────────────
X = rng.standard_normal((N, D)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = rng.standard_normal((NQ, D)).astype(np.float32); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
mean = X.mean(0)
Bk = np.ascontiguousarray(X.reshape(N // BS, BS, D).transpose(0, 2, 1)).ravel()

Rg = rng.standard_normal((D, D)); R, _ = np.linalg.qr(Rg); R = R.astype(np.float32)
Xr, Qr = X @ R, Q @ R
Bkr = np.ascontiguousarray(Xr.reshape(N // BS, BS, D).transpose(0, 2, 1)).ravel()

ratios_exact = np.ones(D + 1, dtype=np.float32)
ratios_ads   = np.ones(D + 1, dtype=np.float32)
for v in range(1, D):
    ratios_ads[v] = (v / D) * (1 + ALPHA / np.sqrt(v)) ** 2

ident = np.arange(D, dtype=np.uint32)

def chunked_order(q, chunk):
    """Per-query order: chunks of `chunk` consecutive dims, chunks sorted by importance."""
    nch = D // chunk
    imp = ((q - mean) ** 2)[:nch * chunk].reshape(nch, chunk).sum(1)
    corder = np.argsort(imp)[::-1]
    return (corder[:, None] * chunk + np.arange(chunk)[None, :]).ravel().astype(np.uint32)

def best(fn):
    for _ in range(WARMUP): fn()
    b = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t0)
    return b

# ── validation ────────────────────────────────────────────────────────────────
def validate():
    q = Q[0]
    gt = set(np.argpartition(((X - q) ** 2).sum(1), KNN)[:KNN].tolist())
    tid = np.empty(KNN, np.uint32); td = np.empty(KNN, np.float32)
    lib.knn_l2_blocked(fp(Bk), fp(q), up(ident), fp(ratios_exact), N, D, BS, KNN, up(tid), fp(td))
    r1 = len(set(tid.tolist()) & gt) / KNN
    lib.knn_l2_blocked(fp(Bk), fp(q), up(chunked_order(q, 8)), fp(ratios_exact), N, D, BS, KNN, up(tid), fp(td))
    r2 = len(set(tid.tolist()) & gt) / KNN
    print(f"Validation: natural recall={r1:.3f}  chunked-bond(8) recall={r2:.3f}  (both must be 1.000)")
    return r1 == 1.0 and r2 == 1.0

# ── Part A: access cost vs chunk size ─────────────────────────────────────────
def part_a():
    print("\nPart A — blocked access cost (full 128-dim scan), chunked order vs sequential:")
    out = np.empty(N, np.float32)
    orders = {c: [chunked_order(Q[i], c) for i in range(NQ)] for c in CHUNKS}
    t_seq = best(lambda: [lib.scan_partial_blocked(fp(Bk), fp(Q[i]), up(ident), D, N, D, BS, fp(out)) for i in range(NQ)]) / NQ * 1e3
    res = {}
    for c in CHUNKS:
        t = best(lambda: [lib.scan_partial_blocked(fp(Bk), fp(Q[i]), up(orders[c][i]), D, N, D, BS, fp(out)) for i in range(NQ)]) / NQ * 1e3
        res[c] = t / t_seq
        print(f"  chunk={c:>3}  {t:.4f} ms/q   penalty={t/t_seq:.2f}× sequential")
    return res

# ── Part B: pruning (dims / recall / wall-clock) ──────────────────────────────
def run_scheme(Bk_, qf, orderf, ratios):
    tid = np.empty(KNN, np.uint32); td = np.empty(KNN, np.float32)
    gts = [set(np.argpartition(((X - Q[i]) ** 2).sum(1), KNN)[:KNN].tolist()) for i in range(NQ)]
    df, rc = [], []
    for i in range(NQ):
        nd = lib.knn_l2_blocked(fp(Bk_), fp(qf(i)), up(orderf(i)), fp(ratios), N, D, BS, KNN, up(tid), fp(td))
        df.append(nd / (N * D)); rc.append(len(set(tid.tolist()) & gts[i]) / KNN)
    ms = best(lambda: [lib.knn_l2_blocked(fp(Bk_), fp(qf(i)), up(orderf(i)), fp(ratios), N, D, BS, KNN, up(tid), fp(td)) for i in range(NQ)]) / NQ * 1e3
    return float(np.mean(df)), float(np.mean(rc)), ms

def part_b():
    print("\nPart B — pruning: dims-scanned / recall / wall-clock")
    res = {}
    res["natural"]    = run_scheme(Bk,  lambda i: Q[i],  lambda i: ident,                     ratios_exact)
    res["adsampling"] = run_scheme(Bkr, lambda i: Qr[i], lambda i: ident,                     ratios_ads)
    corders = {c: [chunked_order(Q[i], c) for i in range(NQ)] for c in CHUNKS}
    for c in CHUNKS:
        res[f"bond[{c}]"] = run_scheme(Bk, lambda i: Q[i], lambda i, c=c: corders[c][i], ratios_exact)
    for k, (d, r, m) in res.items():
        print(f"  {k:>11}: dims={d*100:5.1f}%  recall={r:.3f}  {m:.4f} ms/q")
    return res

# ── plot ──────────────────────────────────────────────────────────────────────
def plot(access, b):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle(
        "Experiment 8 — block-aware (chunked) BOND vs ADSampling, blocked ×64 layout, single-vector L2 128D\n"
        "Reorder dims in cache-aligned CHUNKS so per-query ordering stays prefetcher-friendly",
        fontsize=9.5, fontweight="bold")

    # Panel 1: access penalty vs chunk
    ax = axes[0]
    ax.plot(CHUNKS, [access[c] for c in CHUNKS], "o-", lw=2, color="#CC3311")
    ax.axhline(1.0, ls="--", lw=0.8, color="#888")
    ax.set_xscale("log", base=2); ax.set_xticks(CHUNKS); ax.set_xticklabels(CHUNKS)
    ax.set_xlabel("chunk size (dims/chunk)"); ax.set_ylabel("access penalty (× sequential)")
    ax.set_title("Part A: scatter penalty shrinks with chunk", fontsize=9, fontweight="bold")
    ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: chunked-bond dims-scanned + wall-clock vs chunk
    ax = axes[1]; ax2 = ax.twinx()
    dims = [b[f"bond[{c}]"][0] * 100 for c in CHUNKS]
    ms   = [b[f"bond[{c}]"][2] for c in CHUNKS]
    l1, = ax.plot(CHUNKS, dims, "s-", color="#0077BB", lw=2, label="dims scanned %")
    l2, = ax2.plot(CHUNKS, ms, "o-", color="#EE7733", lw=2, label="ms/query")
    ax.axhline(b["natural"][0]*100, ls=":", color="#888", lw=1)
    ax2.axhline(b["adsampling"][2], ls="--", color="#009988", lw=1.2, label="adsampling ms")
    ax.set_xscale("log", base=2); ax.set_xticks(CHUNKS); ax.set_xticklabels(CHUNKS)
    ax.set_xlabel("chunk size"); ax.set_ylabel("dims scanned (%)", color="#0077BB")
    ax2.set_ylabel("ms/query", color="#EE7733")
    ax.set_title("Part B: chunked-BOND sweet spot", fontsize=9, fontweight="bold")
    ax.legend(handles=[l1, l2], fontsize=7, loc="center left")
    ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top"]].set_visible(False)

    # Panel 3: wall-clock comparison
    ax = axes[2]
    best_chunk = min(CHUNKS, key=lambda c: b[f"bond[{c}]"][2])
    labels = ["natural", f"bond[1]\n(full)", f"bond[{best_chunk}]\n(block-aware)", "adsampling"]
    keys   = ["natural", "bond[1]", f"bond[{best_chunk}]", "adsampling"]
    vals   = [b[k][2] for k in keys]
    cols   = ["#888888", "#CC3311", "#0077BB", "#009988"]
    ax.bar(labels, vals, color=cols)
    for i, k in enumerate(keys):
        ax.annotate(f"{b[k][2]:.2f}\nr={b[k][1]:.2f}", (i, b[k][2]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7)
    ax.set_ylabel("ms/query (min)")
    ax.set_title("Wall-clock: block-aware BOND vs full BOND", fontsize=9, fontweight="bold")
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

    out = FIGS / "pruning_kernels.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")

def main():
    print(f"Experiment 8 (block-aware): N={N:,}, D={D}, BS={BS}, knn={KNN}, α={ALPHA}\n")
    if not validate():
        print("VALIDATION FAILED — aborting."); return
    a = part_a()
    b = part_b()
    plot(a, b)

if __name__ == "__main__":
    main()
