"""
Experiment 11 — cheap probe: does MaxSim doc-pruning improve with candidate-set SIZE?

Motivation (from exp 9/10): pruning barely helped, and a true brute-force scan beat it. One
suspected reason is that the candidate set was tiny (we sampled 400–800 docs). The doc-level
bound prunes a doc when UB_d < T, where T = the K-th best score so far. With few candidates
and K=10, T sits low → almost nothing is prunable. Real ColBERT/PLAID re-ranks shortlists of
8k–65k+ candidates, where T is far more selective → the doc-bound should fire earlier.

This probe sweeps n_docs and measures CELLS scanned (% of full) per arm — the implementation-
independent algorithmic-work signal. NO wall-clock timing (that was exp 10's job). Uses the
exp-09 kernel, whose `cells` counter reflects TRUE pruning (it scans only the live set), unlike
exp-10's dense-warmup kernel which deliberately over-scans for speed.

Read the result as: if cells% DROPS as n_docs grows, larger candidate sets do unlock pruning
(our small samples understated it); if it stays ~100%, size is not the lever and MaxSim score
compression is the real ceiling.

Run: source "$HOME/Data Engineering/research/.venv/bin/activate"; python probe.py
"""
import ctypes, pathlib, gc
import numpy as np
import matplotlib.pyplot as plt

HERE  = pathlib.Path(__file__).parent
FIGS  = HERE / "figures"; FIGS.mkdir(exist_ok=True)
CACHE = HERE.parent / "02_bond_variance" / ".cache"
lib   = ctypes.CDLL(str(HERE / "maxsim_kernels.so"))

f32p = ctypes.POINTER(ctypes.c_float); u32p = ctypes.POINTER(ctypes.c_uint32)
u64p = ctypes.POINTER(ctypes.c_uint64); csz = ctypes.c_size_t
lib.maxsim_knn.argtypes = [f32p, u64p, csz, f32p, csz, csz, u32p, u32p, csz,
                           f32p, ctypes.c_float, csz, u32p, f32p, u64p]
lib.maxsim_knn.restype  = ctypes.c_uint64
def fp(a): return a.ctypes.data_as(f32p)
def up(a): return a.ctypes.data_as(u32p)
def lp(a): return a.ctypes.data_as(u64p)

# ── config ────────────────────────────────────────────────────────────────────
D, K, M, NQ = 128, 10, 16, 6                       # fixed query-token count M for comparability
SHRINKS = {"safe": 1.0, "approx": 0.85}            # safe (recall 1.0) + one approximate point
# per-dataset candidate-set sweep (capped to corpus size; SCIDOCS capped for memory)
SWEEP = {
    "NFCorpus": [400, 1600, 3633],
    "SciFact":  [400, 1600, 5183],
    "ArguAna":  [400, 1600, 4000, 8674],
    "SCIDOCS":  [400, 1600, 6400, 16000],
}
FETCH = np.array([4, 8, 8, 12, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64,
                  128, 128, 128, 128, 256, 256, 512, 1024, 2048, 4096], dtype=np.uint32)
IDENT = np.arange(D, dtype=np.uint32)

_ROT = None
def rotation():
    global _ROT
    if _ROT is None:
        g = np.random.default_rng(123).standard_normal((D, D))
        Qr, _ = np.linalg.qr(g); _ROT = Qr.astype(np.float32)
    return _ROT

def build_layout(docs, R=None):
    offs = np.zeros(len(docs) + 1, dtype=np.uint64); chunks = []
    for k, d in enumerate(docs):
        dd = (d @ R) if R is not None else d
        chunks.append(np.ascontiguousarray(dd.T).ravel())
        offs[k + 1] = offs[k] + d.shape[0]
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32), offs

def qcum(Q, order):
    c = np.zeros((Q.shape[0], D + 1), dtype=np.float32)
    c[:, 1:] = np.cumsum(Q[:, order] ** 2, axis=1)
    return np.ascontiguousarray(c, dtype=np.float32)

def bond_order(Q, mu, tp=0.25):
    imp = ((Q - mu) ** 2).sum(0); idx = np.argsort(imp)[::-1]; t = int(np.floor(D * tp))
    return np.concatenate([np.sort(idx[:t]), np.sort(idx[t:])]).astype(np.uint32)

def run(flat, offs, n_docs, Q, order, Qc, shrink):
    m = Q.shape[0]
    tid = np.empty(K, np.uint32); ts = np.empty(K, np.float32); st = np.zeros(3, np.uint64)
    lib.maxsim_knn(fp(flat), lp(offs), n_docs, fp(np.ascontiguousarray(Q)), m, D,
                   up(order), up(FETCH), FETCH.size, fp(Qc), ctypes.c_float(shrink), K,
                   up(tid), fp(ts), lp(st))
    return tid, st

def brute_topk(Q, docs):
    s = np.array([float((Q @ d.T).max(1).sum()) for d in docs])
    return set(np.argpartition(s, -K)[-K:].tolist())

# ── probe one dataset ─────────────────────────────────────────────────────────
def probe(ds):
    dp, qp = CACHE / f"{ds}_doc_embs.npy", CACHE / f"{ds}_qry_embs.npy"
    if not dp.exists():
        print(f"  [{ds}] cache missing — skipped"); return None
    docs_all = np.load(dp, allow_pickle=True); qrys_all = np.load(qp, allow_pickle=True)
    R = rotation(); rng = np.random.default_rng(7)
    # fixed query set: longest NQ queries, first M tokens (require >= M tokens)
    longest = sorted(range(len(qrys_all)), key=lambda i: -qrys_all[i].shape[0])[:NQ]
    m_eff = min(M, min(qrys_all[i].shape[0] for i in longest))
    Qs = [np.ascontiguousarray(qrys_all[i][:m_eff], dtype=np.float32) for i in longest]
    pts = [n for n in SWEEP[ds] if n <= len(docs_all)]
    print(f"\n[{ds}] corpus={len(docs_all)} docs, m={m_eff}, {NQ} queries, n_docs sweep {pts}")
    res = {s: {a: {"work": [], "recall": []} for a in ("natural", "bond", "ada")} for s in SHRINKS}
    for n in pts:
        idx = rng.choice(len(docs_all), n, replace=False)
        docs = [np.ascontiguousarray(docs_all[i], dtype=np.float32) for i in idx]
        mu = np.concatenate(docs, 0).mean(0).astype(np.float32)
        tot = sum(d.shape[0] for d in docs)
        flat_raw, offs = build_layout(docs); flat_rot, _ = build_layout(docs, R)
        gt = [brute_topk(q, docs) for q in Qs]
        arms = {
            "natural": (flat_raw, Qs, [IDENT] * NQ),
            "bond":    (flat_raw, Qs, [bond_order(q, mu) for q in Qs]),
            "ada":     (flat_rot, [q @ R for q in Qs], [IDENT] * NQ),
        }
        line = f"  n={n:>6}: "
        for s_name, s in SHRINKS.items():
            for a, (flat, QA, ords) in arms.items():
                wk, rc = [], []
                for i in range(NQ):
                    tid, st = run(flat, offs, n, QA[i], ords[i], qcum(QA[i], ords[i]), s)
                    wk.append(st[0] / (tot * D * m_eff)); rc.append(len(set(tid.tolist()) & gt[i]) / K)
                res[s_name][a]["work"].append(float(np.mean(wk)))
                res[s_name][a]["recall"].append(float(np.mean(rc)))
            line += (f"[{s_name}] nat={res[s_name]['natural']['work'][-1]*100:3.0f}% "
                     f"bond={res[s_name]['bond']['work'][-1]*100:3.0f}% "
                     f"ada={res[s_name]['ada']['work'][-1]*100:3.0f}%"
                     + (f"(r{res[s_name]['bond']['recall'][-1]:.2f})" if s_name != "safe" else "") + "   ")
        print(line)
        del docs, flat_raw, flat_rot; gc.collect()
    return {"pts": pts, "res": res}

def plot(all_res):
    valid = {ds: r for ds, r in all_res.items() if r}
    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    fig.suptitle("Experiment 11 — does MaxSim pruning improve with candidate-set size?  "
                 "Cells scanned (% of full) vs n_docs, m=16\n"
                 "Solid = SAFE (recall 1.0); dashed = approx (shrink 0.85). Down-slope ⇒ bigger "
                 "shortlists unlock the doc-bound (T more selective).", fontsize=9.5, fontweight="bold")
    colors = {"natural": "#888888", "bond": "#0077BB", "ada": "#009988"}
    for k, (ds, r) in enumerate(valid.items()):
        ax = axes[0][k]
        for a in ("natural", "bond", "ada"):
            ax.plot(r["pts"], [w * 100 for w in r["res"]["safe"][a]["work"]], "o-", color=colors[a], lw=2, ms=4, label=f"{a}")
            ax.plot(r["pts"], [w * 100 for w in r["res"]["approx"][a]["work"]], "s--", color=colors[a], lw=1.5, ms=3, alpha=0.7)
        ax.set_xscale("log"); ax.set_xlabel("candidate docs (n_docs)")
        if k == 0: ax.set_ylabel("cells scanned (% of full)")
        ax.set_title(ds, fontsize=9, fontweight="bold"); ax.set_ylim(0, 105)
        if k == 0: ax.legend(fontsize=7)
        ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    out = FIGS / "pruning_scale_probe.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")

def main():
    print("Experiment 11 — pruning-vs-candidate-set-size probe (cells only, exp-09 kernel)")
    all_res = {ds: probe(ds) for ds in SWEEP}
    plot(all_res)
    print("\nInterpretation: a downward slope in safe-mode cells% as n_docs grows means the "
          "doc-bound\nfires more with realistic shortlists (our small samples understated pruning). "
          "A flat ~100%\nmeans candidate-set size is not the lever and MaxSim score compression is "
          "the real ceiling.")

if __name__ == "__main__":
    main()
