"""
Experiment 1: PDX kernel speedup on 128D

Benchmark the PDX auto-vectorized inner-product kernel vs. the SimSIMD explicit
SIMD kernel (with a FAISS reference) on 128D float32 vectors, on the target
architecture (AVX-512 here).

This drives the REAL PDX kernels, not a numpy stand-in. It loads the compiled
`kernels.so` from the PDX repo's `benchmarks/kernels_playground/` via cppyy and
calls two standalone C++ kernels over synthetic 128D float32 data:

  - F32_PDX_IP  : PDX auto-vectorized inner-product kernel over the PDX columnar
                  layout (block size 64; `row_major_to_pdx`).  (kernels.cpp:291)
  - F32_SIMD_IP : the inlined SimSIMD Skylake AVX-512 inner-product kernel.
                  (kernels.cpp:86 — `simsimd_ip_f32_skylake_cycle`)

Plus a FAISS exhaustive-knn reference (faiss-cpu, METRIC_INNER_PRODUCT). All run
single-threaded so we compare SIMD kernels like-for-like (no BLAS / no threading
is involved — both kernels are explicit single-threaded SIMD scans).

Each standalone call already sweeps all `n_queries` queries against all `N`
vectors inside C++, so one call is a large work unit (timing item a). We take the
MIN over many repeats (item b: the noise is one-sided, the min is the true
compute floor). Per-query latency = batch_min / n_queries.

Decision gate: confirm >=1.5x kernel speedup (simd / pdx). If <1.2x at 128D, the
PDX layout advantage is too thin to justify the storage change.

Prereqs (already done once):
  - PDX installed:  CXX=/usr/bin/clang++ uv pip install .   (from the PDX repo)
  - cppyy + faiss installed in the venv
  - kernels.so compiled:
        cd PDX/benchmarks/kernels_playground
        clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -o kernels.so kernels.cpp

Run:
    source "$HOME/Data Engineering/research/.venv/bin/activate"
    python "$HOME/Data Engineering/research/preliminaries/03_pdx_kernel/pdx_kernel_speedup.py"
"""
import os, sys, time, pathlib
os.environ.setdefault("EXTRA_CLING_ARGS", "-O3 -march=native")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE        = pathlib.Path(__file__).parent
FIGURES_DIR = HERE / "figures"; FIGURES_DIR.mkdir(exist_ok=True)

# Locate the PDX kernels playground (holds the compiled kernels.so).
REPO_ROOT    = HERE.parents[2]                       # /home/martan/Data Engineering
KERNELS_DIR  = REPO_ROOT / "PDX" / "benchmarks" / "kernels_playground"
KERNELS_SO   = KERNELS_DIR / "kernels.so"

DIM        = 128
# N must be a multiple of the PDX block size (64) for row_major_to_pdx.
N_VECTORS  = [1_024, 5_120, 10_048, 50_048, 100_032]
KNN_VALUES = [10, 100]
N_QUERIES  = 100
N_WARMUP   = 5
N_REPEATS  = 15
PDX_BLOCK  = 64

# ── Regime analysis (explains *why* 128D is marginal) ─────────────────────────
# The columnar-layout advantage is a COMPUTE-side win, so it only shows up when
# (a) the dimension is high enough to amortize per-block overhead, and
# (b) the candidate set is cache-resident (otherwise both kernels hit the DRAM
#     bandwidth roofline and converge to ~1.0x).
DIM_SWEEP    = [128, 256, 512, 768, 1024, 1536]  # vary D at fixed cache-resident N
DIM_SWEEP_N  = 8_192
N_SWEEP      = [512, 2_048, 8_192, 32_768, 131_072]  # vary N at fixed D=128 (cache→DRAM)
N_SWEEP_DIM  = 128
REGIME_KNN   = 10

rng = np.random.default_rng(42)


# ── cppyy: load the real PDX / SimSIMD kernels ────────────────────────────────

def load_kernels():
    import cppyy
    if not KERNELS_SO.exists():
        sys.exit(f"kernels.so not found at {KERNELS_SO}\n"
                 f"Compile it:\n  cd {KERNELS_DIR}\n"
                 f"  clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -o kernels.so kernels.cpp")
    cppyy.load_library(str(KERNELS_SO))
    cppyy.cppdef("""
    template<typename T> struct KNNCandidate { uint32_t index; float distance; };
    template<> struct KNNCandidate<float> { uint32_t index; float distance; };
    enum VectorSearchKernel {
        F32_SIMD_IP, F32_SIMD_L2, F32_PDX_IP, F32_PDX_L2,
        U8_SIMD_L2, U8_SIMD_IP, U8_PDX_L2, U8_PDX_IP
    };
    std::vector<KNNCandidate<float>> standalone_f32(
        const VectorSearchKernel kernel,
        const float *first_vector, const float *second_vector, const size_t d,
        const size_t num_queries, const size_t num_vectors, const size_t knn);
    """)
    return cppyy


def row_major_to_pdx(vectors: np.ndarray, block_size: int = PDX_BLOCK) -> np.ndarray:
    """Pack a row-major (N, d) float32 array into the PDX columnar block layout."""
    V, dims = vectors.shape
    assert V % block_size == 0, f"N={V} must be a multiple of block_size={block_size}"
    out = np.empty(V * dims, dtype=vectors.dtype)
    off = 0
    for i in range(V // block_size):
        chunk = vectors[off:off + block_size, :]
        out[i * block_size * dims:(i + 1) * block_size * dims] = chunk.flatten(order="F")
        off += block_size
    return out


# ── timing ────────────────────────────────────────────────────────────────────

def batch_min_ms_per_query(call_full_sweep, n_queries,
                           warmup=N_WARMUP, repeats=N_REPEATS) -> float:
    """One call = full sweep of all queries (item a). Min over repeats (item b)."""
    for _ in range(warmup):
        call_full_sweep()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        call_full_sweep()
        best = min(best, time.perf_counter() - t0)
    return best / n_queries * 1_000.0


# ── benchmark ─────────────────────────────────────────────────────────────────

def run_benchmarks(cppyy) -> dict:
    K = cppyy.gbl.VectorSearchKernel
    std_f32 = cppyy.gbl.standalone_f32

    import faiss
    from faiss.contrib.exhaustive_search import knn as faiss_knn
    faiss.omp_set_num_threads(1)   # single-threaded, like the SIMD/PDX kernels

    results = {knn: {"pdx": [], "simd": [], "faiss": [], "recall": []}
               for knn in KNN_VALUES}

    for N in N_VECTORS:
        print(f"N = {N:>7,} …", flush=True)
        X    = rng.random((N, DIM), dtype=np.float32)
        Q    = X[:N_QUERIES].copy()          # query i's true NN is vector i (recall@1)
        Xpdx = row_major_to_pdx(X, PDX_BLOCK)

        for knn in KNN_VALUES:
            simd_sweep  = lambda: std_f32(K.F32_SIMD_IP, X,    Q, DIM, N_QUERIES, N, knn)
            pdx_sweep   = lambda: std_f32(K.F32_PDX_IP,  Xpdx, Q, DIM, N_QUERIES, N, knn)
            faiss_sweep = lambda: faiss_knn(Q, X, knn, metric=faiss.METRIC_INNER_PRODUCT)

            t_simd  = batch_min_ms_per_query(simd_sweep,  N_QUERIES)
            t_pdx   = batch_min_ms_per_query(pdx_sweep,   N_QUERIES)
            t_faiss = batch_min_ms_per_query(faiss_sweep, N_QUERIES)

            # recall@1 of the PDX kernel: top-1 match index == query index
            r = pdx_sweep()
            matches = np.array([r[i].index for i in range(0, N_QUERIES * knn, knn)])
            recall = float(np.mean(matches == np.arange(N_QUERIES)))

            results[knn]["pdx"].append((N, t_pdx))
            results[knn]["simd"].append((N, t_simd))
            results[knn]["faiss"].append((N, t_faiss))
            results[knn]["recall"].append((N, recall))

            su = t_simd / t_pdx if t_pdx > 0 else float("nan")
            print(f"    knn={knn:>3}  pdx={t_pdx:8.4f}  simd={t_simd:8.4f}  "
                  f"faiss={t_faiss:8.4f}  ms/q  |  pdx/simd={su:5.2f}x  recall@1={recall:.3f}")
    return results


def print_gate(results):
    print("\n" + "=" * 70)
    print("DECISION GATE — PDX kernel speedup (simd / pdx), 128D float32 (real kernels)")
    print("=" * 70)
    header = f"{'N':>10}" + "".join(f"  knn={knn:>3}" for knn in KNN_VALUES)
    print(header); print("-" * len(header))
    for i, N in enumerate(N_VECTORS):
        row = f"{N:>10,}"
        for knn in KNN_VALUES:
            su = results[knn]["simd"][i][1] / results[knn]["pdx"][i][1]
            row += f"  {su:>6.2f}x"
        print(row)

    all_su = [results[knn]["simd"][i][1] / results[knn]["pdx"][i][1]
              for knn in KNN_VALUES for i in range(len(N_VECTORS))]
    peak = max(all_su)
    median = float(np.median(all_su))
    verdict = ("PASS (>=1.5x)" if median >= 1.5 else
               "MARGINAL (>=1.2x)" if median >= 1.2 else "FAIL (<1.2x)")
    print(f"\nGate: >=1.5x strong | >=1.2x marginal | <1.2x insufficient")
    print(f"Peak speedup: {peak:.2f}x   Median speedup: {median:.2f}x -> {verdict}")
    print(f"(PDX recall@1 ranges {min(v for _,v in results[KNN_VALUES[0]]['recall']):.3f}"
          f"–{max(v for _,v in results[KNN_VALUES[0]]['recall']):.3f})")


def plot(results):
    fig, axes = plt.subplots(1, len(KNN_VALUES) + 1, figsize=(15, 4.2))
    fig.suptitle(
        "Real PDX inner-product kernel vs SimSIMD (AVX-512) — 128D float32\n"
        f"{N_QUERIES} queries/sample · min of {N_REPEATS} sweeps · single-threaded kernels",
        fontsize=10, fontweight="bold",
    )
    colors = {"simd": "#888888", "pdx": "#009988", "faiss": "#EE7733"}
    labels = {"simd": "SimSIMD (explicit AVX-512)", "pdx": "PDX kernel (columnar)",
              "faiss": "FAISS knn (IP)"}

    for ax, knn in zip(axes[:len(KNN_VALUES)], KNN_VALUES):
        for method, color in colors.items():
            ns = [r[0] for r in results[knn][method]]
            ms = [r[1] for r in results[knn][method]]
            ax.plot(ns, ms, "o-", lw=1.8, ms=4, color=color, label=labels[method])
        ax.set_title(f"knn = {knn}", fontsize=9, fontweight="bold")
        ax.set_xlabel("N vectors"); ax.set_ylabel("Latency (ms/query, min)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, framealpha=0.8)

    ax2 = axes[-1]
    cmap = plt.colormaps["viridis"].resampled(len(KNN_VALUES))
    for idx, knn in enumerate(KNN_VALUES):
        ns = [r[0] for r in results[knn]["simd"]]
        su = [results[knn]["simd"][i][1] / results[knn]["pdx"][i][1] for i in range(len(ns))]
        ax2.plot(ns, su, "o-", lw=2, ms=5, color=cmap(idx), label=f"knn = {knn}")
    ax2.axhline(1.5, ls="--", lw=1, color="#CC3311", alpha=0.8, label="Gate: 1.5x")
    ax2.axhline(1.2, ls=":",  lw=1, color="#AA3377", alpha=0.6, label="Marginal: 1.2x")
    ax2.axhline(1.0, ls="-",  lw=0.8, color="#888888", alpha=0.5)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.set_xlabel("N vectors"); ax2.set_ylabel("Speedup (simd / pdx)")
    ax2.set_title("PDX speedup over SimSIMD", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7, framealpha=0.8)
    ax2.grid(ls="--", lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    out = FIGURES_DIR / "pdx_kernel_speedup.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")


# ── regime analysis: why 128D is marginal ────────────────────────────────────

def run_regime_analysis(cppyy) -> dict:
    """
    Two sweeps that explain the modest 128D number:
      dim_sweep : fixed cache-resident N, vary D  → layout win grows with D.
      n_sweep   : fixed D=128, vary N             → win vanishes past L3 (DRAM wall).
    Returns {'dim': [(D, simd, pdx, su)], 'n': [(N, footprint_mb, simd, pdx, su)]}.
    """
    K = cppyy.gbl.VectorSearchKernel
    std_f32 = cppyy.gbl.standalone_f32
    out = {"dim": [], "n": []}

    print("\nRegime analysis — effect of DIMENSION (N="
          f"{DIM_SWEEP_N:,}, cache-resident):")
    for D in DIM_SWEEP:
        X    = rng.random((DIM_SWEEP_N, D), dtype=np.float32)
        Q    = X[:N_QUERIES].copy()
        Xpdx = row_major_to_pdx(X, PDX_BLOCK)
        t_simd = batch_min_ms_per_query(
            lambda: std_f32(K.F32_SIMD_IP, X,    Q, D, N_QUERIES, DIM_SWEEP_N, REGIME_KNN), N_QUERIES)
        t_pdx  = batch_min_ms_per_query(
            lambda: std_f32(K.F32_PDX_IP,  Xpdx, Q, D, N_QUERIES, DIM_SWEEP_N, REGIME_KNN), N_QUERIES)
        su = t_simd / t_pdx
        out["dim"].append((D, t_simd, t_pdx, su))
        print(f"    D={D:>4}  simd={t_simd:7.4f}  pdx={t_pdx:7.4f} ms/q  speedup={su:5.2f}x")

    print(f"\nRegime analysis — effect of N (D={N_SWEEP_DIM}, cache→DRAM):")
    for N in N_SWEEP:
        X    = rng.random((N, N_SWEEP_DIM), dtype=np.float32)
        Q    = X[:N_QUERIES].copy()
        Xpdx = row_major_to_pdx(X, PDX_BLOCK)
        fp = N * N_SWEEP_DIM * 4 / 1e6
        t_simd = batch_min_ms_per_query(
            lambda: std_f32(K.F32_SIMD_IP, X,    Q, N_SWEEP_DIM, N_QUERIES, N, REGIME_KNN), N_QUERIES)
        t_pdx  = batch_min_ms_per_query(
            lambda: std_f32(K.F32_PDX_IP,  Xpdx, Q, N_SWEEP_DIM, N_QUERIES, N, REGIME_KNN), N_QUERIES)
        su = t_simd / t_pdx
        out["n"].append((N, fp, t_simd, t_pdx, su))
        print(f"    N={N:>7,}  {fp:6.2f}MB  simd={t_simd:7.4f}  pdx={t_pdx:7.4f} ms/q  speedup={su:5.2f}x")
    return out


def plot_regime(regime):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    fig.suptitle(
        "Why the 128D PDX kernel win is marginal — columnar layout is a compute-side optimization\n"
        "Left: advantage grows with dimension (128D is the floor)  ·  "
        "Right: advantage vanishes once the candidate set spills out of cache",
        fontsize=9.5, fontweight="bold",
    )

    # Panel 1: speedup vs dimension
    ds  = [r[0] for r in regime["dim"]]
    sus = [r[3] for r in regime["dim"]]
    ax1.plot(ds, sus, "o-", lw=2, ms=5, color="#009988")
    for d, s in zip(ds, sus):
        ax1.annotate(f"{s:.2f}x", (d, s), textcoords="offset points", xytext=(0, 7),
                     fontsize=7, ha="center", color="#007766")
    ax1.axhline(1.5, ls="--", lw=1, color="#CC3311", alpha=0.7, label="Gate: 1.5x")
    ax1.axhline(1.0, ls="-",  lw=0.8, color="#888", alpha=0.5)
    ax1.axvline(128, ls=":", lw=1, color="#AA3377", alpha=0.7)
    ax1.text(128, ax1.get_ylim()[0], " ColBERT\n 128D", fontsize=7, color="#AA3377", va="bottom")
    ax1.set_xlabel("Dimension"); ax1.set_ylabel("Speedup (simd / pdx)")
    ax1.set_title(f"Dimension sweep (N={DIM_SWEEP_N:,}, cache-resident)", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=7); ax1.grid(ls="--", lw=0.4, alpha=0.5); ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: speedup vs N (footprint), with L3 boundary
    ns  = [r[0] for r in regime["n"]]
    fps = [r[1] for r in regime["n"]]
    sus = [r[4] for r in regime["n"]]
    ax2.plot(fps, sus, "s-", lw=2, ms=5, color="#EE7733")
    for f, s in zip(fps, sus):
        ax2.annotate(f"{s:.2f}x", (f, s), textcoords="offset points", xytext=(0, 7),
                     fontsize=7, ha="center", color="#CC5500")
    ax2.axhline(1.5, ls="--", lw=1, color="#CC3311", alpha=0.7, label="Gate: 1.5x")
    ax2.axhline(1.0, ls="-",  lw=0.8, color="#888", alpha=0.5)
    ax2.axvspan(16, max(fps) * 1.3, color="#888", alpha=0.10)
    ax2.text(16, 1.02, " ~L3 limit → DRAM-bound", fontsize=7, color="#555", va="bottom")
    ax2.set_xscale("log")
    ax2.set_xlabel("Candidate-set footprint (MB, D=128)"); ax2.set_ylabel("Speedup (simd / pdx)")
    ax2.set_title("Cache→DRAM crossover (D=128)", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7); ax2.grid(ls="--", lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    out = FIGURES_DIR / "pdx_kernel_regime.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"Figure saved: {out}")


def main():
    print("Experiment 1: PDX inner-product kernel vs SimSIMD — 128D float32")
    print(f"N_QUERIES={N_QUERIES}, N_REPEATS={N_REPEATS} (min), single-threaded\n")
    cppyy = load_kernels()
    results = run_benchmarks(cppyy)
    print_gate(results)
    plot(results)
    regime = run_regime_analysis(cppyy)
    plot_regime(regime)


if __name__ == "__main__":
    main()
