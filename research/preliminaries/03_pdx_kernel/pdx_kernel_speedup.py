"""
Experiment 1: PDX kernel speedup on 128D

Benchmark PDX auto-vectorized inner product kernel (columnar / col-major layout)
vs. standard row-major inner product (BLAS/numpy) on 128D float32 vectors.

The core PDX layout advantage for partial inner products:
  Row-major slice A[:, :k]  — non-contiguous, stride = 128 floats  (bad for prefetch)
  Col-major slice B[:k, :]  — contiguous, stride = 1               (SIMD-friendly)

Three cases measured per (N, k):
  1. row      : Q @ A[:, :k].T          — strided, BLAS may internally copy
  2. pdx      : Q[:, :k] @ B[:k, :]    — contiguous, no copy
  3. row+copy : Q @ C[:k, :].T         — explicit copy to contiguous before BLAS

Decision gate: ≥1.5x speedup (pdx / row). If < 1.2x at 128D, the layout
advantage is too thin to justify the storage change.

Run from anywhere:
    source "$HOME/Data Engineering/research/.venv/bin/activate"
    python "$HOME/Data Engineering/research/preliminaries/03_pdx_kernel/pdx_kernel_speedup.py"

Requires: numpy, matplotlib
"""
import pathlib, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE        = pathlib.Path(__file__).parent
FIGURES_DIR = HERE / "figures"; FIGURES_DIR.mkdir(exist_ok=True)

DIM        = 128
N_QUERIES  = 32      # ColBERT: ~32 query tokens
N_VECTORS  = [1_000, 5_000, 10_000, 50_000, 100_000]
K_BUDGETS  = [16, 32, 64, 128]
N_WARMUP   = 5
N_REPEATS  = 30

rng = np.random.default_rng(42)


def timed_ms(fn, warmup=N_WARMUP, repeats=N_REPEATS) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1_000)
    return float(np.median(times))


def run_benchmarks() -> dict:
    """
    Returns results[k][method] = list of (N, ms) pairs.
    """
    results = {k: {"row": [], "pdx": [], "row_copy": []} for k in K_BUDGETS}

    for N in N_VECTORS:
        print(f"N = {N:>7,} …", end="  ", flush=True)

        # Row-major:  (N, 128)
        A_row = rng.standard_normal((N, DIM)).astype(np.float32)
        # Col-major (PDX): (128, N)
        A_col = np.asfortranarray(A_row.T)       # (128, N), Fortran order = column-contiguous
        # Contiguous row for copy baseline: (128, N) C-order
        A_col_c = np.ascontiguousarray(A_row.T)  # same data, C-order

        Q = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

        for k in K_BUDGETS:
            # Slices (created once outside timed loop to measure only compute)
            A_partial_row  = A_row[:, :k]           # non-contiguous view (stride = DIM)
            A_partial_col  = A_col[:k, :]           # contiguous view (Fortran col slice)
            A_partial_copy = np.ascontiguousarray(A_row[:, :k])  # explicit copy
            Q_k            = Q[:, :k]               # non-contiguous query slice

            t_row  = timed_ms(lambda: Q_k @ A_partial_row.T)
            t_pdx  = timed_ms(lambda: Q_k @ A_partial_col)
            t_copy = timed_ms(lambda: Q_k @ A_partial_copy.T)

            results[k]["row"].append((N, t_row))
            results[k]["pdx"].append((N, t_pdx))
            results[k]["row_copy"].append((N, t_copy))

            speedup = t_row / t_pdx if t_pdx > 0 else float("nan")
            print(f"k={k}  {speedup:.2f}x", end="  ", flush=True)

        print()

    return results


def print_gate(results):
    print("\n" + "=" * 60)
    print("DECISION GATE — PDX speedup (row / pdx) at each (N, k)")
    print("=" * 60)
    header = f"{'N':>10}" + "".join(f"  k={k:>3}" for k in K_BUDGETS)
    print(header)
    print("-" * len(header))
    for i, N in enumerate(N_VECTORS):
        row = f"{N:>10,}"
        for k in K_BUDGETS:
            t_row = results[k]["row"][i][1]
            t_pdx = results[k]["pdx"][i][1]
            su = t_row / t_pdx if t_pdx > 0 else float("nan")
            row += f"  {su:>5.2f}x"
        print(row)

    print("\nGate: ≥1.5x = strong  |  ≥1.2x = marginal  |  <1.2x = insufficient")
    max_su = max(
        results[k]["row"][i][1] / results[k]["pdx"][i][1]
        for k in K_BUDGETS for i in range(len(N_VECTORS))
    )
    verdict = "PASS (≥1.5x)" if max_su >= 1.5 else ("MARGINAL (≥1.2x)" if max_su >= 1.2 else "FAIL (<1.2x)")
    print(f"\nPeak speedup: {max_su:.2f}x → {verdict}")


def plot(results):
    fig, axes = plt.subplots(1, len(K_BUDGETS), figsize=(14, 4), sharey=False)
    fig.suptitle(
        "PDX col-major vs row-major inner product — 128D float32\n"
        f"Query batch = {N_QUERIES} tokens  ·  median of {N_REPEATS} runs",
        fontsize=10, fontweight="bold",
    )

    colors = {"row": "#888888", "pdx": "#009988", "row_copy": "#EE7733"}
    labels = {"row": "Row-major (strided)", "pdx": "Col-major / PDX (contiguous)", "row_copy": "Row-major + copy"}

    for ax, k in zip(axes, K_BUDGETS):
        for method, color in colors.items():
            ns  = [r[0] for r in results[k][method]]
            ms  = [r[1] for r in results[k][method]]
            ax.plot(ns, ms, "o-", lw=1.8, ms=4, color=color, label=labels[method])

        ax.set_title(f"k = {k} dims", fontsize=9, fontweight="bold")
        ax.set_xlabel("N vectors")
        ax.set_ylabel("Latency (ms, median)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=7, framealpha=0.8)

    # Speedup inset panel
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.suptitle("PDX speedup over row-major (row / pdx latency)", fontsize=10, fontweight="bold")
    cmap = plt.cm.get_cmap("viridis", len(K_BUDGETS))
    for idx, k in enumerate(K_BUDGETS):
        ns       = [r[0] for r in results[k]["row"]]
        speedups = [results[k]["row"][i][1] / results[k]["pdx"][i][1] for i in range(len(ns))]
        ax2.plot(ns, speedups, "o-", lw=2, ms=5, color=cmap(idx), label=f"k = {k}")
    ax2.axhline(1.5, ls="--", lw=1, color="#CC3311", alpha=0.8, label="Gate: 1.5×")
    ax2.axhline(1.2, ls=":",  lw=1, color="#AA3377", alpha=0.6, label="Marginal: 1.2×")
    ax2.axhline(1.0, ls="-",  lw=0.8, color="#888888", alpha=0.5)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.set_xlabel("N vectors"); ax2.set_ylabel("Speedup (×)")
    ax2.legend(fontsize=8, framealpha=0.8)
    ax2.grid(ls="--", lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    out1 = FIGURES_DIR / "pdx_kernel_latency.png"
    out2 = FIGURES_DIR / "pdx_kernel_speedup.png"
    fig.tight_layout(); fig.savefig(out1, dpi=180, bbox_inches="tight"); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(out2, dpi=180, bbox_inches="tight"); plt.close(fig2)
    print(f"\nFigures saved:\n  {out1}\n  {out2}")


def main():
    print(f"Benchmarking PDX vs row-major inner products — 128D float32")
    print(f"N_QUERIES={N_QUERIES}, N_REPEATS={N_REPEATS}\n")
    results = run_benchmarks()
    print_gate(results)
    plot(results)


if __name__ == "__main__":
    main()
