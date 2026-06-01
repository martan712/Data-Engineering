"""
Experiment 2: Synchronized multi-token scan throughput

Benchmark two approaches to ColBERT MaxSim scoring against a pool of candidate
documents (the PLAID shortlist re-scoring scenario):

  Standard   — m separate matmuls, one per query token:
                 for each query token q_i: sim_i = q_i @ D.T → max over doc tokens

  Synchronized — one batched matmul, all m query tokens share one pass:
                 SIM = Q @ D.T  (m × N_dtokens_total), then reshape + max

Both produce identical scores. The synchronized version reduces BLAS call
overhead and improves cache reuse when m is large.

Vary:
  m          — number of query tokens: 10, 20, 30
  block_size — number of candidate documents per cache block: 64, 256, 1024

Decision gate: synchronized / standard speedup ≥ 1.5× at m=30.
               If < 1.2×, the shared-scan argument does not hold in practice.

Run from anywhere:
    source "$HOME/Data Engineering/research/.venv/bin/activate"
    python "$HOME/Data Engineering/research/preliminaries/04_multitokenscan/multitokenscan.py"

Requires: numpy, matplotlib
"""
import pathlib, time
import numpy as np
import matplotlib.pyplot as plt

HERE        = pathlib.Path(__file__).parent
FIGURES_DIR = HERE / "figures"; FIGURES_DIR.mkdir(exist_ok=True)

DIM           = 128
N_DOC_TOKENS  = 64          # avg tokens per document (fixed padding)
N_CANDIDATES  = 8_192       # PLAID shortlist size (typical)
M_VALUES      = [10, 20, 30]
BLOCK_SIZES   = [64, 256, 1_024]
N_WARMUP      = 5
N_REPEATS     = 30

rng = np.random.default_rng(42)


def timed_ms(fn, warmup=N_WARMUP, repeats=N_REPEATS) -> float:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1_000)
    return float(np.median(t))


def score_standard(Q, D_flat, N_cands, n_dtokens):
    """
    Q      : (m, 128)
    D_flat : (N_cands * n_dtokens, 128)  — all doc tokens, row-major

    Scores each query token independently (m separate matmuls).
    Returns (m, N_cands) MaxSim contribution per query token.
    """
    scores = np.empty((Q.shape[0], N_cands), dtype=np.float32)
    for i in range(Q.shape[0]):
        sim = (Q[i:i+1] @ D_flat.T).reshape(N_cands, n_dtokens)  # (N_cands, n_dtokens)
        scores[i] = sim.max(axis=1)
    return scores.sum(axis=0)   # final MaxSim = sum over query tokens


def score_synchronized(Q, D_flat, N_cands, n_dtokens, block_size):
    """
    All m query tokens share one pass over document token blocks.
    Returns (N_cands,) MaxSim scores.
    """
    m = Q.shape[0]
    doc_maxsim = np.zeros(N_cands, dtype=np.float32)

    for start in range(0, N_cands, block_size):
        end    = min(start + block_size, N_cands)
        B      = end - start
        D_blk  = D_flat[start * n_dtokens : end * n_dtokens]  # (B*n_dtokens, 128)
        sim    = Q @ D_blk.T                                   # (m, B*n_dtokens)
        sim    = sim.reshape(m, B, n_dtokens)                  # (m, B, n_dtokens)
        doc_maxsim[start:end] = sim.max(axis=2).sum(axis=0)   # sum over query tokens

    return doc_maxsim


def run_benchmarks():
    """Returns results[m][block_size] = {'standard_ms': float, 'sync_ms': float}."""
    D_flat = rng.standard_normal((N_CANDIDATES * N_DOC_TOKENS, DIM)).astype(np.float32)

    results = {m: {bs: {} for bs in BLOCK_SIZES} for m in M_VALUES}

    for m in M_VALUES:
        Q = rng.standard_normal((m, DIM)).astype(np.float32)

        t_std = timed_ms(lambda: score_standard(Q, D_flat, N_CANDIDATES, N_DOC_TOKENS))
        print(f"m={m:>2}  standard = {t_std:.1f} ms")

        for bs in BLOCK_SIZES:
            t_sync = timed_ms(lambda: score_synchronized(Q, D_flat, N_CANDIDATES, N_DOC_TOKENS, bs))
            su = t_std / t_sync if t_sync > 0 else float("nan")
            print(f"       block={bs:>5}  sync = {t_sync:.1f} ms  speedup = {su:.2f}×")
            for bs2 in BLOCK_SIZES:
                results[m][bs2]["standard_ms"] = t_std
            results[m][bs]["sync_ms"] = t_sync

    return results


def print_gate(results):
    print("\n" + "=" * 60)
    print("DECISION GATE — synchronized / standard speedup")
    print(f"N_candidates={N_CANDIDATES}, n_doc_tokens={N_DOC_TOKENS}, DIM={DIM}")
    print("=" * 60)
    header = f"{'m':>4}" + "".join(f"  bs={bs:>5}" for bs in BLOCK_SIZES)
    print(header); print("-" * len(header))
    for m in M_VALUES:
        row = f"{m:>4}"
        for bs in BLOCK_SIZES:
            t_std  = results[m][bs]["standard_ms"]
            t_sync = results[m][bs]["sync_ms"]
            su     = t_std / t_sync
            row += f"  {su:>7.2f}×"
        print(row)

    all_su = [results[m][bs]["standard_ms"] / results[m][bs]["sync_ms"]
              for m in M_VALUES for bs in BLOCK_SIZES]
    best = max(all_su)
    at_m30 = max(results[30][bs]["standard_ms"] / results[30][bs]["sync_ms"] for bs in BLOCK_SIZES)
    verdict = "PASS (≥1.5×)" if at_m30 >= 1.5 else ("MARGINAL (≥1.2×)" if at_m30 >= 1.2 else "FAIL (<1.2×)")
    print(f"\nBest speedup at m=30: {at_m30:.2f}× → {verdict}")
    print(f"Overall peak speedup : {best:.2f}×")


def plot(results):
    fig, axes = plt.subplots(1, len(BLOCK_SIZES), figsize=(13, 4.5), sharey=True)
    fig.suptitle(
        "Synchronized multi-token scan vs per-token-sequential scoring\n"
        f"N_candidates={N_CANDIDATES:,}  ·  doc_tokens={N_DOC_TOKENS}  ·  DIM={DIM}  ·  median {N_REPEATS} runs",
        fontsize=10, fontweight="bold",
    )
    colors = {"standard": "#888888", "sync": "#009988"}

    for ax, bs in zip(axes, BLOCK_SIZES):
        ms_std  = [results[m][bs]["standard_ms"] for m in M_VALUES]
        ms_sync = [results[m][bs]["sync_ms"]      for m in M_VALUES]
        speedups = [s / sy for s, sy in zip(ms_std, ms_sync)]

        ax2 = ax.twinx()
        ax.bar([x - 0.18 for x in range(len(M_VALUES))], ms_std,  width=0.35,
               color=colors["standard"], label="Standard (per token)")
        ax.bar([x + 0.18 for x in range(len(M_VALUES))], ms_sync, width=0.35,
               color=colors["sync"],     label="Synchronized")
        ax2.plot(range(len(M_VALUES)), speedups, "D--", color="#CC3311", ms=6, lw=1.5, label="Speedup")
        ax2.axhline(1.5, ls=":", lw=1, color="#CC3311", alpha=0.5)
        ax2.set_ylabel("Speedup (×)", fontsize=8)
        ax2.set_ylim(0, max(speedups) * 1.5)

        ax.set_xticks(range(len(M_VALUES)))
        ax.set_xticklabels([f"m={m}" for m in M_VALUES])
        ax.set_title(f"block_size = {bs:,}", fontsize=9, fontweight="bold")
        ax.set_ylabel("Latency (ms, median)" if ax is axes[0] else "")
        ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper left")
            ax2.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out = FIGURES_DIR / "multitokenscan_speedup.png"
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"\nFigure saved: {out}")


def main():
    print(f"Synchronized multi-token scan benchmark")
    print(f"N_candidates={N_CANDIDATES}, N_DOC_TOKENS={N_DOC_TOKENS}, DIM={DIM}\n")
    results = run_benchmarks()
    print_gate(results)
    plot(results)


if __name__ == "__main__":
    main()
