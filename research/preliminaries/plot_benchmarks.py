"""
Bar charts comparing PDX algorithms vs baselines.
Reads CSVs from benchmarks/ZEN5-Martan/, writes figures/ alongside this script.
"""
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────��────────────────────────────────────────
HERE        = pathlib.Path(__file__).parent
BENCH_DIR   = HERE / "benchmarks" / "ZEN5-Martan"
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Load and normalise all CSVs ───────────────────────────────────────────────
CSV_FILES = {
    "Brute-force":      "IVF_BRUTEFORCE.csv",
    "FAISS IVF":        "IVF_FAISS.csv",
    "N-ary ADSampling": "IVF_NARY_ADSAMPLING_SIMD.csv",
    "PDX ADSampling":   "IVF_PDX_ADSAMPLING.csv",
    "PDX BSA (DDC)":    "IVF_PDX_BSA.csv",
    "PDX BOND":         "IVF_PDX_BOND.csv",
}

ALGO_COLORS = {
    "Brute-force":      "#888888",
    "FAISS IVF":        "#4477AA",
    "N-ary ADSampling": "#EE7733",
    "PDX ADSampling":   "#009988",
    "PDX BSA (DDC)":    "#CC3311",
    "PDX BOND":         "#AA3377",
}

def load_csv(label: str, filename: str) -> pd.DataFrame:
    path = BENCH_DIR / filename
    if not path.exists():
        print(f"  WARNING: {filename} not found, skipping {label}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalise ivf_nprobe column (different position in FAISS vs C++ CSVs)
    if "ivf_nprobe" not in df.columns:
        df = df.rename(columns={"ivf_nprobe": "ivf_nprobe"})
    df["label"] = label
    return df

frames = [load_csv(lbl, fn) for lbl, fn in CSV_FILES.items()]
data = pd.concat([f for f in frames if not f.empty], ignore_index=True)
data["ivf_nprobe"] = pd.to_numeric(data["ivf_nprobe"], errors="coerce")
data["recall"]     = pd.to_numeric(data["recall"],     errors="coerce")
data["avg"]        = pd.to_numeric(data["avg"],         errors="coerce")

datasets = data["dataset"].dropna().unique()

# ── Helpers ──────────────────────────────────────────────────────────���────────
ALGO_ORDER = list(CSV_FILES.keys())

def nearest_nprobe(df: pd.DataFrame, label: str, target_nprobe: int) -> pd.Series | None:
    sub = df[(df["label"] == label) & (df["ivf_nprobe"] == target_nprobe)]
    return sub.iloc[0] if not sub.empty else None

def bar_chart_at_nprobe(ds: str, nprobe: int, ax: plt.Axes):
    sub = data[data["dataset"] == ds]
    algos, latencies, recalls = [], [], []
    for lbl in ALGO_ORDER:
        row = nearest_nprobe(sub, lbl, nprobe)
        if row is not None:
            algos.append(lbl)
            latencies.append(row["avg"])
            recalls.append(row["recall"])

    colors = [ALGO_COLORS[a] for a in algos]
    x = np.arange(len(algos))
    bars = ax.bar(x, latencies, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Annotate recall above each bar
    for bar, rec in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(latencies) * 0.01,
                f"{rec:.3f}", ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Avg query latency (ms)", fontsize=9)
    ax.set_title(f"{ds}\nnprobe={nprobe}  (recall annotated above bars)", fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

def speedup_chart(ds: str, nprobe: int, ax: plt.Axes):
    sub = data[data["dataset"] == ds]
    baseline_row = nearest_nprobe(sub, "Brute-force", nprobe)
    if baseline_row is None:
        return
    baseline_lat = baseline_row["avg"]

    algos, speedups, recalls = [], [], []
    for lbl in ALGO_ORDER:
        if lbl == "Brute-force":
            continue
        row = nearest_nprobe(sub, lbl, nprobe)
        if row is not None:
            algos.append(lbl)
            speedups.append(baseline_lat / row["avg"])
            recalls.append(row["recall"])

    colors = [ALGO_COLORS[a] for a in algos]
    x = np.arange(len(algos))
    bars = ax.bar(x, speedups, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    for bar, spd in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(speedups) * 0.01,
                f"{spd:.1f}×", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(1, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Speedup over brute-force", fontsize=9)
    ax.set_title(f"{ds}\nnprobe={nprobe}  (speedup vs brute-force)", fontsize=9)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

def pareto_chart(ds: str, ax: plt.Axes, recall_min: float = 0.90):
    sub = data[(data["dataset"] == ds) & (data["recall"] >= recall_min)]
    for lbl in ALGO_ORDER:
        rows = sub[sub["label"] == lbl].sort_values("recall")
        if rows.empty:
            continue
        ax.plot(rows["recall"], rows["avg"],
                marker="o", markersize=3.5, linewidth=1.5,
                label=lbl, color=ALGO_COLORS[lbl])
    ax.set_xlabel("Recall@10", fontsize=9)
    ax.set_ylabel("Avg query latency (ms, log scale)", fontsize=9)
    ax.set_title(f"{ds}\nRecall vs. latency — recall ≥ {recall_min}", fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.6, which="both")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# ── Generate figures ──────────────────────────────────────────────────────────
for ds in datasets:
    safe_ds = ds.replace("/", "_")

    # Figure 1: latency bar chart at nprobe=32 and nprobe=128
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Avg query latency by algorithm", fontsize=11, fontweight="bold")
    bar_chart_at_nprobe(ds, 32,  axes[0])
    bar_chart_at_nprobe(ds, 128, axes[1])
    fig.tight_layout()
    out = FIGURES_DIR / f"{safe_ds}_latency_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")

    # Figure 2: speedup bar chart at nprobe=32 and nprobe=128
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Speedup over brute-force", fontsize=11, fontweight="bold")
    speedup_chart(ds, 32,  axes[0])
    speedup_chart(ds, 128, axes[1])
    fig.tight_layout()
    out = FIGURES_DIR / f"{safe_ds}_speedup_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")

    # Figure 3: Pareto recall vs latency line chart
    fig, ax = plt.subplots(figsize=(8, 5))
    pareto_chart(ds, ax)
    fig.tight_layout()
    out = FIGURES_DIR / f"{safe_ds}_pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")

print(f"\nAll figures written to: {FIGURES_DIR}")
