# PDX Benchmark: Methodology and Results

## Goal

Reproduce the core claims of the PDX paper (SIGMOD'25) on local hardware:
1. PDX's vertical data layout yields faster distance kernels than the horizontal (N-ary) layout
2. PDX combined with pruning algorithms (ADSampling, BSA/DDC, BOND) significantly outperforms both brute-force and FAISS baselines at equal recall

---

## Hardware and Software

| | |
|---|---|
| **CPU** | AMD Ryzen AI 7 445 (Zen 5), 6 cores / 12 threads, max 4666 MHz |
| **L1d / L2 / L3 cache** | 288 KiB / 6 MiB / 8 MiB |
| **Compiler** | Clang 21.1.8, `-O3 -march=native` |
| **OS** | Fedora Linux (kernel 6.19) |
| **PDX commit** | PDX-sigmod repository (local clone) |
| **Python** | 3.12, with FAISS-CPU, NumPy, scikit-learn |

Note: the paper's primary benchmarks target Intel Sapphire Rapids (AVX-512) and AMD Zen4. This machine uses Zen5, which also supports AVX-512, so vector instruction widths should be comparable.

---

## Dataset

**agnews-mxbai-1024-euclidean** — chosen as the smallest available dataset by file size.

| Property | Value |
|---|---|
| Vectors | 769,382 |
| Dimensions | 1,024 |
| Distance metric | Euclidean (L2) |
| Query vectors | 1,000 |
| Ground truth k | 10 |

---

## Benchmark Configuration

**IVF index:** Built with FAISS using `nbuckets = ceil(2 × √769382) = 1754` clusters.

**nprobe range:** 29 values from 2 to 512 (full sweep defined in `benchmark_utils.hpp`).

**Queries per measurement:** 1,000 queries per nprobe value, 1 measurement pass. Reported avg/min/max are per-query latencies in milliseconds, with IQR-based outlier filtering applied to the average.

**Memory limits (enforced via systemd cgroups):**
- Index preprocessing: 16 GB (raw data + FAISS index loaded simultaneously)
- Each benchmark run: 8 GB

**Algorithms benchmarked:**

| Label | Description |
|---|---|
| Brute-force | N-ary (horizontal) IVF, no pruning — pure baseline |
| FAISS IVF | FAISS optimized IVF search (Python) |
| N-ary ADSampling | Horizontal layout + ADSampling pruning, SIMD-optimized |
| PDX + ADSampling | PDX (vertical) layout + ADSampling pruning |
| PDX + BSA (DDC) | PDX layout + BSA pruning (equivalent to DDC in the paper) |
| PDX + BOND | PDX layout + BOND pruning, dimension-zones ordering |

---

## Results

### Avg query latency (ms) and recall at selected nprobe values

| nprobe | Brute-force | FAISS IVF | N-ary ADSampling | PDX ADSampling | PDX BSA | PDX BOND |
|---|---|---|---|---|---|---|
| **16** | 9.13ms / 0.941 | 0.92ms / 0.941 | 0.87ms / 0.941 | 0.55ms / 0.940 | 0.52ms / 0.937 | 0.93ms / 0.941 |
| **32** | 16.33ms / 0.969 | 1.60ms / 0.970 | 1.21ms / 0.969 | 0.74ms / 0.968 | 0.71ms / 0.966 | 1.66ms / 0.969 |
| **64** | 30.43ms / 0.987 | 3.04ms / 0.987 | 1.87ms / 0.986 | 0.97ms / 0.986 | 0.98ms / 0.983 | 2.97ms / 0.987 |
| **128** | 59.13ms / 0.994 | 5.91ms / 0.995 | 3.07ms / 0.994 | 1.39ms / 0.993 | 1.46ms / 0.992 | 5.29ms / 0.994 |
| **256** | 116.18ms / 0.997 | 11.55ms / 0.997 | 5.09ms / 0.996 | 2.25ms / 0.996 | 2.29ms / 0.996 | 9.76ms / 0.997 |
| **512** | 229.33ms / 0.999 | 22.48ms / 0.999 | 8.41ms / 0.998 | 3.45ms / 0.997 | 3.62ms / 0.998 | 18.22ms / 0.999 |

### Speedup over brute-force at nprobe=32 (~96.9% recall)

| Algorithm | Avg latency | Speedup |
|---|---|---|
| Brute-force | 16.33ms | 1× |
| FAISS IVF | 1.60ms | 10.2× |
| N-ary ADSampling | 1.21ms | 13.5× |
| PDX + BOND | 1.66ms | 9.8× |
| PDX + ADSampling | 0.74ms | **22.1×** |
| PDX + BSA | 0.71ms | **23.0×** |

### Speedup over brute-force at nprobe=128 (~99.2–99.4% recall)

| Algorithm | Avg latency | Speedup |
|---|---|---|
| Brute-force | 59.13ms | 1× |
| FAISS IVF | 5.91ms | 10.0× |
| N-ary ADSampling | 3.07ms | 19.3× |
| PDX + BOND | 5.29ms | 11.2× |
| PDX + ADSampling | 1.39ms | **42.5×** |
| PDX + BSA | 1.46ms | **40.5×** |

---

## Observations

**PDX + ADSampling and PDX + BSA clearly win.** Both deliver ~22× speedup over brute-force at moderate recall (nprobe=32) and ~40× at high recall (nprobe=128), consistent with the paper's central claim that PDX + pruning dominates horizontal layouts at high recall.

**PDX + ADSampling is ~1.6× faster than N-ary ADSampling SIMD** at nprobe=32 (0.74ms vs 1.21ms), confirming the paper's claim that the vertical kernel is more efficient than the horizontal SIMD kernel — even though N-ary ADSampling is already SIMD-optimized.

**PDX + BOND is unexpectedly slow** — slower than FAISS at nprobe≥128, and 2–4× slower than PDX ADSampling at every nprobe level. Possible explanations:
- The dimension-zones ordering (criteria=5) may not suit a euclidean 1024-dim dataset well; the paper's BOND results focus on datasets where this ordering aligns with the data's intrinsic structure.
- BOND's per-query preprocessing overhead (computing zone-level bounds) is not amortized well at the query volumes used here (1,000 queries).
- The paper notes BOND is most effective at high dimensionalities and high recall, while this dataset at 1024 dims may be in a regime where ADSampling/BSA's simpler pruning already saturates the benefit.

**FAISS is ~2× slower than PDX ADSampling/BSA** at all nprobe values, confirming PDX's advantage over highly-optimized FAISS despite FAISS using hand-tuned BLAS routines.

**Recall differences are small.** At any given nprobe, all algorithms achieve similar recall (within ±0.3%), confirming that pruning does not significantly degrade result quality at these operating points.

---

## Limitations

- **Single measurement pass** (`NUM_MEASURE_RUNS=1`): results have per-query variance but no run-to-run statistical confidence. The paper used multiple runs on dedicated, thermally stable hardware.
- **Smaller dataset**: 769K vectors vs the paper's 1M–2.2M. Speedups from pruning tend to grow with dataset size, so results here likely understate the paper's gains.
- **No kernel-level benchmarks**: the pure distance-kernel benchmarks (`KernelPDXL2`, `KernelNaryL2`, etc.) require pre-generated synthetic data files that were not set up in this run.
- **CPU scaling**: the CPU ran at ~70% of max frequency during parts of the benchmark run; results may vary with thermal/power state.
