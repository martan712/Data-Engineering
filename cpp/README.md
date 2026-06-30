# C++ Kernels — Build Instructions and Stage Mapping

This directory contains the project's C++ contribution: two kernels that
implement BOND-MaxSim at different granularities and purposes.

## Kernel Overview

### `per_document_oracle/`

**Purpose:** Exact-safe per-document pruning oracle.

**Algorithm:** BOND-MaxSim with per-document granularity (Stage 1 "Option B",
§5.2). Each document's tokens are laid out dimension-major within the document
(`docs[offset*D + z*n_d + j]`). The kernel scans dimensions one at a time,
maintaining pair intervals `[L_ij, U_ij]` (§2.2), the document upper bound
`UB_d = sum_i max_j U_ij` (§2.3), and the token lower bound `L_i = max_j L_ij`
(§2.4) per document. The token-pruning survival invariant (§2.4) guarantees that
`shrink=1` is exact-safe under unit normalization, exact arithmetic, and
set-equality top-k semantics (§2.5).

**Ported from:** `archive/preliminaries/09_maxsim_pruning/maxsim_kernels.cpp`
(accounting mode — true-pruning cells counter) and
`archive/preliminaries/10_maxsim_pruning_opt/maxsim_kernels.cpp` (throughput
mode — dense warmup before switching to survivor-only scan).

**Two modes (Stage 1 §6):**
- **Accounting mode** (exp-09 style): scans only the live token set from
  dimension 0; `cells` counter is the true algorithmic-work signal,
  hardware-independent and cross-implementation comparable.
- **Throughput mode** (exp-10 style): dense all-token warmup over the first
  `D/4` dimensions (nothing prunes early due to MaxSim score compression), then
  positional survivor-only scan; reports honest wall-clock ms/query.

**Stage 1 sections:** §5.2 (Option B / oracle role), §2 (exact-safe proof), §6
(cost model and two-mode split), §8 items 1–3 (blocking checks: normalization
guard, exact-agreement, two counting modes).

**Experiments:** `experiments/stage2_testbed/` (exp-09 accounting smoke,
exp-10 throughput smoke), `experiments/stage3_mechanism/` (bound-slack,
pruning-rate, order ablation arms).

---

### `wide_block_maxsim_bond/`

**Purpose:** Stage 2 deliverable — multi-vector MaxSim extension of PDX-sigmod BOND.

**Algorithm:** Faithful PDX-BOND extended to MaxSim (Stage 1 §5.3). Lays out
document tokens dimension-major **across many documents at once** (a real PDX
vectorgroup of tokens), then runs BOND dimension-incrementally over that wide
column. Bounds are exactly those in §2: per-pair intervals `[L_ij, U_ij]`
aggregated to the per-document upper bound `UB_d = sum_i max_j U_ij`. Per-document
bound state (`L_i` and the live set) is maintained via `doc_offsets` even though
the block interleaves tokens from many documents (§2.4 note). The synchronized
wide scan means `tau_k` must be provided via staged finalization or a seed (§4.4).

**Substrate:** Extends `extern/PDX-sigmod/include/pdx/bond.hpp`
(`IndexPDXBONDFlat`, `PDXBondSearcher`) — the single-vector L2 PDXearch variant
at commit `fdc62f2`. The extension changes L2 single-vector distance to
inner-product MaxSim with the two-level (token → document) bound from §2.

**Stage 1 sections:** §5.1 (PDX-sigmod BOND reference implementation), §5.3
(faithful wide-block design), §4.4 (threshold dynamics in synchronized scan —
seeded threshold is first-class), §4.5 (per-query dimension order via
`GetDimensionsAccessOrder`), §8 item 7 (Stage 2 deliverable: extend PDX-sigmod
BOND to MaxSim).

**Experiments:** `experiments/stage3_mechanism/` (BOND vs ADSampling ablation
for MaxSim, exhaustive-but-pruned), `experiments/stage4_integration/` (wide-block
kernel as the dimension-pruning arm vs IVF/PLAID candidate-generation arms).

---

## Build

```bash
# From repo root — requires submodules initialized at pinned commits
./setup.sh

# Or build just one kernel:
make -C cpp/per_document_oracle/
make -C cpp/wide_block_maxsim_bond/
```

Dependencies: C++17 compiler, `extern/PDX-sigmod/` submodule initialized
(for `wide_block_maxsim_bond`; `per_document_oracle` is self-contained).

Output: `per_document_oracle.so` and `wide_block_maxsim_bond.so` loaded at
runtime by `src/bondmaxsim/kernels/bindings.py`.
