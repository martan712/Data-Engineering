"""bondmaxsim.kernels — ctypes bindings to the C++ kernels + build helpers.

Single responsibility: load the compiled C++ shared libraries from
cpp/per_document_oracle/ (src/bondmaxsim/kernels/per_document.py) and
cpp/wide_block_maxsim_bond/ (src/bondmaxsim/kernels/wide_block.py), expose
their entry points as typed Python callables, and provide a build helper that
invokes the Makefile.

Ported artifact: C++ kernel interface from
  cpp/per_document_oracle/ (ported from research/preliminaries/09_maxsim_pruning/
  maxsim_kernels.cpp — accounting mode — and research/preliminaries/10_maxsim_pruning_opt/
  maxsim_kernels.cpp — throughput mode).
Stage 1 reference: docs/stage1_bond_maxsim_formalization.md §6 (cost model:
  accounting mode counts only live-set FMAs; throughput mode uses dense warmup),
  §8 items 3 (two counting modes) and 7 (extend PDX-sigmod BOND to MaxSim).
"""

from bondmaxsim.kernels.per_document import (
    load_per_document_oracle,
    run_accounting,
    run_full,
    run_throughput,
)
from bondmaxsim.kernels.wide_block import (
    load_wide_block_kernel,
    run_wide_block_accounting,
    run_wide_block_brute,
    run_wide_block_throughput,
)

__all__ = [
    "load_per_document_oracle",
    "run_accounting",
    "run_throughput",
    "run_full",
    "load_wide_block_kernel",
    "run_wide_block_accounting",
    "run_wide_block_brute",
    "run_wide_block_throughput",
]
