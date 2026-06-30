"""bondmaxsim.kernels — ctypes bindings to the C++ kernels + build helpers.

Single responsibility: load the compiled C++ shared libraries from
cpp/per_document_oracle/ and (eventually) cpp/wide_block_maxsim_bond/, expose
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
