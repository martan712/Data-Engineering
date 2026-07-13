# Controlled Results

Only result artifacts produced under `docs/benchmark_protocol.md` belong here.
Each JSON must include raw timing samples, complete method configuration,
hardware/software metadata, input identifiers, the Git commit, and whether the
worktree was dirty.

The tracked artifacts currently cover:

- exact, FAISS-IVF, and PDX-IVF integration/validation/held-out pilots;
- a one-thread scaling and cross-engine equality follow-up;
- exact-safe BOND-MaxSim smoke, validation, and held-out runs;
- free exact-top-k oracle-seed and PCA mechanism ablations.
- matched held-out IVF and raw-order BOND transfer pilots on NFCorpus.

All files are still labeled `pilot_not_final_claim`. They were generated from a
dirty worktree while the benchmark implementation was being completed. Their
hashes and roles are recorded in `manifest.json`; the final release must rerun
the frozen configurations from a clean commit before changing this status.
