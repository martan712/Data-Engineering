# Final Controlled Results

These five artifacts were generated from clean commit `0cc6145` in one Ubuntu
WSL2 environment. Every run records `dirty=false`, initial CPU affinity
`[0, 2, 4, 6]`, four OpenMP/BLAS threads, one warm-up, five measured interleaved
runs, package versions, input hashes, stage timings, and raw outer-timer samples.

The online boundary starts with resident normalized query token vectors and ends
with ranked document IDs. Encoding and index construction are outside this
boundary and reported separately. Therefore, reported ratios describe online
search/reranking for the named arms, not end-to-end retrieval-system speedup.

The JSON `status` field retains the runner's `pilot_not_final_claim` value for
schema compatibility. This directory's manifest is the release disposition:
only artifacts whose Git metadata is clean and matches `source_commit` are
accepted here.

The BOND comparison uses float64 products and accumulation in both exhaustive
and exact-safe kernels. The IVF comparison retains the float32 exact-reranking
contract used by ColBERT embeddings. Results from the earlier dirty worktree
remain under `results/controlled/` and are not inputs to final figures.
