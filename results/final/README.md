# Final Controlled Results

These seven artifacts were generated from clean commit `0a11fda` in one Ubuntu
WSL2 environment using Python 3.11.9. Every run records `dirty=false`, initial
CPU affinity `[0, 2, 4, 6]`, four OpenMP/BLAS/Rayon threads, one warm-up, five
measured interleaved runs, package versions, input hashes, stage timings, and
raw outer-timer samples.

The online boundary starts with resident normalized query token vectors and ends
with ranked document IDs. Encoding and index construction are outside this
boundary and reported separately. Ratios therefore describe online processing
for the named arms, not end-to-end retrieval-system speedup.

The two IVF artifacts contain compiled exact, FAISS-IVF, PDX-IVF, and the PLAID
configurations frozen from `results/validation/`. The two PLAID full-score files
are explicitly post-hoc sensitivity runs, not selected release configurations.
PyLate/fast-plaid exposes a configured `n_full_scores` value but not the realized
candidate count, so it is not treated as measured work equivalent to IVF's
realized rerank count.

The BOND comparison uses float64 products and accumulation in both exhaustive
and exact-safe kernels. IVF exact reranking retains the float32 contract used by
the ColBERT embeddings. The JSON `status` value remains
`pilot_not_final_claim` for runner-schema compatibility; `manifest.json` is the
release disposition and accepts only clean artifacts from the source commit.
