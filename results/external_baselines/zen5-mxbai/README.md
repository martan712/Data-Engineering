# External Baseline Results — PDX-sigmod agnews-mxbai-1024 ZEN5 (from local commit 996c714)

These CSV files were committed in the LOCAL-ONLY commit `996c714` ("benchmark
ryzenAI martan mxbai") in the local `PDX-sigmod` checkout.  That commit was
NOT pushed to origin (https://github.com/cwida/PDX, branch `sigmod`).  The
nearest remote ancestor is `fdc62f2` (remote sigmod tip), which is what the
`extern/PDX-sigmod` submodule is pinned to.

Dataset: `agnews-mxbai-1024-euclidean` on AMD ZEN5 (Ryzen AI 7 445) hardware.

## Files

| File | Method |
|------|--------|
| `IVF_BRUTEFORCE.csv`           | Brute-force IVF baseline |
| `IVF_FAISS.csv`                | FAISS IVF |
| `IVF_NARY_ADSAMPLING_SIMD.csv` | N-ary ADSampling (SIMD) |
| `IVF_PDX_ADSAMPLING.csv`       | PDX ADSampling |
| `IVF_PDX_BOND.csv`             | PDX BOND |
| `IVF_PDX_BSA.csv`              | PDX BSA |

## Provenance

- Source repo: https://github.com/cwida/PDX (branch `sigmod`)
- Local commit: `996c71450ec32b7549847708dbe9b797fc0a1e04` (local-only, NOT in remote)
- Remote ancestor: `fdc62f2` (remote sigmod tip, used as `extern/PDX-sigmod` pin)
- Original path in checkout: `benchmarks/results/ZEN5/`
- Preserved on: 2026-06-30 when the local checkout was converted to the
  `extern/PDX-sigmod` git submodule (which is pinned to the remote ancestor).
- Full source diff (fdc62f2..996c714) is captured in
  `extern/patches/PDX-sigmod-996c714.diff`.

## DEVIATION NOTE

The original task spec requested pinning `extern/PDX-sigmod` to `996c714`.
That commit is local-only and not in the upstream remote, so the submodule
is pinned to `fdc62f2` (the remote tip) instead.  The full diff from
`fdc62f2` to `996c714` is preserved in `extern/patches/PDX-sigmod-996c714.diff`.
