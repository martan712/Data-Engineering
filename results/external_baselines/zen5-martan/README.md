# External Baseline Results — PDX-sigmod BOND/ADSampling/BSA/FAISS (ZEN5-Martan)

These CSV files are preserved from the local `PDX-sigmod` checkout (cwida/PDX,
commit `996c714`, branch `sigmod`), where they lived at
`benchmarks/results/ZEN5-Martan/`. They are **not present in the upstream
repository** and represent locally-run single-vector L2 benchmark results
comparing BOND, ADSampling, BSA, and FAISS IVF methods on AMD ZEN5 hardware.

**Referenced by:** Stage 1 §5.1 of
`docs/stage1_bond_maxsim_formalization.md`.

## Files

| File | Method |
|------|--------|
| `IVF_BRUTEFORCE.csv`          | Brute-force IVF baseline |
| `IVF_FAISS.csv`               | FAISS IVF |
| `IVF_NARY_ADSAMPLING_SIMD.csv`| N-ary ADSampling (SIMD) |
| `IVF_PDX_ADSAMPLING.csv`      | PDX ADSampling |
| `IVF_PDX_BOND.csv`            | PDX BOND |
| `IVF_PDX_BSA.csv`             | PDX BSA |

## Provenance

- Source repo: https://github.com/cwida/PDX (branch `sigmod`)
- Pinned commit: `996c71450ec32b7549847708dbe9b797fc0a1e04`
- Original path: `benchmarks/results/ZEN5-Martan/`
- Preserved on: 2026-06-30 when the local checkout was converted to the
  `extern/PDX-sigmod` git submodule at the same pinned commit.
- Local benchmark-configuration modifications used to produce these results
  are captured in `extern/patches/PDX-sigmod.local.patch`.
