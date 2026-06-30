# Local Benchmark-Configuration Patches

These patches capture the local modifications made to the PDX and PDX-sigmod
checkouts in order to produce the benchmark results preserved in
`results/external_baselines/`.  They are stored here so that the exact
benchmark configuration is reproducible even though the modified files are not
present in the upstream repositories.

## DEVIATION: PDX-sigmod local commit 996c714

The original task spec requested pinning `extern/PDX-sigmod` to commit
`996c714` ("benchmark ryzenAI martan mxbai").  However, that commit is
**local-only** — it was never pushed to https://github.com/cwida/PDX.  The
nearest remote ancestor is `fdc62f2` (remote sigmod tip).

To preserve the full content of the local commit:
- `PDX-sigmod-996c714.diff` — complete text-file diff from `fdc62f2` to
  `996c714`, including source changes and benchmark results CSVs.
- `PDX-sigmod.local.patch` — uncommitted working-tree changes on top of
  `996c714` (setup_settings.py switch from agnews-mxbai to openai-1536-angular).
- The committed ZEN5 results CSVs from `996c714` are preserved in
  `results/external_baselines/zen5-mxbai/`.

## Files

| Patch | Base commit | Description |
|-------|------------|-------------|
| `PDX.local.patch`             | `extern/PDX` @ 93531b9      | Uncommitted working-tree changes: `benchmarks/kernels_playground/kernels.py`, `benchmarks/python_scripts/setup_data.py` |
| `PDX-sigmod.local.patch`      | local `PDX-sigmod` @ 996c714 | Uncommitted working-tree changes on top of local commit: `setup_settings.py` (openai-1536-angular switch) |
| `PDX-sigmod-996c714.diff`     | remote sigmod `fdc62f2`      | Full diff fdc62f2..996c714 (the local-only commit); includes setup_data.py, setup_settings.py, benchmark_utils.hpp, setup.py, ZEN5 results CSVs |

## How to re-apply

```bash
# Reproduce the agnews-mxbai benchmark config (local commit 996c714 equivalent):
git -C extern/PDX-sigmod apply extern/patches/PDX-sigmod-996c714.diff

# Reproduce any further working-tree tweaks on top:
git -C extern/PDX-sigmod apply extern/patches/PDX-sigmod.local.patch

# Reproduce the PDX kernel / data modifications:
git -C extern/PDX apply extern/patches/PDX.local.patch
```

After applying, re-run the benchmark scripts in the submodule's
`benchmarks/python_scripts/` directory to reproduce the preserved results.

## Why these modifications exist

The upstream PDX/PDX-sigmod repositories contain generic benchmark scripts.
The local patches configure dataset paths, kernel selections, and output
settings for the specific hardware (AMD ZEN5 / Ryzen AI 7 445, AVX-512,
Fedora 43) used to produce the baseline results cited in Stage 1 §5.1.
