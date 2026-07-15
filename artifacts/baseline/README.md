# Pre-finalization baseline

This directory freezes the repository state before implementation of
`Notes/final-implementation-plan.md`. It is a development inventory only; it
does not define the final result envelope, artifact catalog, or evidence
manifest.

The baseline was captured from revision
`b38abb6caffd2a10ebe0131bc5adcb051fea0cb4` with a clean tracked worktree on
2026-07-15. Historical result and figure files were read but not modified.

Contents:

- `environment.json`: revision, submodules, Python packages, host/toolchain,
  CPU summary, and the original native compiler flags.
- `inventory.json`: checksums and schema-key shapes for all 73 result JSON
  files, checksums for 46 figures, the 8 paper table/figure blocks, and the 19
  result-producing drivers with their apparent data/index dependencies.
- `fixtures/`: the smallest unmodified result for each of the 17 historical
  top-level JSON shapes. The source path and checksum for every fixture are in
  `inventory.json`.
- `agreement-cases.json`: representative before-state outputs, including the
  known invalid replacement that the historical gate incorrectly accepts.
- `command-log.md`: baseline build, test, reader, renderer, and paper outcomes.

Relevant ignored inputs present at capture time included the four embedding
archives and two separate Stage 5 query archives under `data/embeddings/`, BEIR
ID/qrels sidecars, FAISS and PLAID indexes, archive caches, the three built
native shared libraries, `.venv`, `uv.lock`, and generated LaTeX files. These
are intentionally described rather than copied into the baseline.

Regenerate this snapshot before the contracts are frozen with:

```bash
.venv/bin/python scripts/capture_baseline.py \
  --revision b38abb6caffd2a10ebe0131bc5adcb051fea0cb4 \
  --worktree-state clean
```
