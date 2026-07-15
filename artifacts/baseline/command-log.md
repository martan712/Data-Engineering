# Baseline command log

Captured on 2026-07-15 at revision
`b38abb6caffd2a10ebe0131bc5adcb051fea0cb4`.

| Check | Command | Outcome |
|---|---|---|
| Python tests | `.venv/bin/python -m pytest -q` | PASS: 167 tests in 13.71 s |
| Per-document native build | `make -C cpp/per_document_oracle` | PASS: existing target was up to date |
| Wide-block native build | `make -C cpp/wide_block_maxsim_bond` | PASS: existing target was up to date |
| Fused-panel native build | `make -C cpp/fused_panel_maxsim` | PASS: existing target was up to date |
| Stage 5 table renderer | `.venv/bin/python -m experiments.stage5_corect.render_ir_table` | PASS: emitted all four dataset blocks |
| Historical schema reader | `ResultRecord.from_json` over matching result files | PASS: 20 flat records loaded, 0 failures |
| Paper build | `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from `report/` | PASS: `main.pdf` was up to date |
| Package inventory | `uv pip freeze` | FAILED before execution because uv tried to lock its read-only user cache; package versions were captured without uv through `importlib.metadata` |

The test command was run with all three existing `.so` files present. The
baseline pytest configuration had no explicit native-required tier, so the
successful count does not prove that a missing native library would fail CI.

The current native Makefiles all used `-O3 -march=native -DNDEBUG -std=c++20`.
The fused-panel build additionally used OpenMP and compiled position-independent
objects before linking its shared library.

The table renderer's baseline output was generated from the tracked Stage 5
e01/e03 JSON files. The paper still contained handwritten numeric table bodies;
no `\input{...}` publication fragment was detected in any table or figure
block.
