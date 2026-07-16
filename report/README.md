# LaTeX Project Report

Academic report based on the Markdown documentation and the audited release
artifacts under `results/final/`. Figures are generated from those JSON files,
and `tests/test_report_consistency.py` checks the main table values against the
same artifacts (clean commit `0a11fda`).

## Layout

```text
report/
|-- main.tex            # document shell, preamble, title page, abstract
|-- references.bib      # bibliography (from docs/project_report.md references)
|-- sections/
|   |-- introduction.tex
|   |-- background.tex
|   |-- architecture.tex
|   |-- methodology.tex
|   |-- results_ivf.tex
|   |-- results_plaid.tex
|   |-- results_bond.tex
|   |-- results_pca.tex
|   |-- discussion.tex
|   |-- limitations.tex
|   |-- reproducibility.tex
|   |-- conclusion.tex
|   `-- appendix.tex
`-- main.pdf            # compiled output (if a compiler was available)
```

Figures are **not** duplicated: `main.tex` sets
`\graphicspath{{../docs/figures/}}` and references
`fig5_controlled_quality_latency.png`, `fig6_controlled_stage_breakdown.png`,
`fig7_controlled_bond.png`, `fig8_controlled_transfer.png`, and
`fig9_controlled_plaid.png` in place.
The report must therefore be compiled from inside the `report/` directory of
this repository (or with the `docs/figures/` tree present one level up).

## Requirements

A TeX distribution (TeX Live or MiKTeX) with the packages `booktabs`,
`siunitx`, `amsmath`, `graphicx`, `microtype`, `natbib`, `hyperref`,
`cleveref`, `caption`, and `geometry`. `latexmk` additionally requires Perl
(on Windows/MiKTeX, install e.g. Strawberry Perl).

## Compilation

Preferred (from the `report/` directory):

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Without `latexmk` (equivalent manual sequence, also from `report/`):

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex   main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

On MiKTeX, allow on-the-fly package installation the first time
(`--enable-installer` or via the MiKTeX console) so that missing packages are
fetched automatically.

The manual sequence above was last verified on 2026-07-16 and produced the
tracked 24-page `main.pdf` without undefined references, citation warnings, or
layout warnings. `latexmk` was not available in that Windows environment
because MiKTeX could not find Perl.

Clean auxiliary files with `latexmk -c` (keeps the PDF) or `latexmk -C`.

## Metadata TODOs

The title page contains explicit `[TODO: ...]` placeholders for the author,
university, programme, official course code, supervisor, and submission date.
These were intentionally not invented; fill them in `main.tex` (title page
block) before submission.
