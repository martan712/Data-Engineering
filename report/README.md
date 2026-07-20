# LaTeX report

The project report is assembled from `main.tex`, the files under `sections/`,
and `references.bib`. Its tables are checked against the JSON files in
`results/final/` by `tests/test_report_consistency.py`.

The figures remain in `docs/figures/`, so compile from this directory while the
repository layout is intact.

## Compile

With `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Without `latexmk`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The separate `overleaf_report.zip` package places `main.tex`, `sections/`, the
bibliography, and the required figures in one self-contained archive for
Overleaf.
