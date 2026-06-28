# Sokoban Report

This directory contains the course report source and compiled PDF.

## Files to edit

- `main.tex`: report source.
- `references.bib`: bibliography entries.
- `figures/`: plots used by the report.
- `main.pdf`: compiled report for quick review.

## Build

Use XeLaTeX because the report contains Chinese text and uses local CJK fonts.

```bash
cd report
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

If references were changed and the bibliography does not refresh, clean and rebuild:

```bash
cd report
latexmk -C main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The ICML 2022 style files required for local compilation are included in this directory.
