# Overleaf Report Build Notes

This folder is self-contained for Overleaf upload. Upload the entire
`report_overleaf` folder or the generated `report_overleaf.zip` archive.

Use `main.tex` as the Overleaf main document. The source uses the standard
IEEE two-column conference class available on Overleaf.

The report expects:

- `figures/` for all included PNG figures.
- `figures/extra/` for optional training curves and entropy scatter plots.
- `tables/` for LaTeX table snippets used by `\input`.
- `data_tables/` for CSV/JSON source metrics kept for editing/reference.
- `references.bib` for BibTeX citations.

If compiling locally from this folder:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The original repo-generated metrics and artifact CSVs are preserved in
`data_tables/`; the rendered LaTeX tables are in `tables/`.
