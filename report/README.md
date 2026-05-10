# Report Build Notes

Compile from the repository root after generating report artifacts:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_report_configs.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_report_artifacts.py
pdflatex report/main.tex
bibtex main
pdflatex report/main.tex
pdflatex report/main.tex
```

This environment may not have a TeX distribution installed. The source files and BibTeX references are still complete.
