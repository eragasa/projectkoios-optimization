# User manual

`user_manual.pdf` is the checked user-facing manual. Its LaTeX source is
`user_manual.tex` with chapters under `chapters/`.

Build reproducibly from this directory with:

```bash
SOURCE_DATE_EPOCH=1 TZ=UTC pdflatex -interaction=nonstopmode -halt-on-error user_manual.tex
SOURCE_DATE_EPOCH=1 TZ=UTC pdflatex -interaction=nonstopmode -halt-on-error user_manual.tex
```

The manual documents current library contracts. Consumer scientific policy remains in
consumer repositories and is described only as an external need.
