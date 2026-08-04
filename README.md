# Xinwei Qiang — Academic website

A simple, responsive, framework-free academic website generated from the information in `main.tex`.

The standalone blog index lives at `blog/index.html`. Add future post entries there and place each
post in its own subdirectory under `blog/`.

## Preview

```bash
python3 -m http.server 8000
```

Then visit <http://localhost:8000>.

The CV links point to `main.pdf`. Recompile the PDF after editing the TeX source:

```bash
latexmk -pdf main.tex
```
