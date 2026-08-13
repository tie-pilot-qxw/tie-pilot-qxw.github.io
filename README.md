# Xinwei Qiang — Academic website

A simple, responsive, framework-free academic website generated from the information in `main.tex`.

The standalone blog index lives at `blog/index.html`. Add future post entries there and place each
post in its own subdirectory under `blog/`. Each post directory should contain an `article.md`;
the build script discovers all such posts, generates their `index.html` files and refreshes the
site-wide `sitemap.xml` automatically:

```bash
./scripts/build_blog.sh
```

To refresh only the sitemap after adding another standalone page, run:

```bash
./scripts/build_sitemap.sh
```

## Preview

```bash
python3 -m http.server 8000
```

Then visit <http://localhost:8000>.

The CV links point to `main.pdf`. Recompile the PDF after editing the TeX source:

```bash
latexmk -pdf main.tex
```
