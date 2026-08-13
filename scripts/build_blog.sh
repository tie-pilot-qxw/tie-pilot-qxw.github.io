#!/usr/bin/env bash
set -euo pipefail

site_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_url="https://tie-pilot-qxw.github.io"

while IFS= read -r -d '' article; do
  post_dir="$(dirname "$article")"
  slug="$(basename "$post_dir")"

  pandoc "$article" \
    --from=markdown+smart \
    --to=html5 \
    --standalone \
    --template="$site_root/blog/post-template.html" \
    --metadata="canonical:$site_url/blog/$slug/" \
    --toc \
    --toc-depth=2 \
    --wrap=none \
    --output="$post_dir/index.html"

  echo "Built $post_dir/index.html"
done < <(find "$site_root/blog" -mindepth 2 -maxdepth 2 -name article.md -print0 | sort -z)

bash "$site_root/scripts/build_sitemap.sh"
