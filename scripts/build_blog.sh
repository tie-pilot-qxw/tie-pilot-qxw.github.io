#!/usr/bin/env bash
set -euo pipefail

site_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
post_dir="$site_root/blog/reconstructing-a-megakernel-with-pdl"

pandoc "$post_dir/article.md" \
  --from=markdown+smart \
  --to=html5 \
  --standalone \
  --template="$site_root/blog/post-template.html" \
  --toc \
  --toc-depth=2 \
  --wrap=none \
  --output="$post_dir/index.html"

echo "Built $post_dir/index.html"
