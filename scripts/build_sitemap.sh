#!/usr/bin/env bash
set -euo pipefail

site_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_url="${SITE_URL:-https://tie-pilot-qxw.github.io}"
output="$site_root/sitemap.xml"
temporary="$(mktemp "$site_root/.sitemap.xml.XXXXXX")"
trap 'rm -f "$temporary"' EXIT

{
  printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>'
  printf '%s\n' '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'

  while IFS= read -r -d '' page; do
    relative="${page#"$site_root"/}"
    if [[ "$relative" == "index.html" ]]; then
      location="$site_url/"
    else
      location="$site_url/${relative%index.html}"
    fi

    printf '  <url>\n'
    printf '    <loc>%s</loc>\n' "$location"
    printf '  </url>\n'
  done < <(find "$site_root" \
    -path "$site_root/.git" -prune -o \
    -path "$site_root/_site" -prune -o \
    -name index.html -type f -print0 | sort -z)

  printf '%s\n' '</urlset>'
} > "$temporary"

mv "$temporary" "$output"
trap - EXIT
echo "Built $output"
