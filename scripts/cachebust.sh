#!/usr/bin/env bash
# cachebust.sh — Stamp local CSS/JS asset URLs with per-file content hashes
# so returning visitors always get the latest files after a deploy.
#
# Usage:  ./cachebust.sh
#
# What it does:
#   href="/css/style.css"              → href="/css/style.css?v=a1b2c3d4"
#   href="/css/style.css?v=old"        → href="/css/style.css?v=a1b2c3d4"
#   src="/js/main.js"                  → src="/js/main.js?v=e5f6a7b8"
#   src="/js/main.js?v=old"            → src="/js/main.js?v=e5f6a7b8"
#
# The version is derived from each file's content (git hash-object), so:
#   - It only changes when the file itself changes.
#   - Running the script twice without changing assets produces no diff.
#   - External URLs (https://...) are never touched.
#   - Works on Linux, macOS, and Windows (Git Bash) — no md5sum needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Collect unique local asset paths referenced across all HTML files.
# Matches patterns like ="/css/style.css" or ="/js/main.js?v=old"
ASSETS=$(grep -roh '="/[^"?]*\.\(css\|js\)' --include='*.html' . | sed 's/^="//' | sort -u)

STAMPED=0
SKIPPED=0

for asset in $ASSETS; do
  file=".${asset}"

  if [ ! -f "$file" ]; then
    echo "  skip $asset (file not found)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  HASH=$(git hash-object "$file" | cut -c1-8)

  find . -name '*.html' -print0 | while IFS= read -r -d '' htmlfile; do
    sed -i -E "s#(=\"${asset})(\?v=[^\"]*)?\"#\1?v=${HASH}\"#g" "$htmlfile"
  done

  echo "  $asset → ?v=$HASH"
  STAMPED=$((STAMPED + 1))
done

echo ""
echo "✓ Stamped $STAMPED assets across all HTML files (content-hash based)"
[ "$SKIPPED" -gt 0 ] && echo "  ($SKIPPED skipped — files not found)"

exit 0
