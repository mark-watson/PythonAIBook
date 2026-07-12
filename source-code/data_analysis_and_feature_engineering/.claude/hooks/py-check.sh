#!/usr/bin/env bash
set -euo pipefail
input="$(cat)"
file="$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')"
case "$file" in *.py|*.pyi) ;; *) exit 0 ;; esac
[ -f "$file" ] || exit 0

uv run ruff format --quiet "$file" || true
uv run ruff check --fix --quiet "$file" || true

if ! out="$(uv run pyrefly check "$file" 2>&1)"; then
  { echo "pyrefly type errors in $file — fix these:"; echo "$out"; } >&2
  exit 2
fi
