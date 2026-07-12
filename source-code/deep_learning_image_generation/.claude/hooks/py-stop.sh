#!/usr/bin/env bash
set -euo pipefail
input="$(cat)"
active="$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("stop_hook_active", False))')"
[ "$active" = "True" ] && exit 0

if ! out="$(.venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/pyrefly check && .venv/bin/pytest -q 2>&1)"; then
  { echo "Project checks failed — resolve before finishing:"; echo "$out"; } >&2
  exit 2
fi
