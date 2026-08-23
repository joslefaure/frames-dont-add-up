#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found. Run: bash bin/setup.sh" >&2
  exit 2
fi

cd "$ROOT"
"$PYTHON_BIN" src/verify_release.py
"$PYTHON_BIN" src/verify_claims.py

