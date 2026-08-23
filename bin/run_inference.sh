#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MODEL INPUT.parquet OUTPUT.jsonl" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

cd "$ROOT"
"$PYTHON_BIN" src/evaluate_mcq.py \
  --registry configs/registry.json \
  --reproducibility-config configs/reproducibility.json \
  --model "$1" \
  --input "$2" \
  --output "$3"

