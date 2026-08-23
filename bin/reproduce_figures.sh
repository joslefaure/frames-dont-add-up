#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUT="$ROOT/reproduced/figures"
mkdir -p "$OUT"
cd "$ROOT"

"$PYTHON_BIN" src/plot_primary_effects.py \
  --input artifacts/causal_analysis/bootstrap_primary_contrasts.csv \
  --output-prefix "$OUT/primary_effects"
"$PYTHON_BIN" src/plot_e3_proxy_alignment.py \
  --input artifacts/e3_alignment/all_summary.csv \
  --output-prefix "$OUT/e3_proxy_alignment"
"$PYTHON_BIN" src/plot_e5_higher_order.py \
  --input artifacts/e5_expanded_analysis/bootstrap_order_fractions.csv \
  --output-prefix "$OUT/e5_higher_order"
"$PYTHON_BIN" src/plot_e6_controlled.py \
  --input artifacts/e6_controlled_analysis/controlled_logistic_coefficients.csv \
  --output-prefix "$OUT/e6_controlled"
"$PYTHON_BIN" src/plot_e8_controls.py \
  --input artifacts/e8_extension_analysis/topology_contrasts.csv \
  --output-prefix "$OUT/e8_controls"
"$PYTHON_BIN" src/plot_e9_scale.py \
  --input artifacts/e9_scale_analysis/scale_pairs.csv \
  --output-prefix "$OUT/e9_scale"

echo "Regenerated figures in $OUT"
