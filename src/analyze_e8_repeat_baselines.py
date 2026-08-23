#!/usr/bin/env python3
"""Verify repeated E8 baselines are deterministic at the per-item level."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return pd.DataFrame(json.loads(line) for line in Path(path).read_text().splitlines() if line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat-a', required=True, type=Path)
    parser.add_argument('--repeat-b', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    a, b = read(args.repeat_a), read(args.repeat_b)
    keys = ['model', 'benchmark', 'item_id', 'context_index', 'pair_type', 'factorial_cell']
    # evaluate_mcq persists log odds, not a separate probability field. The
    # binary gold-vs-rest probability is exactly sigmoid(log odds).
    for table in (a, b):
        table['gold_probability'] = 1. / (1. + np.exp(-table.gold_vs_rest_logodds))
    values = ['gold_vs_rest_logodds', 'gold_probability', 'correct']
    table = a[keys + values].merge(b[keys + values], on=keys, validate='one_to_one', suffixes=('_a', '_b'))
    table['logodds_abs_delta'] = (table.gold_vs_rest_logodds_a - table.gold_vs_rest_logodds_b).abs()
    table['probability_abs_delta'] = (table.gold_probability_a - table.gold_probability_b).abs()
    table['correct_changed'] = table.correct_a != table.correct_b
    summary = table.groupby(['model', 'benchmark']).agg(
        rows=('item_id', 'size'), max_logodds_abs_delta=('logodds_abs_delta', 'max'),
        max_probability_abs_delta=('probability_abs_delta', 'max'),
        changed_correct=('correct_changed', 'sum')).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    if (summary.max_logodds_abs_delta > 1e-9).any() or (summary.max_probability_abs_delta > 1e-9).any() or summary.changed_correct.any():
        raise ValueError('repeat baseline is not deterministic')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
