#!/usr/bin/env python3
"""Aggregate E8 ordering/slot/no-op factorial controls."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    results = pd.DataFrame(json.loads(line) for line in args.results.read_text().splitlines() if line)
    inputs = pd.read_parquet(args.inputs)
    keys = ['item_id', 'context_index', 'condition']
    # The evaluator preserves E4's pair/factorial metadata; only the newly
    # introduced robustness label must be joined from the prepared table.
    meta = keys + ['robustness_condition']
    table = results.merge(inputs[meta], on=keys, validate='one_to_one')
    index = ['model', 'benchmark', 'item_id', 'context_index', 'robustness_condition', 'pair_type']
    wide = table.pivot(index=index, columns='factorial_cell', values='gold_vs_rest_logodds').reset_index()
    if {'baseline', 'a', 'b', 'ab'} - set(wide):
        raise ValueError('E8 factorial cells incomplete')
    wide['interaction'] = wide.ab - wide.a - wide.b + wide.baseline
    wide['absolute_interaction'] = wide.interaction.abs()
    summary = wide.groupby(['model', 'benchmark', 'robustness_condition', 'pair_type']).agg(
        cells=('interaction', 'size'), mean_interaction=('interaction', 'mean'),
        mean_absolute_interaction=('absolute_interaction', 'mean'),
        max_absolute_interaction=('absolute_interaction', 'max'),
    ).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(args.output_dir / 'per_pair_context.parquet', index=False)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
