#!/usr/bin/env python3
"""Aggregate full E1 fingerprint records without hidden benchmark-specific joins."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path):
    return pd.DataFrame(json.loads(line) for line in Path(path).read_text().splitlines() if line)


def pair_delta(table, left, right, name):
    index = ['model', 'benchmark', 'item_id']
    columns = index + ['gold_vs_rest_logodds', 'correct']
    a = table[table.condition == left][columns].rename(columns={
        'gold_vs_rest_logodds': f'{name}_left_logodds', 'correct': f'{name}_left_correct'})
    b = table[table.condition == right][columns].rename(columns={
        'gold_vs_rest_logodds': f'{name}_right_logodds', 'correct': f'{name}_right_correct'})
    joined = a.merge(b, on=index, validate='one_to_one')
    # Never summarize a contrast from a partially completed condition. E1 has
    # one retained uniform32 row in each InternVL3.5-38B cell before the known
    # 80GB OOM; an inner join would otherwise make that one item look complete.
    expected = table.groupby(['model', 'benchmark']).item_id.nunique().rename('expected')
    left_n = a.groupby(['model', 'benchmark']).item_id.nunique().rename('left_n')
    right_n = b.groupby(['model', 'benchmark']).item_id.nunique().rename('right_n')
    complete = pd.concat([expected, left_n, right_n], axis=1).fillna(0)
    complete = complete[(complete.left_n == complete.expected) &
                        (complete.right_n == complete.expected)].index
    joined = joined.set_index(['model', 'benchmark']).loc[
        lambda x: x.index.isin(complete)].reset_index()
    joined[f'{name}_logodds_delta'] = joined[f'{name}_left_logodds'] - joined[f'{name}_right_logodds']
    joined[f'{name}_accuracy_delta'] = joined[f'{name}_left_correct'].astype(float) - joined[f'{name}_right_correct'].astype(float)
    return joined[index + [f'{name}_logodds_delta', f'{name}_accuracy_delta']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path,
                        help='One JSONL file or a directory containing per-cell JSONL files.')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.input.glob('*.jsonl')) if args.input.is_dir() else [args.input]
    if not paths:
        raise ValueError(f'no JSONL result files found at {args.input}')
    table = pd.concat([read_jsonl(path) for path in paths], ignore_index=True)
    required = {'model', 'benchmark', 'item_id', 'condition', 'correct', 'gold_vs_rest_logodds', 'gold_option_logprob'}
    missing = required - set(table)
    if missing:
        raise ValueError(f'input lacks {sorted(missing)}')
    table['gold_probability'] = np.exp(table.gold_option_logprob.astype(float))
    summary = table.groupby(['model', 'benchmark', 'condition'], dropna=False).agg(
        items=('item_id', 'size'), accuracy=('correct', 'mean'), mean_logodds=('gold_vs_rest_logodds', 'mean'),
        median_logodds=('gold_vs_rest_logodds', 'median'), mean_gold_probability=('gold_probability', 'mean'),
        mean_seconds=('seconds', 'mean')).reset_index()
    # Positive values always denote the named first condition outperforming its control.
    comparisons = [
        ('oracle8', 'uniform8', 'oracle_headroom'),
        ('dose8', 'dose1', 'evidence_dose_8_vs_1'),
        ('oracle8', 'oracle4_distractor4', 'context_interference'),
        ('oracle8', 'oracle8_reversed', 'reverse_order_sensitivity'),
        ('oracle8', 'oracle8_shuffled', 'shuffle_order_sensitivity'),
        ('uniform8', 'question_only', 'uniform8_vs_prior'),
        ('middle1', 'question_only', 'middle1_vs_prior'),
        ('uniform32', 'uniform4', 'coverage_32_vs_4'),
    ]
    deltas = []
    for left, right, name in comparisons:
        if left in set(table.condition) and right in set(table.condition):
            deltas.append(pair_delta(table, left, right, name))
    per_item = None
    if deltas:
        per_item = deltas[0]
        for delta in deltas[1:]:
            per_item = per_item.merge(delta, on=['model', 'benchmark', 'item_id'], how='outer', validate='one_to_one')
        delta_columns = [column for column in per_item if column.endswith('_delta')]
        derived = per_item.groupby(['model', 'benchmark'])[delta_columns].mean().reset_index()
    else:
        per_item = pd.DataFrame(columns=['model', 'benchmark', 'item_id'])
        derived = pd.DataFrame(columns=['model', 'benchmark'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'per_condition_records.parquet', index=False)
    summary.to_csv(args.output_dir / 'condition_summary.csv', index=False)
    per_item.to_parquet(args.output_dir / 'per_item_deltas.parquet', index=False)
    derived.to_csv(args.output_dir / 'fingerprint_summary.csv', index=False)
    print(summary.to_string(index=False))
    print(derived.to_string(index=False))


if __name__ == '__main__':
    main()
