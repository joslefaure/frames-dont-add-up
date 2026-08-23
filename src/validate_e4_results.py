#!/usr/bin/env python3
"""Require exact resumable E4 factorial coverage before re-aggregation."""
import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ['item_id', 'condition', 'context_index', 'pair_type', 'factorial_cell']


def normalize(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for field in ('context_index',):
        table[field] = table[field].astype('Int64')
    for field in ('pair_type', 'factorial_cell'):
        table[field] = table[field].fillna('')
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    expected = normalize(pd.read_parquet(args.inputs))
    actual = normalize(pd.DataFrame(
        json.loads(line) for line in args.results.read_text().splitlines() if line))
    required = set(KEYS) | {'model', 'benchmark', 'gold_vs_rest_logodds'}
    if missing := required - set(actual):
        raise ValueError(f'results lack {sorted(missing)}')
    if expected.duplicated(KEYS).any() or actual.duplicated(KEYS).any():
        raise ValueError('duplicate E4 factorial intervention keys')
    if set(actual.model) != {args.model}:
        raise ValueError(f'expected model {args.model}, found {set(actual.model)}')
    if set(expected.benchmark) != set(actual.benchmark):
        raise ValueError(f'benchmark mismatch: input={set(expected.benchmark)}, results={set(actual.benchmark)}')
    expected_keys = set(map(tuple, expected[KEYS].itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, actual[KEYS].itertuples(index=False, name=None)))
    if missing := expected_keys - actual_keys:
        raise ValueError(f'missing {len(missing)} E4 records; first={next(iter(missing))}')
    if unexpected := actual_keys - expected_keys:
        raise ValueError(f'unexpected {len(unexpected)} E4 records; first={next(iter(unexpected))}')
    print(json.dumps({'status': 'ok', 'model': args.model,
                      'benchmark': actual.benchmark.iloc[0], 'records': len(actual)}))


if __name__ == '__main__':
    main()
