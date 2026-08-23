#!/usr/bin/env python3
"""Require exact E2 result coverage before aggregation or publication."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    args = parser.parse_args()
    expected = pd.read_parquet(args.inputs)
    actual = pd.DataFrame(json.loads(line) for line in args.results.read_text().splitlines() if line)
    fields = {'item_id', 'condition', 'model', 'benchmark', 'correct', 'gold_vs_rest_logodds'}
    if missing := fields - set(actual):
        raise ValueError(f'results lack {sorted(missing)}')
    keys = ['item_id', 'condition']
    if expected.duplicated(keys).any():
        raise ValueError('prepared E2 table has duplicate item/condition keys')
    if actual.duplicated(keys).any():
        raise ValueError('result JSONL has duplicate item/condition keys')
    if actual.model.nunique() != 1 or actual.benchmark.nunique() != 1:
        raise ValueError('result JSONL must contain exactly one model and benchmark')
    input_benchmarks = set(expected.benchmark.unique())
    if input_benchmarks != set(actual.benchmark.unique()):
        raise ValueError(f'benchmark mismatch: input={input_benchmarks}, results={set(actual.benchmark.unique())}')
    expected_keys = set(map(tuple, expected[keys].itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, actual[keys].itertuples(index=False, name=None)))
    if missing := expected_keys - actual_keys:
        raise ValueError(f'missing {len(missing)} prepared intervention results; first={next(iter(missing))}')
    if unexpected := actual_keys - expected_keys:
        raise ValueError(f'unexpected {len(unexpected)} result interventions; first={next(iter(unexpected))}')
    item_counts = actual.groupby('condition').item_id.nunique()
    if not (item_counts == expected.item_id.nunique()).all():
        raise ValueError('selector conditions do not retain the exact common item universe')
    print(json.dumps({'status': 'ok', 'model': actual.model.iloc[0], 'benchmark': actual.benchmark.iloc[0],
                      'items': int(expected.item_id.nunique()), 'conditions': int(expected.condition.nunique()),
                      'records': len(actual)}))


if __name__ == '__main__':
    main()
