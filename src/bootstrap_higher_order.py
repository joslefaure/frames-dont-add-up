#!/usr/bin/env python3
"""Cluster-bootstrap E5's fixed-budget Möbius order fractions by question.

Contexts and subset masks belonging to the same question are resampled
together.  This preserves the repeated-measures design and gives uncertainty
intervals for the descriptive absolute-component fractions reported by E5.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def interval(values, alpha):
    return np.quantile(values, [alpha / 2, 1 - alpha / 2]).tolist()


def bootstrap_cell(table, draws, seed, alpha):
    grouped = []
    for _, item in table.groupby('item_id', sort=False):
        # One mass per order after summing all subset coefficients within a
        # context, then averaging contexts.  Summing masks is essential because
        # the numbers of coefficients at orders 1--4 are 4, 6, 4, and 1.
        per_context = item.groupby(['context_index', 'order']).absolute_component.sum().unstack('order')
        grouped.append(per_context.reindex(columns=range(1, 5)).mean(axis=0).to_numpy())
    values = np.stack(grouped)
    point = values.mean(axis=0)
    point_fraction = point / point.sum()
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 4), dtype=float)
    for draw in range(draws):
        mean = values[rng.integers(0, len(values), len(values))].mean(axis=0)
        samples[draw] = mean / mean.sum()
    rows = []
    for order in range(1, 5):
        low, high = interval(samples[:, order - 1], alpha)
        rows.append({
            'order': order,
            'items': len(values),
            'fraction_absolute_component': point_fraction[order - 1],
            'ci_low': low,
            'ci_high': high,
            'bootstrap_draws': draws,
            'cluster': 'item_id',
        })
    # The main mechanistic contrast is high-order (3+4) versus low-order.
    high = samples[:, 2] + samples[:, 3]
    low, high_ci = interval(high, alpha)
    rows.append({
        'order': '3_plus_4',
        'items': len(values),
        'fraction_absolute_component': point_fraction[2] + point_fraction[3],
        'ci_low': low,
        'ci_high': high_ci,
        'bootstrap_draws': draws,
        'cluster': 'item_id',
    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--alpha', type=float, default=.05)
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError('--draws must be at least 100')
    rows = []
    for path in sorted(args.input_root.glob('*/mobius_components.parquet')):
        table = pd.read_parquet(path)
        required = {'model', 'benchmark', 'item_id', 'order', 'absolute_component'}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f'{path} lacks {sorted(missing)}')
        pairs = table[['model', 'benchmark']].drop_duplicates().to_dict('records')
        if len(pairs) != 1:
            raise ValueError(f'{path} must contain exactly one model/benchmark cell')
        cell = pairs[0]
        seed = args.seed + int.from_bytes(f"{cell['model']}:{cell['benchmark']}".encode(), 'little') % 1000003
        for row in bootstrap_cell(table[table.order > 0], args.draws, seed, args.alpha):
            rows.append({**cell, **row})
    result = pd.DataFrame(rows).sort_values(['model', 'benchmark', 'order'], key=lambda x: x.astype(str))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
