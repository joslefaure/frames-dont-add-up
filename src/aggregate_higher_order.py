#!/usr/bin/env python3
"""Estimate E5 Möbius interaction components from all fixed-budget subsets."""
import argparse
import json
from pathlib import Path

import pandas as pd


def mobius(values, mask):
    total = 0.0
    subset = mask
    while True:
        total += (-1) ** ((mask.bit_count() - subset.bit_count())) * values[subset]
        if subset == 0:
            break
        subset = (subset - 1) & mask
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    table = pd.DataFrame(rows)
    required = {'model', 'benchmark', 'item_id', 'context_index', 'subset_mask', 'gold_vs_rest_logodds'}
    missing = required - set(table)
    if missing:
        raise ValueError(f'input lacks {sorted(missing)}')
    out = []
    for keys, group in table.groupby(['model', 'benchmark', 'item_id', 'context_index'], sort=False):
        values = dict(zip(group.subset_mask.astype(int), group.gold_vs_rest_logodds.astype(float)))
        if set(values) != set(range(16)):
            raise ValueError(f'{keys}: expected masks 0..15, found {sorted(values)}')
        for mask in range(16):
            out.append({
                'model': keys[0], 'benchmark': keys[1], 'item_id': keys[2], 'context_index': keys[3],
                'subset_mask': mask, 'order': mask.bit_count(), 'mobius_component': mobius(values, mask),
            })
    components = pd.DataFrame(out)
    components['absolute_component'] = components.mobius_component.abs()
    summary = components.groupby(['model', 'benchmark', 'order']).agg(
        components=('mobius_component', 'size'), mean_component=('mobius_component', 'mean'),
        mean_absolute_component=('absolute_component', 'mean'),
        absolute_component_mass=('absolute_component', 'sum')).reset_index()
    nonzero = summary[summary.order > 0].copy()
    # Component counts differ by order (4, 6, 4, 1 for four candidates), so
    # total mass must sum every coefficient magnitude.  Normalizing the four
    # order-wise means would incorrectly give the lone fourth-order coefficient
    # the same aggregate weight as all six pair coefficients.
    totals = nonzero.groupby(['model', 'benchmark']).absolute_component_mass.sum().rename(
        'total_absolute_component_mass').reset_index()
    fractions = nonzero.merge(totals, on=['model', 'benchmark'])
    fractions['fraction_absolute_component'] = (
        fractions.absolute_component_mass / fractions.total_absolute_component_mass)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    components.to_parquet(args.output_dir / 'mobius_components.parquet', index=False)
    summary.to_csv(args.output_dir / 'order_summary.csv', index=False)
    fractions.to_csv(args.output_dir / 'order_fractions.csv', index=False)
    print(fractions.to_string(index=False))


if __name__ == '__main__':
    main()
