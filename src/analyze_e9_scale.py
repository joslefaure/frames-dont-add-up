#!/usr/bin/env python3
"""Summarize pre-specified Qwen and InternVL scale-pair effects from completed audits."""
import argparse
from pathlib import Path

import pandas as pd


PAIRS = {'qwen': ('qwen3_vl_4b', 'qwen3_vl_32b'), 'internvl': ('internvl35_8b', 'internvl35_38b')}


def read_e3(root):
    values = []
    for path in root.glob('*/summary.csv'):
        table = pd.read_csv(path)
        pivot = table.pivot(index=['model', 'benchmark'], columns='candidate_stratum', values=['mean_utility', 'mean_context_std'])
        for (model, benchmark), row in pivot.iterrows():
            values.append({'model': model, 'benchmark': benchmark,
                           'evidence_minus_outside_utility': row[('mean_utility', 'evidence')] - row[('mean_utility', 'outside')],
                           'evidence_context_std': row[('mean_context_std', 'evidence')]})
    return pd.DataFrame(values)


def read_e4(root):
    values = []
    for path in root.glob('*/summary.csv'):
        table = pd.read_csv(path)
        pivot = table.pivot(index=['model', 'benchmark'], columns='pair_type', values='mean_absolute_interaction')
        for (model, benchmark), row in pivot.iterrows():
            values.append({'model': model, 'benchmark': benchmark,
                           'evidence_pair_absolute_interaction': row['evidence_evidence'],
                           'evidence_minus_outside_absolute_interaction': row['evidence_evidence'] - row['outside_outside']})
    return pd.DataFrame(values)


def read_e2(path):
    table = pd.read_csv(path)
    return table.groupby(['model', 'benchmark']).agg(
        best_selector_logodds=('mean_logodds', 'max'), selector_accuracy=('accuracy', 'max')).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e3-root', required=True, type=Path)
    parser.add_argument('--e4-root', required=True, type=Path)
    parser.add_argument('--e2-summary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    table = read_e3(args.e3_root).merge(read_e4(args.e4_root), on=['model', 'benchmark'], how='outer')
    table = table.merge(read_e2(args.e2_summary), on=['model', 'benchmark'], how='outer')
    rows = []
    for family, (small, large) in PAIRS.items():
        left = table[table.model == small].set_index('benchmark')
        right = table[table.model == large].set_index('benchmark')
        for benchmark in sorted(set(left.index) & set(right.index)):
            row = {'family': family, 'small_model': small, 'large_model': large, 'benchmark': benchmark}
            for metric in table.columns.drop(['model', 'benchmark']):
                row[f'{metric}_small'] = left.loc[benchmark, metric]
                row[f'{metric}_large'] = right.loc[benchmark, metric]
                row[f'{metric}_large_minus_small'] = right.loc[benchmark, metric] - left.loc[benchmark, metric]
            rows.append(row)
    output = pd.DataFrame(rows).sort_values(['family', 'benchmark'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == '__main__':
    main()
