#!/usr/bin/env python3
"""Source-video clustered bootstrap intervals for the primary E3/E4 contrasts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_difference(table, left, right, value, draws, rng):
    pivot = table.pivot(index='source_video_id', columns='group', values=value).dropna(subset=[left, right])
    values = (pivot[left] - pivot[right]).to_numpy()
    samples = np.empty(draws)
    for index in range(draws):
        samples[index] = values[rng.integers(0, len(values), len(values))].mean()
    # A paired source-video sign-flip test supplies a null p-value without
    # treating questions from the same video as independent observations.
    observed = float(values.mean())
    null = (values[None, :] * rng.choice((-1., 1.), size=(draws, len(values)))).mean(axis=1)
    p_two_sided = float((1 + (np.abs(null) >= abs(observed)).sum()) / (draws + 1))
    return len(values), observed, *np.quantile(samples, [.025, .975]).tolist(), p_two_sided


def benjamini_hochberg(values):
    """Return monotone BH-adjusted p-values in the original row order."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(1., adjusted)
    return result


def e3_rows(root, input_root, draws, seed):
    rows = []
    for path in sorted(root.glob('*/per_candidate.parquet')):
        table = pd.read_parquet(path)
        model, benchmark = table.model.iloc[0], table.benchmark.iloc[0]
        design = pd.read_parquet(input_root / benchmark / 'e3' / 'all_conditions.parquet')
        source = design[design.candidate_index >= 0].drop_duplicates(['item_id', 'candidate_index'])[
            ['item_id', 'candidate_index', 'source_video_id']]
        table = table.merge(source, on=['item_id', 'candidate_index'], validate='one_to_one')
        grouped = table.groupby(['source_video_id', 'candidate_stratum']).utility_mean.mean().reset_index(name='value')
        grouped = grouped.rename(columns={'candidate_stratum': 'group'})
        n, estimate, low, high, p = bootstrap_difference(grouped, 'evidence', 'outside', 'value', draws,
                                                           np.random.default_rng(seed + len(rows)))
        rows.append({'experiment': 'e3', 'model': model, 'benchmark': benchmark,
                     'contrast': 'evidence_minus_outside_utility', 'clusters': n,
                     'estimate': estimate, 'ci_low': low, 'ci_high': high, 'p_two_sided': p, 'draws': draws})
    return rows


def e4_rows(root, input_root, draws, seed):
    rows = []
    for path in sorted(root.glob('*/per_pair_context.parquet')):
        table = pd.read_parquet(path)
        model, benchmark = table.model.iloc[0], table.benchmark.iloc[0]
        design = pd.read_parquet(input_root / benchmark / 'e4' / 'all_conditions.parquet')
        source = design.drop_duplicates(['item_id', 'context_index', 'pair_type'])[
            ['item_id', 'context_index', 'pair_type', 'source_video_id']]
        table = table.merge(source, on=['item_id', 'context_index', 'pair_type'], validate='one_to_one')
        for value, contrast in [('absolute_interaction', 'evidence_minus_outside_absolute_interaction'),
                                ('interaction', 'evidence_minus_outside_signed_interaction')]:
            grouped = table.groupby(['source_video_id', 'pair_type'])[value].mean().reset_index(name='value')
            grouped = grouped.rename(columns={'pair_type': 'group'})
            n, estimate, low, high, p = bootstrap_difference(grouped, 'evidence_evidence', 'outside_outside',
                                                               'value', draws, np.random.default_rng(seed + len(rows)))
            rows.append({'experiment': 'e4', 'model': model, 'benchmark': benchmark,
                         'contrast': contrast, 'clusters': n,
                         'estimate': estimate, 'ci_low': low, 'ci_high': high, 'p_two_sided': p, 'draws': draws})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e3-root', required=True, type=Path)
    parser.add_argument('--e4-root', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260720)
    args = parser.parse_args()
    rows = e3_rows(args.e3_root, args.input_root, args.draws, args.seed)
    rows += e4_rows(args.e4_root, args.input_root, args.draws, args.seed + 100000)
    output = pd.DataFrame(rows).sort_values(['experiment', 'model', 'benchmark', 'contrast'])
    output['p_bh_within_experiment'] = output.groupby('experiment', group_keys=False).p_two_sided.transform(
        benjamini_hochberg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == '__main__':
    main()
