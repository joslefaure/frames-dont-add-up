#!/usr/bin/env python3
"""Compare singleton, pairwise, and exhaustive selection on E5 subsets.

The analysis treats each four-candidate factorial table as an observed set
function.  At K=2 and K=3 it selects sets using (i) singleton utilities,
(ii) singleton plus pairwise Mobius terms, and (iii) the exhaustive observed
value.  Bootstrap intervals resample source videos and keep all questions and
contexts from a source video together.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path):
    return pd.DataFrame(json.loads(line) for line in path.read_text().splitlines() if line)


def mask_for(indices):
    return sum(1 << index for index in indices)


def best_by(candidates, score):
    """Use a stable lexicographic tie break while retaining achieved value."""
    ranked = sorted((float(score(indices)), tuple(indices)) for indices in candidates)
    return ranked[-1][1]


def analyze_group(group, source_video_id):
    keys = group.iloc[0]
    values = dict(zip(group.subset_mask.astype(int), group.gold_vs_rest_logodds.astype(float)))
    gold_prob = dict(zip(group.subset_mask.astype(int), np.exp(group.gold_option_logprob.astype(float))))
    correct = dict(zip(group.subset_mask.astype(int), group.correct.astype(bool)))
    if set(values) != set(range(16)):
        raise ValueError(f"{keys.item_id}/{keys.context_index}: incomplete subset grid")
    singleton = {index: values[1 << index] - values[0] for index in range(4)}
    pair = {
        (left, right): (
            values[(1 << left) | (1 << right)]
            - values[1 << left]
            - values[1 << right]
            + values[0]
        )
        for left, right in itertools.combinations(range(4), 2)
    }
    rows = []
    for budget in (2, 3):
        candidate_sets = list(itertools.combinations(range(4), budget))
        singleton_set = best_by(candidate_sets, lambda chosen: sum(singleton[index] for index in chosen))
        pairwise_set = best_by(
            candidate_sets,
            lambda chosen: (
                sum(singleton[index] for index in chosen)
                + sum(pair[indices] for indices in itertools.combinations(chosen, 2))
            ),
        )
        oracle_set = best_by(candidate_sets, lambda chosen: values[mask_for(chosen)])
        oracle_mask = mask_for(oracle_set)
        oracle_value = values[oracle_mask]
        for method, chosen in (
            ('singleton', singleton_set),
            ('pairwise', pairwise_set),
            ('exhaustive', oracle_set),
        ):
            chosen_mask = mask_for(chosen)
            rows.append({
                'model': keys.model,
                'benchmark': keys.benchmark,
                'item_id': keys.item_id,
                'source_video_id': source_video_id,
                'context_index': int(keys.context_index),
                'budget': budget,
                'method': method,
                'selected_mask': chosen_mask,
                'oracle_mask': oracle_mask,
                'optimal_set_recovered': values[chosen_mask] >= oracle_value - 1e-10,
                'logodds': values[chosen_mask],
                'logodds_regret': oracle_value - values[chosen_mask],
                'gold_probability': gold_prob[chosen_mask],
                'gold_probability_regret': gold_prob[oracle_mask] - gold_prob[chosen_mask],
                'correct': correct[chosen_mask],
            })
    return rows


def bootstrap(table, draws, seed):
    metrics = ('optimal_set_recovered', 'logodds_regret', 'gold_probability_regret', 'correct')
    rows = []
    for group_index, ((model, benchmark, budget, method), group) in enumerate(
        table.groupby(['model', 'benchmark', 'budget', 'method'], sort=True)
    ):
        by_video = group.groupby('source_video_id')[list(metrics)].mean()
        values = by_video.to_numpy(float)
        rng = np.random.default_rng(seed + group_index)
        samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
        for metric_index, metric in enumerate(metrics):
            low, high = np.quantile(samples[:, metric_index], [.025, .975])
            rows.append({
                'model': model,
                'benchmark': benchmark,
                'budget': budget,
                'method': method,
                'metric': metric,
                'source_video_clusters': len(values),
                'estimate': values[:, metric_index].mean(),
                'ci_low': low,
                'ci_high': high,
                'bootstrap_draws': draws,
            })
    return pd.DataFrame(rows)


def paired_contrasts(table, draws, seed):
    comparisons = (
        ('pairwise_minus_singleton', 'pairwise', 'singleton'),
        ('exhaustive_minus_singleton', 'exhaustive', 'singleton'),
        ('exhaustive_minus_pairwise', 'exhaustive', 'pairwise'),
    )
    # Signs are oriented so that positive always means the left method is
    # better: recovery/accuracy increase, while regret decreases.
    metrics = {
        'optimal_set_recovery_gain': ('optimal_set_recovered', 1.0),
        'logodds_regret_reduction': ('logodds_regret', -1.0),
        'gold_probability_regret_reduction': ('gold_probability_regret', -1.0),
        'accuracy_gain': ('correct', 1.0),
    }
    rows = []
    keys = ['model', 'benchmark', 'item_id', 'source_video_id', 'context_index', 'budget']
    for group_index, ((model, benchmark, budget), group) in enumerate(
        table.groupby(['model', 'benchmark', 'budget'], sort=True)
    ):
        for contrast, left, right in comparisons:
            for metric_name, (column, orientation) in metrics.items():
                pivot = group.pivot(index=keys, columns='method', values=column).dropna(subset=[left, right])
                difference = orientation * (pivot[left].astype(float) - pivot[right].astype(float))
                video = difference.groupby('source_video_id').mean().to_numpy()
                rng = np.random.default_rng(seed + 1000 * group_index + len(rows))
                samples = video[rng.integers(0, len(video), size=(draws, len(video)))].mean(axis=1)
                null = (video[None, :] * rng.choice((-1.0, 1.0), size=(draws, len(video)))).mean(axis=1)
                estimate = video.mean()
                low, high = np.quantile(samples, [.025, .975])
                p_value = (1 + (np.abs(null) >= abs(estimate)).sum()) / (draws + 1)
                rows.append({
                    'model': model, 'benchmark': benchmark, 'budget': budget,
                    'contrast': contrast, 'metric': metric_name,
                    'source_video_clusters': len(video), 'estimate': estimate,
                    'ci_low': low, 'ci_high': high, 'p_two_sided': p_value,
                    'bootstrap_draws': draws,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', required=True, type=Path)
    parser.add_argument('--inputs-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260720)
    args = parser.parse_args()
    source_maps = {}
    for path in sorted(args.inputs_root.glob('*/all_conditions.parquet')):
        design = pd.read_parquet(path)
        source_maps[path.parent.name] = design.drop_duplicates('item_id').set_index('item_id').source_video_id
    rows = []
    for path in sorted(args.results_root.glob('*.jsonl')):
        table = read_jsonl(path)
        benchmark = str(table.benchmark.iloc[0])
        if benchmark not in source_maps:
            raise ValueError(f'{path}: no E5 input/source-video map for {benchmark}')
        for (_, _, item_id, _), group in table.groupby(
            ['model', 'benchmark', 'item_id', 'context_index'], sort=False
        ):
            rows.extend(analyze_group(group, source_maps[benchmark].loc[item_id]))
    result = pd.DataFrame(rows).sort_values(
        ['model', 'benchmark', 'item_id', 'context_index', 'budget', 'method'])
    summary = result.groupby(['model', 'benchmark', 'budget', 'method']).agg(
        records=('item_id', 'size'),
        source_videos=('source_video_id', 'nunique'),
        optimal_set_recovery=('optimal_set_recovered', 'mean'),
        mean_logodds_regret=('logodds_regret', 'mean'),
        mean_gold_probability_regret=('gold_probability_regret', 'mean'),
        accuracy=('correct', 'mean'),
    ).reset_index()
    intervals = bootstrap(result, args.draws, args.seed)
    contrasts = paired_contrasts(result, args.draws, args.seed + 500000)
    macro = summary.groupby(['budget', 'method']).agg(
        settings=('model', 'size'),
        optimal_set_recovery=('optimal_set_recovery', 'mean'),
        mean_logodds_regret=('mean_logodds_regret', 'mean'),
        mean_gold_probability_regret=('mean_gold_probability_regret', 'mean'),
        accuracy=('accuracy', 'mean'),
    ).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_dir / 'per_selection.parquet', index=False)
    summary.to_csv(args.output_dir / 'setting_summary.csv', index=False)
    intervals.to_csv(args.output_dir / 'cluster_bootstrap.csv', index=False)
    contrasts.to_csv(args.output_dir / 'paired_method_contrasts.csv', index=False)
    macro.to_csv(args.output_dir / 'macro_summary.csv', index=False)
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
