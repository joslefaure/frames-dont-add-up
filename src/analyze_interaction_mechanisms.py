#!/usr/bin/env python3
"""Classify continuous and answer-level outcomes of E4 pair composition."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path):
    return pd.DataFrame(json.loads(line) for line in path.read_text().splitlines() if line)


def summarize(table):
    indicators = [
        'positive_complementarity', 'joint_worse_than_best_singleton',
        'both_help_but_joint_worse', 'complementary_rescue',
        'destructive_interference', 'both_singletons_correct_joint_wrong',
    ]
    return table.groupby(['model', 'benchmark', 'pair_type']).agg(
        records=('item_id', 'size'),
        source_videos=('source_video_id', 'nunique'),
        **{name: (name, 'mean') for name in indicators},
        mean_joint_penalty_vs_best=('joint_penalty_vs_best', 'mean'),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results_root.glob('*.jsonl')):
        result = read_jsonl(path)
        model, benchmark = result.model.iloc[0], result.benchmark.iloc[0]
        design = pd.read_parquet(args.input_root / benchmark / 'e4' / 'all_conditions.parquet')
        source = design.drop_duplicates(['item_id', 'context_index', 'pair_type'])[
            ['item_id', 'context_index', 'pair_type', 'source_video_id']]
        index = ['model', 'benchmark', 'item_id', 'context_index', 'pair_type']
        logodds = result.pivot(index=index, columns='factorial_cell', values='gold_vs_rest_logodds')
        correct = result.pivot(index=index, columns='factorial_cell', values='correct').astype(bool)
        wide = logodds.join(correct, lsuffix='_logodds', rsuffix='_correct').reset_index()
        wide = wide.merge(source, on=['item_id', 'context_index', 'pair_type'], validate='one_to_one')
        wide['utility_a'] = wide.a_logodds - wide.baseline_logodds
        wide['utility_b'] = wide.b_logodds - wide.baseline_logodds
        wide['utility_ab'] = wide.ab_logodds - wide.baseline_logodds
        wide['interaction'] = wide.utility_ab - wide.utility_a - wide.utility_b
        best_singleton = wide[['utility_a', 'utility_b']].max(axis=1)
        wide['joint_penalty_vs_best'] = best_singleton - wide.utility_ab
        wide['positive_complementarity'] = wide.interaction > 0
        wide['joint_worse_than_best_singleton'] = wide.utility_ab < best_singleton
        wide['both_help_but_joint_worse'] = (
            (wide.utility_a > 0) & (wide.utility_b > 0) & wide.joint_worse_than_best_singleton)
        wide['complementary_rescue'] = (~wide.a_correct) & (~wide.b_correct) & wide.ab_correct
        wide['destructive_interference'] = (wide.a_correct | wide.b_correct) & (~wide.ab_correct)
        wide['both_singletons_correct_joint_wrong'] = (
            wide.a_correct & wide.b_correct & (~wide.ab_correct))
        rows.append(wide)
    records = pd.concat(rows, ignore_index=True)
    summary = summarize(records)
    macro = summary.groupby('pair_type').agg(
        settings=('model', 'size'),
        positive_complementarity=('positive_complementarity', 'mean'),
        joint_worse_than_best_singleton=('joint_worse_than_best_singleton', 'mean'),
        both_help_but_joint_worse=('both_help_but_joint_worse', 'mean'),
        complementary_rescue=('complementary_rescue', 'mean'),
        destructive_interference=('destructive_interference', 'mean'),
        both_singletons_correct_joint_wrong=('both_singletons_correct_joint_wrong', 'mean'),
        mean_joint_penalty_vs_best=('mean_joint_penalty_vs_best', 'mean'),
    ).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(args.output_dir / 'per_pair_mechanisms.parquet', index=False)
    summary.to_csv(args.output_dir / 'setting_summary.csv', index=False)
    macro.to_csv(args.output_dir / 'macro_summary.csv', index=False)
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
