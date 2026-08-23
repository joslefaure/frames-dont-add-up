#!/usr/bin/env python3
"""Stratify the primary E4 contrast by released annotation breadth."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def intervals(value):
    return json.loads(value) if isinstance(value, str) else list(value)


def union_length(raw):
    spans = sorted((float(left), float(right)) for left, right in raw)
    merged = []
    for left, right in spans:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return sum(max(0.0, right - left) for left, right in merged)


def bootstrap(video_values, draws, rng):
    values = np.asarray(video_values, dtype=float)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return values.mean(), *np.quantile(samples, [.025, .975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-root', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260720)
    args = parser.parse_args()
    record_tables = []
    for path in sorted(args.analysis_root.glob('*/per_pair_context.parquet')):
        table = pd.read_parquet(path)
        model, benchmark = table.model.iloc[0], table.benchmark.iloc[0]
        table = table[table.pair_type.isin(['evidence_evidence', 'outside_outside'])]
        pivot = table.pivot(
            index=['model', 'benchmark', 'item_id', 'context_index'],
            columns='pair_type', values='absolute_interaction').dropna().reset_index()
        pivot['absolute_interaction_gap'] = (
            pivot.evidence_evidence - pivot.outside_outside)
        design = pd.read_parquet(args.input_root / benchmark / 'e4' / 'all_conditions.parquet')
        item = design.drop_duplicates('item_id')[[
            'item_id', 'source_video_id', 'duration', 'evidence_intervals',
            'sampling_evidence_intervals']].copy()
        item['annotation_intervals'] = item.evidence_intervals.map(intervals)
        item['sampling_intervals'] = item.sampling_evidence_intervals.map(intervals)
        item['annotation_interval_count'] = item.annotation_intervals.map(len)
        item['annotation_coverage'] = [
            union_length(raw) / max(float(duration), 1e-9)
            for raw, duration in zip(item.annotation_intervals, item.duration)]
        item['sampling_coverage'] = [
            union_length(raw) / max(float(duration), 1e-9)
            for raw, duration in zip(item.sampling_intervals, item.duration)]
        record_tables.append(pivot.merge(
            item.drop(columns=['evidence_intervals', 'sampling_evidence_intervals']),
            on='item_id', validate='many_to_one'))
    records = pd.concat(record_tables, ignore_index=True)
    item_coverage = records.drop_duplicates(['benchmark', 'item_id'])[
        ['benchmark', 'item_id', 'sampling_coverage']].copy()
    item_coverage['coverage_quartile'] = item_coverage.groupby('benchmark').sampling_coverage.transform(
        lambda values: pd.qcut(values.rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4']))
    records = records.merge(item_coverage[['benchmark', 'item_id', 'coverage_quartile']],
                            on=['benchmark', 'item_id'], validate='many_to_one')
    rows = []
    for index, ((model, benchmark, quartile), group) in enumerate(
        records.groupby(['model', 'benchmark', 'coverage_quartile'], observed=True, sort=True)):
        per_video = group.groupby('source_video_id').absolute_interaction_gap.mean()
        estimate, low, high = bootstrap(
            per_video, args.draws, np.random.default_rng(args.seed + index))
        rows.append({
            'model': model, 'benchmark': benchmark, 'coverage_quartile': quartile,
            'items': group.item_id.nunique(), 'source_video_clusters': len(per_video),
            'median_sampling_coverage': group.drop_duplicates('item_id').sampling_coverage.median(),
            'equal_video_gap': estimate, 'ci_low': low, 'ci_high': high,
        })
    quartiles = pd.DataFrame(rows)
    thresholds = []
    for model, benchmark in records[['model', 'benchmark']].drop_duplicates().itertuples(index=False):
        cell = records[(records.model == model) & (records.benchmark == benchmark)]
        for threshold in (.10, .25, .50, 1.0):
            subset = cell[cell.sampling_coverage <= threshold]
            per_video = subset.groupby('source_video_id').absolute_interaction_gap.mean()
            if per_video.empty:
                continue
            estimate, low, high = bootstrap(
                per_video, args.draws,
                np.random.default_rng(args.seed + len(thresholds) + 10000))
            thresholds.append({
                'model': model, 'benchmark': benchmark,
                'maximum_sampling_coverage': threshold,
                'items': subset.item_id.nunique(), 'source_video_clusters': len(per_video),
                'equal_video_gap': estimate, 'ci_low': low, 'ci_high': high,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(args.output_dir / 'per_pair_annotation_breadth.parquet', index=False)
    quartiles.to_csv(args.output_dir / 'coverage_quartiles.csv', index=False)
    pd.DataFrame(thresholds).to_csv(args.output_dir / 'coverage_thresholds.csv', index=False)
    print(quartiles.to_string(index=False))


if __name__ == '__main__':
    main()
