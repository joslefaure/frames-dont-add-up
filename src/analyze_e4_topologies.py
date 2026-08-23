#!/usr/bin/env python3
"""Recover evidence-pair topology from frozen E4 design and summarize effects."""
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_intervals(value):
    return json.loads(value) if isinstance(value, str) else value


def interval_index(timestamp, intervals):
    for index, (start, end) in enumerate(intervals):
        if float(start) - 1e-6 <= float(timestamp) <= float(end) + 1e-6:
            return index
    return None


def evidence_relation(row):
    if row.pair_type != 'evidence_evidence':
        return row.pair_type
    intervals = parse_intervals(row.sampling_evidence_intervals)
    left = interval_index(row.candidate_a_time, intervals)
    right = interval_index(row.candidate_b_time, intervals)
    if left is None or right is None:
        return 'evidence_evidence_unresolved'
    return 'evidence_evidence_same_interval' if left == right else 'evidence_evidence_distinct_interval'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactions', required=True, type=Path,
                        help='aggregate_causal.py E4 per_pair_context.parquet')
    parser.add_argument('--design-input', required=True, type=Path,
                        help='frozen E4 all_conditions.parquet')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    interactions = pd.read_parquet(args.interactions)
    design = pd.read_parquet(args.design_input)
    design = design.drop_duplicates(['item_id', 'context_index', 'pair_type'])[
        ['item_id', 'context_index', 'pair_type', 'candidate_a_time', 'candidate_b_time',
         'sampling_evidence_intervals', 'source_video_id']]
    table = interactions.merge(design, on=['item_id', 'context_index', 'pair_type'], validate='one_to_one')
    table['topology'] = table.apply(evidence_relation, axis=1)
    summary = table.groupby(['model', 'benchmark', 'topology']).agg(
        cells=('interaction', 'size'), mean_interaction=('interaction', 'mean'),
        median_interaction=('interaction', 'median'),
        mean_absolute_interaction=('absolute_interaction', 'mean'),
        negative_rate=('interaction', lambda x: float((x < 0).mean()))).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'per_pair_topologies.parquet', index=False)
    summary.to_csv(args.output_dir / 'topology_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
