#!/usr/bin/env python3
"""Join E2 predictions to frozen selector choices and report charged-frame metrics."""
import argparse
import json
from pathlib import Path

import pandas as pd


def in_intervals(time, intervals):
    return any(float(start) <= float(time) <= float(end) for start, end in intervals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    results = pd.DataFrame(json.loads(line) for line in args.results.read_text().splitlines() if line)
    inputs = pd.read_parquet(args.inputs)
    columns = ['item_id', 'condition', 'duration', 'frame_times', 'sampling_evidence_intervals',
               'candidate_frame_count_charged', 'selected_candidate_indices']
    table = results.merge(inputs[columns], on=['item_id', 'condition'], validate='one_to_one', suffixes=('', '_input'))
    def metrics(row):
        times = json.loads(row.frame_times_input)
        intervals = json.loads(row.sampling_evidence_intervals)
        recall = sum(in_intervals(time, intervals) for time in times) / len(times) if intervals else float('nan')
        coverage = (max(times) - min(times)) / row.duration if len(times) > 1 else 0.
        return pd.Series({'annotated_evidence_recall_at_k': recall, 'temporal_coverage': coverage})
    table = pd.concat([table, table.apply(metrics, axis=1)], axis=1)
    summary = table.groupby(['model', 'benchmark', 'condition']).agg(
        items=('item_id', 'size'), accuracy=('correct', 'mean'), mean_logodds=('gold_vs_rest_logodds', 'mean'),
        evidence_recall_at_k=('annotated_evidence_recall_at_k', 'mean'),
        temporal_coverage=('temporal_coverage', 'mean'),
        charged_candidate_frames=('candidate_frame_count_charged', 'mean'),
        mean_seconds=('seconds', 'mean')).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'per_item.parquet', index=False)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
