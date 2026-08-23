#!/usr/bin/env python3
"""Create an ecological chronological variant of frozen E4 interventions.

Frame identities and factorial membership are unchanged.  Only the presentation
order is changed by sorting each condition's timestamp/path pairs.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def chronological(row):
    row = row.copy()
    times = json.loads(row.frame_times) if isinstance(row.frame_times, str) else list(row.frame_times)
    paths = json.loads(row.frame_paths) if isinstance(row.frame_paths, str) else list(row.frame_paths)
    if len(times) != len(paths) or len(times) != int(row.frame_count):
        raise ValueError(f'{row.item_id}/{row.condition}: timestamp/path/count mismatch')
    ordered = sorted(enumerate(zip(times, paths)), key=lambda value: (float(value[1][0]), value[0]))
    sorted_times = [float(pair[0]) for _, pair in ordered]
    sorted_paths = [pair[1] for _, pair in ordered]
    row['frame_times'] = json.dumps(sorted_times)
    row['frame_paths'] = json.dumps(sorted_paths)
    row['robustness_condition'] = 'chronological'
    row['condition'] = f'e8_chronological_{row.pair_type}_{row.factorial_cell}'
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--pair-types', nargs='+', default=['evidence_evidence', 'outside_outside'])
    args = parser.parse_args()
    source = pd.read_parquet(args.input)
    source = source[source.pair_type.isin(args.pair_types)].copy()
    table = pd.DataFrame([chronological(row) for _, row in source.iterrows()])
    expected_cells = {'baseline', 'a', 'b', 'ab'}
    coverage = table.groupby(['item_id', 'context_index', 'pair_type']).factorial_cell.agg(set)
    if not coverage.map(lambda cells: cells == expected_cells).all():
        raise ValueError('incomplete chronological factorial coverage')
    for _, row in table.iterrows():
        times = json.loads(row.frame_times)
        if any(left > right for left, right in zip(times, times[1:])):
            raise ValueError(f'{row.item_id}/{row.condition}: timestamps are not chronological')
    identity = source.copy()
    identity['identity'] = identity.apply(
        lambda row: sorted(zip(json.loads(row.frame_times), json.loads(row.frame_paths))), axis=1)
    transformed = table.copy()
    transformed['identity'] = transformed.apply(
        lambda row: sorted(zip(json.loads(row.frame_times), json.loads(row.frame_paths))), axis=1)
    if identity.identity.tolist() != transformed.identity.tolist():
        raise ValueError('chronological transform changed frame identity')
    table = table.drop(columns=['identity'], errors='ignore').sort_values(
        ['sample_order', 'context_index', 'pair_type', 'factorial_cell'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {
        'source_input': str(args.input.resolve()),
        'rows': len(table),
        'items': int(table.item_id.nunique()),
        'contexts': int(table.context_index.nunique()),
        'pair_types': args.pair_types,
        'frame_budget': sorted(table.frame_count.unique().astype(int).tolist()),
        'transform': 'stable ascending timestamp sort; frame identities unchanged',
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
