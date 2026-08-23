#!/usr/bin/env python3
"""Materialize ordering, slot, and no-op controls for E4 on the E5 items."""
import argparse
import json
from pathlib import Path

import pandas as pd


def transformed(row, condition):
    row = row.copy()
    paths, times = json.loads(row.frame_paths), json.loads(row.frame_times)
    if condition == 'reversed':
        paths, times = paths[::-1], times[::-1]
    elif condition == 'slot_swap':
        # E4 fixes the candidate slots at the final two positions.
        paths[-2:], times[-2:] = paths[-2:][::-1], times[-2:][::-1]
    row['frame_paths'], row['frame_times'] = json.dumps(paths), json.dumps(times)
    row['robustness_condition'] = condition
    row['condition'] = f'e8_{condition}_{row.pair_type}_{row.factorial_cell}'
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e4-input', required=True, type=Path)
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    e4 = pd.read_parquet(args.e4_input)
    e5 = pd.read_parquet(args.e5_input)
    item_ids = set(e5.item_id.unique())
    e4 = e4[e4.item_id.isin(item_ids) & e4.pair_type.isin(['evidence_evidence', 'outside_outside'])].copy()
    if e4.item_id.nunique() != 25 or e4.context_index.nunique() != 3:
        raise ValueError('E8 requires the 25-item E5 subset and all three E4 contexts')
    rows = []
    for _, row in e4.iterrows():
        rows.extend([transformed(row, 'reversed'), transformed(row, 'slot_swap')])
    # A no-op 2x2 uses the same baseline context in all four factorial cells.
    baselines = e4[(e4.pair_type == 'evidence_evidence') & (e4.factorial_cell == 'baseline')]
    for _, row in baselines.iterrows():
        for cell in ('baseline', 'a', 'b', 'ab'):
            noop = row.copy()
            noop['pair_type'] = 'noop'
            noop['factorial_cell'] = cell
            noop['condition'] = f'e8_noop_noop_{cell}'
            noop['robustness_condition'] = 'noop'
            noop['candidate_a'] = -1
            noop['candidate_b'] = -1
            rows.append(noop)
    table = pd.DataFrame(rows).sort_values(['item_id', 'context_index', 'robustness_condition', 'pair_type', 'factorial_cell'])
    expected = 25 * 3 * ((2 * 2 * 4) + 4)
    if len(table) != expected:
        raise ValueError(f'row count {len(table)} != {expected}')
    if table.duplicated(['item_id', 'context_index', 'condition']).any():
        raise ValueError('duplicate E8 intervention key')
    if (table.frame_count != 8).any():
        raise ValueError('E8 must retain an 8-frame budget')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {'e4_input': str(args.e4_input.resolve()), 'e5_input': str(args.e5_input.resolve()),
                  'items': 25, 'contexts': 3, 'rows': len(table), 'frame_budget': 8,
                  'controls': ['reversed', 'slot_swap', 'noop']}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
