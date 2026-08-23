#!/usr/bin/env python3
"""Materialize two byte-identical baseline input tables for E8 repeat checks."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e4-input', required=True, type=Path)
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    e5_ids = set(pd.read_parquet(args.e5_input).item_id.unique())
    base = pd.read_parquet(args.e4_input)
    base = base[(base.item_id.isin(e5_ids)) & (base.factorial_cell == 'baseline') &
                (base.pair_type.isin(['evidence_evidence', 'outside_outside']))].copy()
    expected = 25 * 3 * 2
    if len(base) != expected:
        raise ValueError(f'baseline rows {len(base)} != {expected}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for repeat in ('a', 'b'):
        table = base.copy()
        table['robustness_condition'] = f'repeat_baseline_{repeat}'
        table['condition'] = [f'e8_repeat_{repeat}_{pair}' for pair in table.pair_type]
        if table.duplicated(['item_id', 'context_index', 'condition']).any():
            raise ValueError('duplicate repeat-baseline key')
        table.to_parquet(args.output_dir / f'repeat_{repeat}.parquet', index=False)
    provenance = {'e4_input': str(args.e4_input.resolve()), 'e5_input': str(args.e5_input.resolve()),
                  'items': 25, 'contexts': 3, 'pair_types': ['evidence_evidence', 'outside_outside'],
                  'rows_per_repeat': expected, 'control': 'two byte-identical baseline input tables'}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
