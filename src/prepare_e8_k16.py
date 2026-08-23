#!/usr/bin/env python3
"""Expand frozen E4 interventions to K=16 with shared extra distractors."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_causal_candidates import outside_times
from prepare_fingerprint_inputs import extract_frames


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e4-input', required=True, type=Path)
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    args = parser.parse_args()
    e4 = pd.read_parquet(args.e4_input)
    e5_ids = set(pd.read_parquet(args.e5_input).item_id.unique())
    e4 = e4[e4.item_id.isin(e5_ids) & e4.pair_type.isin(['evidence_evidence', 'outside_outside'])].copy()
    if e4.item_id.nunique() != 25 or set(e4.context_index.unique()) != {0, 1, 2}:
        raise ValueError('E8 K16 requires the 25-item E5 subset and the original three E4 contexts')
    rows = []
    # Compute one new eight-frame distractor block per item/context and share
    # it across topology and factorial cells, preserving the 2x2 contrast.
    for (item_id, context_index), block in e4.groupby(['item_id', 'context_index'], sort=True):
        reference = block.iloc[0]
        duration = float(reference.duration)
        intervals = reference.sampling_evidence_intervals
        if isinstance(intervals, str):
            intervals = json.loads(intervals)
        intervals = [[float(start), float(end)] for start, end in intervals]
        # The original E4 uses three eight-frame blocks. Draw from a separate
        # deterministic block bank so these K16 additions are outside evidence
        # and not simply repeated original frames.
        pool = outside_times(duration, intervals, 48)
        extra_times = pool[(3 + int(context_index)) * 8:(4 + int(context_index)) * 8]
        if len(extra_times) != 8:
            raise ValueError(f'{item_id}: insufficient extra distractor times')
        key = stable(args.seed, item_id, context_index, 'e8_k16_extra')[:20]
        extra_paths = extract_frames(Path(reference.source_video_path), extra_times,
                                     args.output_dir / 'frames' / f'ctx{context_index}', key)
        for _, row in block.iterrows():
            paths, times = json.loads(row.frame_paths), json.loads(row.frame_times)
            expanded = row.copy()
            expanded['frame_paths'] = json.dumps(paths + extra_paths)
            expanded['frame_times'] = json.dumps(times + extra_times)
            expanded['frame_count'] = 16
            expanded['robustness_condition'] = 'k16'
            expanded['condition'] = f'e8_k16_{row.pair_type}_{row.factorial_cell}'
            rows.append(expanded)
        print(f'{item_id} context={context_index}', flush=True)
    table = pd.DataFrame(rows).sort_values(['sample_order', 'context_index', 'pair_type', 'factorial_cell'])
    expected = 25 * 3 * 2 * 4
    if len(table) != expected:
        raise ValueError(f'row count {len(table)} != {expected}')
    if table.duplicated(['item_id', 'context_index', 'condition']).any():
        raise ValueError('duplicate E8 K16 intervention key')
    if (table.frame_count != 16).any():
        raise ValueError('every E8 K16 row must expose exactly 16 frames')
    if table.frame_paths.map(lambda raw: len(json.loads(raw)) != 16).any():
        raise ValueError('E8 K16 path count mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {
        'e4_input': str(args.e4_input.resolve()), 'e5_input': str(args.e5_input.resolve()),
        'seed': args.seed, 'items': 25, 'contexts': 3, 'rows': len(table), 'frame_budget': 16,
        'pair_types': ['evidence_evidence', 'outside_outside'],
        'control': 'original K8 factorial frames plus eight shared outside-evidence distractors',
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
