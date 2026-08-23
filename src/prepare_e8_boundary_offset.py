#!/usr/bin/env python3
"""Rebuild the frozen E4 topology intervention with a wider boundary offset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_causal_candidates import outside_times
from prepare_factorial_inputs import PAIR_TYPES
from prepare_fingerprint_inputs import extract_frames


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def offset_boundaries(intervals, duration, offset):
    edges = []
    for start, end in intervals:
        edges.extend([max(0., float(start) - offset), min(float(duration), float(end) + offset)])
    indices = np.linspace(0, len(edges) - 1, 2).round().astype(int)
    return [float(edges[index]) for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--offset-seconds', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=20260720)
    args = parser.parse_args()
    if args.offset_seconds <= 1:
        raise ValueError('boundary-offset robustness must differ from the original ±1 second construction')
    item_ids = set(pd.read_parquet(args.e5_input).item_id.unique())
    if len(item_ids) != 25:
        raise ValueError(f'expected the frozen 25-item E5 subset, found {len(item_ids)}')
    candidates = [json.loads(line) for line in args.candidates.read_text().splitlines() if line]
    candidates = sorted((row for row in candidates if row['item_id'] in item_ids), key=lambda row: row['item_id'])
    if len(candidates) != 25:
        raise ValueError('the E5 subset is not fully represented in the E3 candidates')
    rows = []
    for sample_order, item in enumerate(candidates):
        duration, intervals = float(item['duration']), item['sampling_evidence_intervals']
        times = list(map(float, item['candidate_times']))
        times[2:4] = offset_boundaries(intervals, duration, args.offset_seconds)
        # The non-boundary candidates and all contexts are intentionally held fixed.
        distractors = outside_times(duration, intervals, 3 * 8)
        for context_index in range(3):
            block = distractors[context_index * 8:(context_index + 1) * 8]
            shared, placeholder_a, placeholder_b = block[:6], block[6], block[7]
            for pair_type, candidate_a, candidate_b in PAIR_TYPES:
                cells = {
                    'baseline': shared + [placeholder_a, placeholder_b],
                    'a': shared + [times[candidate_a], placeholder_b],
                    'b': shared + [placeholder_a, times[candidate_b]],
                    'ab': shared + [times[candidate_a], times[candidate_b]],
                }
                for cell, frame_times in cells.items():
                    key = stable(args.seed, item['item_id'], context_index, pair_type, cell, 'boundary_offset', args.offset_seconds)[:20]
                    paths = extract_frames(Path(item['source_video_path']), frame_times,
                                           args.output_dir / 'frames' / f'ctx{context_index}', key)
                    rows.append({
                        **{key: value for key, value in item.items() if key != 'contexts'},
                        'sample_order': sample_order, 'context_index': context_index,
                        'condition': f'e8_boundary_offset_{pair_type}_{cell}',
                        'robustness_condition': 'boundary_offset_4s', 'pair_type': pair_type,
                        'factorial_cell': cell, 'candidate_a': candidate_a, 'candidate_b': candidate_b,
                        'candidate_a_time': times[candidate_a], 'candidate_b_time': times[candidate_b],
                        'frame_times': json.dumps(frame_times), 'frame_paths': json.dumps(paths),
                        'frame_count': 8, 'boundary_offset_seconds': args.offset_seconds, 'builder_seed': args.seed,
                    })
        print(f'[{sample_order + 1}/25] {item["item_id"]}', flush=True)
    table = pd.DataFrame(rows).sort_values(['sample_order', 'context_index', 'pair_type', 'factorial_cell'])
    if len(table) != 25 * 3 * len(PAIR_TYPES) * 4 or (table.frame_count != 8).any():
        raise ValueError('invalid boundary-offset factorial table')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {'candidates': str(args.candidates.resolve()), 'e5_input': str(args.e5_input.resolve()),
                  'seed': args.seed, 'items': 25, 'contexts': 3, 'rows': len(table), 'frame_budget': 8,
                  'pair_types': [name for name, _, _ in PAIR_TYPES], 'boundary_offset_seconds': args.offset_seconds,
                  'control': 'only boundary candidates shift from original ±1 s to ±4 s around the same sampling windows'}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
