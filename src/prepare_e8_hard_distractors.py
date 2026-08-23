#!/usr/bin/env python3
"""Build E4 factorial controls with high-relevance outside-evidence context frames."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_factorial_inputs import PAIR_TYPES


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def outside(time, intervals):
    return not any(float(start) <= float(time) <= float(end) for start, end in intervals)


def candidate_path(template_paths, index):
    path = Path(json.loads(template_paths)[0])
    prefix = path.name.rsplit('_', 1)[0]
    return str(path.parent / f'{prefix}_{index:03d}.jpg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--e2-input', required=True, type=Path)
    parser.add_argument('--e4-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    args = parser.parse_args()
    e5_ids = set(pd.read_parquet(args.e5_input).item_id.unique())
    if len(e5_ids) != 25:
        raise ValueError(f'E8 requires 25 E5 items, found {len(e5_ids)}')
    candidates = {row['item_id']: row for row in
                  (json.loads(line) for line in args.candidates.read_text().splitlines() if line)}
    e2 = pd.read_parquet(args.e2_input)
    e2 = e2[e2.condition == 'siglip_relevance8'].set_index('item_id')
    if not e5_ids <= set(candidates) or not e5_ids <= set(e2.index):
        raise ValueError('all E5 items must have both frozen E3 and E2 provenance')
    selected_pair_types = [pair for pair in PAIR_TYPES if pair[0] in {'evidence_evidence', 'outside_outside'}]
    e4 = pd.read_parquet(args.e4_input)
    e4 = e4[e4.item_id.isin(e5_ids) & e4.pair_type.isin({pair[0] for pair in selected_pair_types})]
    candidate_paths = {}
    for _, row in e4.iterrows():
        paths = json.loads(row.frame_paths)
        if row.factorial_cell in {'a', 'ab'}:
            candidate_paths[(row.item_id, int(row.candidate_a))] = paths[-2]
        if row.factorial_cell in {'b', 'ab'}:
            candidate_paths[(row.item_id, int(row.candidate_b))] = paths[-1]
    expected_candidate_paths = len(e5_ids) * 4
    if len(candidate_paths) != expected_candidate_paths:
        raise ValueError(f'expected {expected_candidate_paths} frozen E4 candidate paths, found {len(candidate_paths)}')
    rows = []
    for sample_order, item_id in enumerate(sorted(e5_ids)):
        item, selector = candidates[item_id], e2.loc[item_id]
        intervals = item['sampling_evidence_intervals']
        duration = float(item['duration'])
        scores = json.loads(selector.siglip_relevance)
        charged = int(selector.candidate_frame_count_charged)
        if charged != 64 or len(scores) != charged:
            raise ValueError(f'{item_id}: expected exactly 64 frozen relevance scores')
        timestamps = np.linspace(0., max(0., duration - 1e-3), charged).tolist()
        ranked = sorted((index for index, time in enumerate(timestamps) if outside(time, intervals)),
                        key=lambda index: (-float(scores[index]), index))
        if len(ranked) < 8:
            raise ValueError(f'{item_id}: fewer than eight outside-evidence candidate frames')
        # All context/placeholder distractors have high relevance to the
        # question while remaining outside the privileged evidence window.
        hard_indices = sorted(ranked[:8])
        hard_times = [float(timestamps[index]) for index in hard_indices]
        hard_paths = [candidate_path(selector.frame_paths, index) for index in hard_indices]
        missing = [path for path in hard_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f'{item_id}: missing frozen E2 candidate frame {missing[0]}')
        shared_times, placeholders_times = hard_times[:6], hard_times[6:]
        shared_paths, placeholders_paths = hard_paths[:6], hard_paths[6:]
        candidate_times = item['candidate_times']
        for pair_type, candidate_a, candidate_b in selected_pair_types:
            cells = {
                'baseline': (shared_times + placeholders_times, shared_paths + placeholders_paths),
                'a': (shared_times + [candidate_times[candidate_a], placeholders_times[1]],
                      shared_paths + [candidate_paths[(item_id, candidate_a)], placeholders_paths[1]]),
                'b': (shared_times + [placeholders_times[0], candidate_times[candidate_b]],
                      shared_paths + [placeholders_paths[0], candidate_paths[(item_id, candidate_b)]]),
                'ab': (shared_times + [candidate_times[candidate_a], candidate_times[candidate_b]],
                       shared_paths + [candidate_paths[(item_id, candidate_a)], candidate_paths[(item_id, candidate_b)]]),
            }
            for cell, (times, paths) in cells.items():
                original = item
                rows.append({
                    **{key: value for key, value in original.items() if key not in ('contexts',)},
                    'sample_order': sample_order, 'context_index': 0,
                    'condition': f'e8_hard_distractor_{pair_type}_{cell}',
                    'robustness_condition': 'hard_distractor', 'pair_type': pair_type,
                    'factorial_cell': cell, 'candidate_a': candidate_a, 'candidate_b': candidate_b,
                    'candidate_a_time': candidate_times[candidate_a], 'candidate_b_time': candidate_times[candidate_b],
                    'frame_times': json.dumps(times), 'frame_paths': json.dumps(paths),
                    'frame_count': 8, 'hard_distractor_indices': json.dumps(hard_indices),
                    'candidate_frame_count_charged': charged, 'builder_seed': args.seed,
                })
    table = pd.DataFrame(rows)
    expected_rows = 25 * 2 * 4
    if len(table) != expected_rows:
        raise ValueError(f'row count {len(table)} != {expected_rows}')
    if table.duplicated(['item_id', 'context_index', 'condition']).any():
        raise ValueError('duplicate E8 hard-distractor intervention key')
    if table.frame_paths.map(lambda raw: any(path is None or not Path(path).is_file() for path in json.loads(raw))).any():
        raise FileNotFoundError('a frozen hard-distractor or E4 candidate frame is missing')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {
        'candidates': str(args.candidates.resolve()), 'e5_input': str(args.e5_input.resolve()),
        'e2_input': str(args.e2_input.resolve()), 'e4_input': str(args.e4_input.resolve()), 'seed': args.seed, 'items': 25,
        'rows': len(table), 'frame_budget': 8, 'candidate_frames_charged': 64,
        'pair_types': [pair[0] for pair in selected_pair_types],
        'control': 'top-SigLIP relevance distractors constrained outside annotated evidence',
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
