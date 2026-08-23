#!/usr/bin/env python3
"""Build E4-style factorial controls in two new frozen distractor contexts."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_causal_candidates import outside_times
from prepare_factorial_inputs import PAIR_TYPES
from prepare_fingerprint_inputs import extract_frames


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--context-indices', type=int, nargs='+', default=[3, 4])
    args = parser.parse_args()
    if sorted(args.context_indices) != [3, 4]:
        raise ValueError('E8 extra-context control is preregistered for new contexts 3 and 4')
    e5 = pd.read_parquet(args.e5_input)
    item_ids = set(e5.item_id.unique())
    if len(item_ids) != 25:
        raise ValueError(f'E8 requires the frozen 25-item E5 subset, found {len(item_ids)}')
    candidates = [json.loads(line) for line in args.candidates.read_text().splitlines() if line]
    candidates = [row for row in candidates if row['item_id'] in item_ids]
    candidates.sort(key=lambda row: row['item_id'])
    if len(candidates) != 25:
        raise ValueError('not all E5 items were found in the frozen E3 candidate file')

    selected_pair_types = [pair for pair in PAIR_TYPES if pair[0] in {'evidence_evidence', 'outside_outside'}]
    rows = []
    for sample_order, item in enumerate(candidates):
        duration = float(item['duration'])
        intervals = item['sampling_evidence_intervals']
        candidate_times = item['candidate_times']
        # Match the original deterministic context construction exactly, then
        # retain only the two previously unseen blocks.
        distractors = outside_times(duration, intervals, (max(args.context_indices) + 1) * 8)
        for context_index in args.context_indices:
            block = distractors[context_index * 8:(context_index + 1) * 8]
            shared, placeholder_a, placeholder_b = block[:6], block[6], block[7]
            for pair_type, candidate_a, candidate_b in selected_pair_types:
                cells = {
                    'baseline': shared + [placeholder_a, placeholder_b],
                    'a': shared + [candidate_times[candidate_a], placeholder_b],
                    'b': shared + [placeholder_a, candidate_times[candidate_b]],
                    'ab': shared + [candidate_times[candidate_a], candidate_times[candidate_b]],
                }
                for cell, times in cells.items():
                    key = stable(args.seed, item['item_id'], context_index, pair_type, cell)[:20]
                    paths = extract_frames(Path(item['source_video_path']), times,
                                           args.output_dir / 'frames' / f'ctx{context_index}', key)
                    rows.append({
                        **{key: value for key, value in item.items() if key not in ('contexts',)},
                        'sample_order': sample_order, 'context_index': context_index,
                        'condition': f'e8_extra_context_{pair_type}_{cell}',
                        'robustness_condition': 'extra_context', 'pair_type': pair_type,
                        'factorial_cell': cell, 'candidate_a': candidate_a, 'candidate_b': candidate_b,
                        'candidate_a_time': candidate_times[candidate_a],
                        'candidate_b_time': candidate_times[candidate_b],
                        'frame_times': json.dumps(times), 'frame_paths': json.dumps(paths),
                        'frame_count': 8, 'builder_seed': args.seed,
                    })
        print(f'[{sample_order + 1}/25] {item["item_id"]}', flush=True)
    table = pd.DataFrame(rows).sort_values(['sample_order', 'context_index', 'pair_type', 'factorial_cell'])
    expected = 25 * 2 * 2 * 4
    if len(table) != expected:
        raise ValueError(f'row count {len(table)} != {expected}')
    if table.duplicated(['item_id', 'context_index', 'condition']).any():
        raise ValueError('duplicate E8 extra-context intervention key')
    if (table.frame_count != 8).any():
        raise ValueError('E8 extra-context control must retain an 8-frame budget')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {
        'candidates': str(args.candidates.resolve()), 'e5_input': str(args.e5_input.resolve()),
        'seed': args.seed, 'items': 25, 'contexts': args.context_indices, 'rows': len(table),
        'frame_budget': 8, 'pair_types': [pair[0] for pair in selected_pair_types],
        'control': 'two new distractor contexts with original E4 candidate identities',
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
