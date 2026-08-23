#!/usr/bin/env python3
"""Build fixed-budget E4 four-cell pairwise interventions from frozen E3 candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_causal_candidates import outside_times
from prepare_fingerprint_inputs import extract_frames


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


PAIR_TYPES = [
    ('evidence_evidence', 0, 1),
    ('evidence_boundary', 0, 2),
    ('evidence_outside', 0, 4),
    ('boundary_outside', 2, 4),
    ('outside_outside', 4, 5),
]


def load_rows(path, limit):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    rows.sort(key=lambda row: row['item_id'])
    return rows[:limit] if limit else rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--contexts', type=int, default=3)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output_rows = []
    for sample_order, item in enumerate(load_rows(args.candidates, args.limit)):
        duration = float(item['duration'])
        intervals = item['sampling_evidence_intervals']
        candidate_times = item['candidate_times']
        # Eight distractors per context: six shared frames plus two explicit
        # placeholders. Therefore every factorial cell exposes exactly K=8.
        distractors = outside_times(duration, intervals, args.contexts * 8)
        for context_index in range(args.contexts):
            block = distractors[context_index * 8:(context_index + 1) * 8]
            shared, placeholder_a, placeholder_b = block[:6], block[6], block[7]
            for pair_type, candidate_a, candidate_b in PAIR_TYPES:
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
                    output_rows.append({
                        **{key: value for key, value in item.items() if key not in ('contexts',)},
                        'sample_order': sample_order, 'context_index': context_index,
                        'condition': f'{pair_type}_{cell}', 'pair_type': pair_type,
                        'factorial_cell': cell, 'candidate_a': candidate_a, 'candidate_b': candidate_b,
                        'candidate_a_time': candidate_times[candidate_a],
                        'candidate_b_time': candidate_times[candidate_b],
                        'frame_times': json.dumps(times), 'frame_paths': json.dumps(paths),
                        'frame_count': 8, 'builder_seed': args.seed,
                    })
        print(f'[{sample_order + 1}] {item["item_id"]}', flush=True)
    table = pd.DataFrame(output_rows).sort_values(['sample_order', 'context_index', 'pair_type', 'factorial_cell'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {'candidates': str(args.candidates.resolve()), 'seed': args.seed,
                  'contexts': args.contexts, 'items': table.item_id.nunique(), 'rows': len(table),
                  'pair_types': [name for name, _, _ in PAIR_TYPES], 'frame_budget': 8}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
