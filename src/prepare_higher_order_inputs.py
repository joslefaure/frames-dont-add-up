#!/usr/bin/env python3
"""Build fixed-K, all-subset E5 higher-order interventions.

Each of four evidence candidates occupies a fixed slot.  An absent candidate is
replaced by its context-specific outside-evidence placeholder, hence every one
of the 16 subsets exposes exactly eight frames.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_causal_candidates import evidence_times, outside_times, point_sampling_intervals
from prepare_fingerprint_inputs import extract_frames, video_info


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def load_items(path, seed, limit):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    return sorted(rows, key=lambda row: stable(seed, 'e5_item', row['item_id']))[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--items', type=int, default=25)
    parser.add_argument('--contexts', type=int, default=2)
    parser.add_argument('--point-window-seconds', type=float, default=2.)
    args = parser.parse_args()
    output_rows = []
    for sample_order, item in enumerate(load_items(args.manifest, args.seed, args.items)):
        _, _, duration = video_info(Path(item['source_video_path']))
        intervals = [[max(0., float(start)), min(duration, float(end))]
                     for start, end in item['evidence_intervals']]
        sampling_intervals = point_sampling_intervals(intervals, duration, args.point_window_seconds)
        evidence = evidence_times(sampling_intervals, 4)
        for context_index in range(args.contexts):
            # Four shared outside frames and four slot-specific placeholders.
            context = outside_times(duration, sampling_intervals, 8 * args.contexts)
            block = context[context_index * 8:(context_index + 1) * 8]
            shared, placeholders = block[:4], block[4:]
            for subset_mask in range(16):
                active = [index for index in range(4) if subset_mask & (1 << index)]
                times = shared + [evidence[index] if index in active else placeholders[index] for index in range(4)]
                key = stable(args.seed, item['item_id'], context_index, subset_mask)[:20]
                paths = extract_frames(Path(item['source_video_path']), times,
                                       args.output_dir / 'frames' / f'ctx{context_index}', key)
                output_rows.append({
                    **item, 'sample_order': sample_order, 'duration': duration,
                    'sampling_evidence_intervals': sampling_intervals, 'context_index': context_index,
                    'condition': f'subset_{subset_mask:02d}', 'subset_mask': subset_mask,
                    'active_candidates': json.dumps(active), 'candidate_times': json.dumps(evidence),
                    'frame_times': json.dumps(times), 'frame_paths': json.dumps(paths),
                    'frame_count': 8, 'builder_seed': args.seed,
                })
        print(f'[{sample_order + 1}/{args.items}] {item["item_id"]}', flush=True)
    table = pd.DataFrame(output_rows).sort_values(['sample_order', 'context_index', 'subset_mask'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    (args.output_dir / 'provenance.json').write_text(json.dumps({
        'manifest': str(args.manifest.resolve()), 'seed': args.seed, 'items': args.items,
        'contexts': args.contexts, 'subsets_per_item': 16, 'frame_budget': 8,
        'selection': 'deterministic hash rank; four interval-distributed evidence samples',
    }, indent=2) + '\n')
    print(json.dumps({'rows': len(table), 'items': table.item_id.nunique()}, sort_keys=True))


if __name__ == '__main__':
    main()
