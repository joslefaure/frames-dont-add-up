#!/usr/bin/env python3
"""Construct frozen E3 candidate/context interventions at an exact frame budget."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_fingerprint_inputs import (
    evenly_spaced, extract_frames, in_evidence, video_info,
)


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def point_sampling_intervals(intervals, duration, half_width):
    return [[max(0.0, start - half_width), min(duration, end + half_width)] if start == end else [start, end]
            for start, end in intervals]


def outside_times(duration, intervals, count):
    candidates = np.linspace(0.02 * duration, 0.98 * duration, 2049)
    valid = [float(time) for time in candidates if not in_evidence(float(time), intervals)]
    if not valid:
        raise RuntimeError('no outside-evidence candidate times')
    indices = np.linspace(0, len(valid) - 1, count).round().astype(int)
    return [valid[index] for index in indices]


def evidence_times(intervals, count):
    lengths = np.asarray([max(1e-3, end - start) for start, end in intervals], dtype=float)
    ideal = count * lengths / lengths.sum()
    allocation = np.floor(ideal).astype(int)
    for index in np.argsort(-(ideal - allocation))[:count - allocation.sum()]:
        allocation[index] += 1
    result = []
    for (start, end), number in zip(intervals, allocation):
        result.extend(evenly_spaced(start, end, int(number)))
    return sorted(result)


def candidates_for(intervals, sampling_intervals, duration):
    evidence = evidence_times(sampling_intervals, 2)
    boundaries = []
    for start, end in sampling_intervals:
        boundaries.extend([max(0.0, start - 1.0), min(duration, end + 1.0)])
    # Spread the two boundary candidates through the available interval edges.
    boundary_indices = np.linspace(0, len(boundaries) - 1, 2).round().astype(int)
    boundary = [boundaries[index] for index in boundary_indices]
    outside = outside_times(duration, sampling_intervals, 4)
    times = evidence + boundary + outside
    strata = ['evidence'] * 2 + ['boundary'] * 2 + ['outside'] * 4
    return times, strata


def load_manifest(path, limit):
    items = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    items.sort(key=lambda row: row['item_id'])
    return items[:limit] if limit else items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--contexts', type=int, default=3)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--point-window-seconds', type=float, default=2.0)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.budget != 8:
        raise ValueError('the frozen E3 design uses budget=8 (seven-context plus one replacement)')
    output_rows, manifest_rows = [], []
    for sample_order, item in enumerate(load_manifest(args.manifest, args.limit)):
        if not item['evidence_intervals']:
            raise ValueError(f'{item["item_id"]}: no usable evidence interval')
        _, _, duration = video_info(Path(item['source_video_path']))
        intervals = [[max(0.0, float(start)), min(duration, float(end))]
                     for start, end in item['evidence_intervals']]
        sampling_intervals = point_sampling_intervals(intervals, duration, args.point_window_seconds)
        candidate_times, candidate_strata = candidates_for(intervals, sampling_intervals, duration)
        contexts = []
        for context_index in range(args.contexts):
            # A deterministic rotated set of seven outside-evidence frames.
            pool = outside_times(duration, sampling_intervals, 7 * args.contexts)
            offset = context_index * 7
            context = pool[offset:offset + 7]
            contexts.append(context)
            variants = [('baseline', -1, context)]
            variants += [(f'candidate{index}', index, context[:3] + [time] + context[3:])
                         for index, time in enumerate(candidate_times)]
            for condition, candidate_index, times in variants:
                key = stable(args.seed, item['item_id'], context_index, condition)[:20]
                paths = extract_frames(Path(item['source_video_path']), times,
                                       args.output_dir / 'frames' / f'ctx{context_index}', key)
                output_rows.append({
                    **item, 'sample_order': sample_order, 'duration': duration,
                    'sampling_evidence_intervals': sampling_intervals, 'context_index': context_index,
                    'condition': condition, 'candidate_index': candidate_index,
                    'candidate_time': None if candidate_index < 0 else candidate_times[candidate_index],
                    'candidate_stratum': None if candidate_index < 0 else candidate_strata[candidate_index],
                    'frame_times': json.dumps(times), 'frame_paths': json.dumps(paths), 'frame_count': 8,
                    'builder_seed': args.seed,
                })
        manifest_rows.append({
            **item, 'duration': duration, 'sampling_evidence_intervals': sampling_intervals,
            'candidate_times': candidate_times, 'candidate_strata': candidate_strata, 'contexts': contexts,
        })
        print(f'[{sample_order + 1}] {item["item_id"]}', flush=True)
    table = pd.DataFrame(output_rows).sort_values(['sample_order', 'context_index', 'candidate_index'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    (args.output_dir / 'candidates.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in manifest_rows))
    provenance = {'manifest': str(args.manifest.resolve()), 'seed': args.seed, 'budget': args.budget,
                  'contexts': args.contexts, 'items': len(manifest_rows), 'rows': len(table),
                  'candidate_design': '2 evidence, 2 boundary-adjacent, 4 outside-evidence'}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
