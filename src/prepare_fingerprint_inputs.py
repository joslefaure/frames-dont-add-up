#!/usr/bin/env python3
"""Materialize fixed video-frame conditions from a frozen JSONL manifest.

This builder is deliberately model-agnostic: it decodes RGB frames once and
records their timestamps and paths, so every answerer sees the same evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def stable(seed, *parts):
    return hashlib.sha256(':'.join(map(str, (seed,) + parts)).encode()).hexdigest()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if count <= 0 or fps <= 0:
        raise RuntimeError(f'cannot read video metadata: {path}')
    return count, fps, count / fps


def evenly_spaced(start, stop, count):
    if count == 0:
        return []
    if stop <= start:
        return [float(start)] * count
    return np.linspace(start, stop, count + 2)[1:-1].astype(float).tolist()


def in_evidence(time, intervals):
    return any(float(start) <= time <= float(end) for start, end in intervals)


def evidence_times(intervals, count):
    if count == 0:
        return []
    lengths = np.asarray([max(1e-3, end - start) for start, end in intervals], dtype=float)
    ideal = count * lengths / lengths.sum()
    allocation = np.floor(ideal).astype(int)
    for index in np.argsort(-(ideal - allocation))[:count - allocation.sum()]:
        allocation[index] += 1
    times = []
    for (start, end), amount in zip(intervals, allocation):
        times.extend(evenly_spaced(float(start), float(end), int(amount)))
    return sorted(times)


def outside_times(duration, intervals, count):
    if count == 0:
        return []
    candidates = np.linspace(0.02 * duration, 0.98 * duration, 1025)
    valid = [float(time) for time in candidates if not in_evidence(float(time), intervals)]
    if not valid:
        valid = candidates.astype(float).tolist()
    indices = np.linspace(0, len(valid) - 1, count).round().astype(int)
    return [valid[index] for index in indices]


def extract_frames(video, times, frame_dir, key):
    count, fps, duration = video_info(video)
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    result = []
    for index, time in enumerate(times):
        time = max(0.0, min(float(time), duration - 1 / fps))
        frame_index = max(0, min(count - 1, int(round(time * fps))))
        target = frame_dir / f'{key}_{index:02d}.jpg'
        if not target.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            # A few source files have isolated seek/decode failures.  Preserve the
            # requested timestamp whenever possible, otherwise use the nearest
            # decodable frame rather than aborting an entire frozen intervention.
            if not ok:
                for offset in range(1, 6):
                    for candidate in (frame_index - offset, frame_index + offset):
                        if not 0 <= candidate < count:
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                        ok, frame = cap.read()
                        if ok:
                            break
                    if ok:
                        break
            if not ok:
                cap.release()
                raise RuntimeError(f'cannot decode {video} at frame {frame_index}')
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                cap.release()
                raise RuntimeError(f'cannot write {target}')
        result.append(str(target.resolve()))
    cap.release()
    return result


def parse_rows(path, limit):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    rows.sort(key=lambda row: row['item_id'])
    return rows[:limit] if limit else rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--point-window-seconds', type=float, default=2.0,
                        help='Half-width used only to sample a point reference as distinct frames.')
    parser.add_argument('--conditions', nargs='+',
                        default=['question_only', 'middle1', 'uniform4', 'uniform8', 'uniform16', 'uniform32', 'oracle8', 'dose1', 'dose2', 'dose4', 'dose8', 'oracle4_distractor4', 'oracle8_reversed', 'oracle8_shuffled'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    supported = {'question_only', 'middle1', 'uniform4', 'uniform8', 'uniform16', 'uniform32',
                 'oracle8', 'dose1', 'dose2', 'dose4', 'dose8', 'oracle4_distractor4', 'oracle8_reversed', 'oracle8_shuffled'}
    unknown = set(args.conditions) - supported
    if unknown:
        raise ValueError(f'unknown conditions: {sorted(unknown)}')
    rows, output_rows = parse_rows(args.manifest, args.limit), []
    for order, item in enumerate(rows):
        video = Path(item['source_video_path'])
        _, _, duration = video_info(video)
        intervals = [[max(0.0, float(start)), min(duration, float(end))]
                     for start, end in item['evidence_intervals']]
        intervals = [[start, end] for start, end in intervals if end >= start]
        sampling_intervals = [
            [max(0.0, start - args.point_window_seconds), min(duration, end + args.point_window_seconds)]
            if start == end else [start, end]
            for start, end in intervals
        ]
        uniform = lambda count: evenly_spaced(0.0, duration, count)
        oracle = lambda count: evidence_times(intervals, count)
        definitions = {
            'question_only': [], 'middle1': [duration / 2], 'uniform4': uniform(4),
            'uniform8': uniform(8), 'uniform16': uniform(16), 'uniform32': uniform(32),
        }
        if intervals:
            ordered_oracle = evidence_times(sampling_intervals, args.budget)
            shuffled_oracle = sorted(ordered_oracle, key=lambda time: stable(args.seed, item['item_id'], 'oracle8_shuffle', time))
            definitions.update({
                'oracle8': ordered_oracle,
                'dose1': sorted(evidence_times(sampling_intervals, 1) + outside_times(duration, sampling_intervals, args.budget - 1)),
                'dose2': sorted(evidence_times(sampling_intervals, 2) + outside_times(duration, sampling_intervals, args.budget - 2)),
                'dose4': sorted(evidence_times(sampling_intervals, 4) + outside_times(duration, sampling_intervals, args.budget - 4)),
                'dose8': evidence_times(sampling_intervals, args.budget),
                'oracle4_distractor4': sorted(evidence_times(sampling_intervals, 4) + outside_times(duration, sampling_intervals, 4)),
                'oracle8_reversed': list(reversed(ordered_oracle)),
                'oracle8_shuffled': shuffled_oracle,
            })
        for condition in args.conditions:
            if condition not in definitions:
                continue  # Census-only data have no interval-conditioned rows.
            times = definitions[condition]
            key = stable(args.seed, item['item_id'], condition)[:20]
            paths = extract_frames(video, times, args.output_dir / 'frames' / condition, key) if times else []
            output_rows.append({
                **item, 'sample_order': order, 'condition': condition, 'duration': duration,
                'sampling_evidence_intervals': sampling_intervals,
                'frame_times': json.dumps(times), 'frame_paths': json.dumps(paths),
                'frame_count': len(times), 'builder_seed': args.seed,
            })
        print(f'[{order + 1}/{len(rows)}] {item["item_id"]}', flush=True)
    table = pd.DataFrame(output_rows).sort_values(['condition', 'sample_order'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    for condition, group in table.groupby('condition'):
        group.to_parquet(args.output_dir / f'{condition}.parquet', index=False)
    provenance = {'manifest': str(args.manifest.resolve()), 'seed': args.seed, 'budget': args.budget,
                  'point_window_seconds': args.point_window_seconds,
                  'conditions_requested': args.conditions, 'items': len(rows), 'rows': len(table)}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
