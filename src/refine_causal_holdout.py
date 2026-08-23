#!/usr/bin/env python3
"""Refine a causal holdout by requiring valid outside-evidence support."""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2

from build_manifests import stratified_holdout
from prepare_fingerprint_inputs import video_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--census', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--size', type=int, default=500)
    parser.add_argument('--decode-probe', action='store_true',
                        help='Require deterministic frame probes to decode before sampling.')
    parser.add_argument('--exclude-video-id', action='append', default=[])
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.census.read_text().splitlines() if line]
    supported, excluded, decode_cache = [], [], {}
    for row in rows:
        if row['source_video_id'] in set(args.exclude_video_id):
            excluded.append((row['item_id'], 'known_decode_failure'))
            continue
        if not row['evidence_intervals']:
            excluded.append((row['item_id'], 'no_grounding'))
            continue
        _, _, duration = video_info(Path(row['source_video_path']))
        candidates = np.linspace(0.02 * duration, 0.98 * duration, 2049)
        outside = [time for time in candidates if not any(start <= time <= end for start, end in row['evidence_intervals'])]
        if not outside:
            excluded.append((row['item_id'], 'no_outside_evidence_support'))
            continue
        if args.decode_probe:
            path = row['source_video_path']
            if path not in decode_cache:
                cap = cv2.VideoCapture(path)
                frame_count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
                probes = np.linspace(0.02, 0.98, 17)
                decode_cache[path] = frame_count > 0 and fps > 0
                for fraction in probes:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(fraction * (frame_count - 1))))
                    ok, _ = cap.read()
                    if not ok:
                        decode_cache[path] = False
                        break
                cap.release()
            if not decode_cache[path]:
                excluded.append((row['item_id'], 'decode_probe_failed'))
                continue
        supported.append(row)
    selected = stratified_holdout(supported, args.size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in selected))
    (args.output.with_suffix('.exclusions.json')).write_text(json.dumps({
        'census_items': len(rows), 'causal_eligible': len(supported), 'excluded': excluded,
        'selected': len(selected), 'seed': args.seed, 'decode_probe': args.decode_probe,
    }, indent=2) + '\n')
    print(f'selected {len(selected)} from {len(supported)} causal-eligible items; excluded {len(excluded)}')


if __name__ == '__main__':
    main()
