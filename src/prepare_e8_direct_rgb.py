#!/usr/bin/env python3
"""Re-decode frozen E4 frame timestamps into lossless PNG RGB controls."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd


def stable(*parts):
    return hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:20]


def decode_png(video, timestamp, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return str(target.resolve())
    cap = cv2.VideoCapture(str(video))
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    if count <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError(f'cannot read video metadata: {video}')
    index = max(0, min(count - 1, int(round(float(timestamp) * fps))))
    frame = None
    for offset in (0, -1, 1, -2, 2, -5, 5):
        candidate = index + offset
        if not 0 <= candidate < count:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, image = cap.read()
        if ok:
            frame = image
            break
    cap.release()
    if frame is None or not cv2.imwrite(str(target), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f'cannot write direct RGB PNG for {video} at frame {index}')
    return str(target.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e4-input', required=True, type=Path)
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    e5_ids = set(pd.read_parquet(args.e5_input).item_id.unique())
    table = pd.read_parquet(args.e4_input)
    table = table[table.item_id.isin(e5_ids) &
                  table.pair_type.isin(['evidence_evidence', 'outside_outside'])].copy()
    expected = 25 * 3 * 2 * 4
    if len(table) != expected:
        raise ValueError(f'E8 direct-RGB rows {len(table)} != {expected}')
    cache = {}
    new_paths = []
    for row_number, (_, row) in enumerate(table.iterrows(), 1):
        paths = []
        for frame_index, timestamp in enumerate(json.loads(row.frame_times)):
            key = (row.item_id, float(timestamp))
            if key not in cache:
                name = f'{stable(row.item_id, f"{float(timestamp):.9f}")}.png'
                cache[key] = decode_png(Path(row.source_video_path), timestamp,
                                        args.output_dir / 'frames' / name)
            paths.append(cache[key])
        new_paths.append(json.dumps(paths))
        if row_number % 50 == 0:
            print(f'[{row_number}/{expected}]', flush=True)
    table['frame_paths'] = new_paths
    table['robustness_condition'] = 'direct_rgb_png'
    table['condition'] = [f'e8_direct_rgb_{pair}_{cell}'
                          for pair, cell in zip(table.pair_type, table.factorial_cell)]
    if table.frame_paths.map(lambda raw: len(json.loads(raw)) != 8).any():
        raise ValueError('direct-RGB frame budget changed')
    if table.frame_paths.map(lambda raw: any(Path(path).suffix != '.png' or not Path(path).is_file()
                                              for path in json.loads(raw))).any():
        raise FileNotFoundError('missing direct RGB PNG frame')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {'e4_input': str(args.e4_input.resolve()), 'e5_input': str(args.e5_input.resolve()),
                  'items': 25, 'contexts': 3, 'rows': len(table), 'frame_budget': 8,
                  'pair_types': ['evidence_evidence', 'outside_outside'],
                  'codec': 'lossless PNG encoded directly from source-video RGB decode',
                  'unique_source_decodes': len(cache)}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
