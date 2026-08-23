#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('--require-grounding', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.path.read_text().splitlines()]
    assert rows, 'empty manifest'
    ids = [row['item_id'] for row in rows]
    assert len(ids) == len(set(ids)), 'duplicate item IDs'
    for row in rows:
        assert Path(row['source_video_path']).is_file(), row['source_video_path']
        assert len(row['options']) in (4, 5), row['item_id']
        assert row['answer'] in 'ABCDE'[:len(row['options'])], row['item_id']
        if args.require_grounding:
            assert row['evidence_intervals'], row['item_id']
    print(f'valid {len(rows)} rows: {args.path}')


if __name__ == '__main__':
    main()
