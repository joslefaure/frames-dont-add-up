#!/usr/bin/env python3
"""Write the frozen E5 item subset as a selector-ready manifest."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_parquet(args.e5_input).sort_values('sample_order')
    columns = ['annotation_kind', 'answer', 'benchmark', 'category', 'duration_band', 'evidence_intervals',
               'evidence_reference', 'item_id', 'options', 'question', 'question_type', 'source_video_id',
               'source_video_path']
    columns = [column for column in columns if column in table]
    table = table.drop_duplicates('item_id')[columns]
    if len(table) != 25:
        raise ValueError(f'candidate-density control requires 25 frozen E5 items, found {len(table)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, sort_keys=True,
                                               default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value)) + '\n'
                                   for row in table.to_dict('records')))
    print(json.dumps({'items': len(table), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
