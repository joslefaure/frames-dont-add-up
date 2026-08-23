#!/usr/bin/env python3
"""Construct fixed-budget selector interaction-debt interventions for E6.

For every E2 selector output, retain E5's frozen question and two outside-frame
contexts.  We evaluate: baseline, each selected frame replacing its assigned
placeholder, and the selector's complete eight-frame set.  Thus
``sum_i U_i - (L(S) - L(C))`` is identified without changing the visual budget.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def load_paths(raw):
    value = json.loads(raw)
    if len(value) != 8:
        raise ValueError(f'expected eight paths, found {len(value)}')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e2-input', required=True, type=Path)
    parser.add_argument('--e5-input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    e2 = pd.read_parquet(args.e2_input)
    e5 = pd.read_parquet(args.e5_input)
    required_e2 = {'item_id', 'condition', 'frame_paths', 'frame_times', 'selected_candidate_indices'}
    required_e5 = {'item_id', 'context_index', 'subset_mask', 'frame_paths', 'frame_times'}
    if missing := required_e2 - set(e2):
        raise ValueError(f'E2 input lacks {sorted(missing)}')
    if missing := required_e5 - set(e5):
        raise ValueError(f'E5 input lacks {sorted(missing)}')
    baseline = e5[e5.subset_mask == 0].copy()
    if baseline.duplicated(['item_id', 'context_index']).any():
        raise ValueError('duplicate E5 baseline context')
    e2 = e2[e2.item_id.isin(set(baseline.item_id))].copy()
    conditions = sorted(e2.condition.unique())
    if len(conditions) != 8:
        raise ValueError(f'expected eight E2 selector conditions, found {conditions}')
    rows = []
    for context in baseline.to_dict('records'):
        base_paths = load_paths(context['frame_paths'])
        base_times = json.loads(context['frame_times'])
        selections = e2[e2.item_id == context['item_id']]
        if len(selections) != 8:
            raise ValueError(f"{context['item_id']}: missing selector output")
        for selection in selections.to_dict('records'):
            selector = selection['condition']
            selected_paths = load_paths(selection['frame_paths'])
            selected_times = json.loads(selection['frame_times'])
            common = {
                **{key: value for key, value in context.items() if key not in {'frame_paths', 'frame_times', 'condition'}},
                'benchmark': selection['benchmark'],
                'selector_condition': selector,
                'selected_candidate_indices': selection['selected_candidate_indices'],
                'selector_frame_times': selection['frame_times'],
                'frame_count': 8,
            }
            rows.append({
                **common, 'condition': f'{selector}__baseline', 'e6_cell': 'baseline',
                'selected_slot': -1, 'frame_paths': json.dumps(base_paths),
                'frame_times': json.dumps(base_times),
            })
            for slot in range(8):
                paths, times = list(base_paths), list(base_times)
                paths[slot], times[slot] = selected_paths[slot], selected_times[slot]
                rows.append({
                    **common, 'condition': f'{selector}__single_{slot}', 'e6_cell': 'single',
                    'selected_slot': slot, 'frame_paths': json.dumps(paths), 'frame_times': json.dumps(times),
                })
            rows.append({
                **common, 'condition': f'{selector}__full', 'e6_cell': 'full',
                'selected_slot': -1, 'frame_paths': json.dumps(selected_paths),
                'frame_times': json.dumps(selected_times),
            })
    table = pd.DataFrame(rows).sort_values(['item_id', 'context_index', 'selector_condition', 'e6_cell', 'selected_slot'])
    expected = baseline.item_id.nunique() * baseline.context_index.nunique() * 8 * 10
    if len(table) != expected:
        raise ValueError(f'row count {len(table)} != expected {expected}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'all_conditions.parquet', index=False)
    provenance = {
        'e2_input': str(args.e2_input.resolve()), 'e5_input': str(args.e5_input.resolve()),
        'items': int(baseline.item_id.nunique()), 'contexts': int(baseline.context_index.nunique()),
        'selectors': conditions, 'rows': len(table), 'frame_budget': 8,
        'design': 'baseline + eight single-slot replacements + full selector set per selector/context',
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, sort_keys=True))


if __name__ == '__main__':
    main()
