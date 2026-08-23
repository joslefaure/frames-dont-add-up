#!/usr/bin/env python3
"""Fail fast on the structural invariants of frozen intervention tables."""
import argparse
import json
from pathlib import Path

import pandas as pd


def paths_exist(series):
    missing = []
    for raw in series:
        for path in json.loads(raw):
            if not Path(path).is_file():
                missing.append(path)
                if len(missing) == 5:
                    return missing
    return missing


def require(observed, expected, label):
    if set(observed) != set(expected):
        raise ValueError(f'{label}: expected {sorted(expected)}, found {sorted(observed)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--experiment', choices=['e1', 'e1_census', 'e2', 'e2_k16', 'e3', 'e4', 'e6', 'e8_density'], required=True)
    args = parser.parse_args()
    table = pd.read_parquet(args.input)
    if table.empty:
        raise ValueError('prepared table is empty')
    items = table.item_id.nunique()
    expected_frame_count = 16 if args.experiment == 'e2_k16' else 8
    if args.experiment in {'e2', 'e2_k16', 'e3', 'e4', 'e6', 'e8_density'} and (table.frame_count.isna().any() or (table.frame_count != expected_frame_count).any()):
        raise ValueError(f'every intervention row must expose exactly {expected_frame_count} frames')
    missing = paths_exist(table.frame_paths)
    if missing:
        raise FileNotFoundError(f'missing prepared frame files (first five): {missing}')
    if args.experiment == 'e6':
        expected_selectors = {'uniform8', 'random8', 'siglip_relevance8', 'dino_change8', 'dino_diversity8',
                              'siglip_dino_submodular8', 'bolt_its8', 'aks_fixedk8'}
        require(table.selector_condition.unique(), expected_selectors, 'E6 selector conditions')
        contexts = table.context_index.nunique()
        if len(table) != items * contexts * len(expected_selectors) * 10:
            raise ValueError(f'E6 row count {len(table)} != {items} * {contexts} * 8 * 10')
        if table.duplicated(['item_id', 'context_index', 'condition']).any():
            raise ValueError('duplicate E6 item/context/condition row')
        structure = table.groupby(['item_id', 'context_index', 'selector_condition']).agg(
            rows=('e6_cell', 'size'), baselines=('e6_cell', lambda x: int((x == 'baseline').sum())),
            singles=('e6_cell', lambda x: int((x == 'single').sum())), fulls=('e6_cell', lambda x: int((x == 'full').sum())))
        if not ((structure.rows == 10) & (structure.baselines == 1) & (structure.singles == 8) & (structure.fulls == 1)).all():
            raise ValueError('E6 selector/context blocks must contain baseline, eight singles, and full set')
    elif args.experiment in {'e2', 'e2_k16', 'e8_density'}:
        budget = 16 if args.experiment == 'e2_k16' else 8
        expected = {f'uniform{budget}', f'random{budget}', f'siglip_relevance{budget}', f'dino_change{budget}',
                    f'dino_diversity{budget}', f'siglip_dino_submodular{budget}', f'bolt_its{budget}', f'aks_fixedk{budget}'}
        require(table.condition.unique(), expected, 'E2 selector conditions')
        if len(table) != items * len(expected):
            raise ValueError(f'E2 row count {len(table)} != {items} * {len(expected)}')
        if table.duplicated(['item_id', 'condition']).any():
            raise ValueError('duplicate E2 item/condition row')
        counts = table.groupby('item_id').condition.nunique()
        if not (counts == len(expected)).all():
            raise ValueError('E2 conditions do not share an identical retained item universe')
        charged = table.candidate_frame_count_charged
        if charged.isna().any() or charged.nunique() != 1:
            raise ValueError('every selector-density table must have one explicit charged candidate count')
        if args.experiment in {'e2', 'e2_k16'} and (charged != 64).any():
            raise ValueError('every E2 row must charge exactly 64 candidate frames')
        if args.experiment == 'e8_density' and int(charged.iloc[0]) not in {32, 128}:
            raise ValueError('E8 candidate-density table must charge either 32 or 128 frames')
        if table.selected_candidate_indices.map(lambda raw: len(json.loads(raw))).ne(budget).any():
            raise ValueError(f'every E2 selector row must contain exactly {budget} selected candidate indices')
    elif args.experiment == 'e3':
        expected = {'baseline', *(f'candidate{i}' for i in range(8))}
        require(table.condition.unique(), expected, 'E3 conditions')
        if len(table) != items * 3 * 9:
            raise ValueError(f'E3 row count {len(table)} != {items} * 3 * 9')
        if table.duplicated(['item_id', 'context_index', 'condition']).any():
            raise ValueError('duplicate E3 item/context/condition row')
    elif args.experiment == 'e4':
        pair_types = {'evidence_evidence', 'evidence_boundary', 'evidence_outside', 'boundary_outside', 'outside_outside'}
        cells = {'baseline', 'a', 'b', 'ab'}
        require(table.pair_type.unique(), pair_types, 'E4 pair types')
        require(table.factorial_cell.unique(), cells, 'E4 factorial cells')
        if len(table) != items * 3 * 5 * 4:
            raise ValueError(f'E4 row count {len(table)} != {items} * 3 * 5 * 4')
        if table.duplicated(['item_id', 'context_index', 'pair_type', 'factorial_cell']).any():
            raise ValueError('duplicate E4 item/context/pair/cell row')
    else:
        expected = ({'question_only', 'middle1', 'uniform4', 'uniform8', 'uniform16', 'uniform32'}
                    if args.experiment == 'e1_census' else
                    {'question_only', 'middle1', 'uniform4', 'uniform8', 'uniform16', 'uniform32',
                     'oracle8', 'dose1', 'dose2', 'dose4', 'dose8', 'oracle4_distractor4',
                     'oracle8_reversed', 'oracle8_shuffled'})
        require(table.condition.unique(), expected, 'E1 census conditions' if args.experiment == 'e1_census' else 'E1 conditions')
        if len(table) != items * len(expected):
            raise ValueError(f'E1 row count {len(table)} != {items} * {len(expected)}')
        if table.duplicated(['item_id', 'condition']).any():
            raise ValueError('duplicate E1 item/condition row')
    print(json.dumps({'experiment': args.experiment, 'input': str(args.input), 'rows': len(table), 'items': items, 'status': 'ok'}))


if __name__ == '__main__':
    main()
