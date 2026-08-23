#!/usr/bin/env python3
"""Test whether E6 interaction debt predicts correctness beyond selector proxies."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def in_intervals(time, intervals):
    return any(float(start) <= float(time) <= float(end) for start, end in intervals)


def selector_covariates(input_root):
    tables = []
    for benchmark_dir in sorted(input_root.iterdir()):
        path = benchmark_dir / 'all_conditions.parquet'
        if not path.is_file():
            continue
        table = pd.read_parquet(path)
        for row in table.itertuples(index=False):
            indices = json.loads(row.selected_candidate_indices)
            scores = json.loads(row.siglip_relevance)
            times = json.loads(row.frame_times)
            intervals = json.loads(row.sampling_evidence_intervals)
            tables.append({
                'benchmark': row.benchmark, 'item_id': row.item_id,
                'selector_condition': row.condition, 'source_video_id': row.source_video_id,
                'evidence_recall': float(sum(in_intervals(time, intervals) for time in times) / len(times)),
                'summed_siglip_relevance': float(sum(scores[index] for index in indices)),
                'temporal_coverage': float((max(times) - min(times)) / row.duration if len(times) > 1 else 0.),
                'frame_count': int(row.frame_count),
            })
    result = pd.DataFrame(tables)
    if result.duplicated(['benchmark', 'item_id', 'selector_condition']).any():
        raise ValueError('duplicate E2 selector covariate row')
    return result


def fit_clustered_logistic(table, continuous):
    """Unpenalized IRLS logit with a source-video cluster sandwich covariance."""
    pieces, names = [np.ones((len(table), 1))], ['Intercept']
    for column in continuous:
        pieces.append(table[[f'z_{column}']].to_numpy(float))
        names.append(f'z_{column}')
    for column in ('model', 'benchmark'):
        levels = sorted(table[column].unique())
        for level in levels[1:]:
            pieces.append((table[column] == level).to_numpy(float)[:, None])
            names.append(f'{column}[{level}]')
    design = np.hstack(pieces)
    target = table.full_correct.astype(float).to_numpy()
    coefficients = np.zeros(design.shape[1])
    for _ in range(100):
        probability = expit(design @ coefficients)
        weight = np.clip(probability * (1 - probability), 1e-7, None)
        hessian = design.T @ (weight[:, None] * design)
        score = design.T @ (target - probability)
        step = np.linalg.solve(hessian, score)
        coefficients += step
        if np.max(np.abs(step)) < 1e-9:
            break
    else:
        raise RuntimeError('logistic IRLS did not converge')
    probability = expit(design @ coefficients)
    weight = np.clip(probability * (1 - probability), 1e-7, None)
    bread = np.linalg.inv(design.T @ (weight[:, None] * design))
    scores = design * (target - probability)[:, None]
    meat = np.zeros_like(bread)
    for _, indices in table.groupby('source_video_id', sort=False).indices.items():
        group_score = scores[np.asarray(indices)].sum(axis=0)
        meat += np.outer(group_score, group_score)
    covariance = bread @ meat @ bread
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    z_values = coefficients / errors
    p_values = 2 * norm.sf(np.abs(z_values))
    return pd.DataFrame({'term': names, 'coefficient': coefficients,
                         'std_error_clustered_source_video': errors, 'z_value': z_values,
                         'p_value': p_values, 'odds_ratio_per_sd': np.exp(coefficients)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e6-root', required=True, type=Path)
    parser.add_argument('--e2-input-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.e6_root.glob('*/per_set.parquet'))
    if not paths:
        raise ValueError('no E6 per-set tables found')
    debt = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    covariates = selector_covariates(args.e2_input_root)
    keys = ['benchmark', 'item_id', 'selector_condition']
    table = debt.merge(covariates, on=keys, validate='many_to_one')
    if table.frame_count.nunique() != 1:
        raise ValueError('frame count is expected to be fixed; add it only if it varies')
    continuous = ['interaction_debt', 'evidence_recall', 'summed_siglip_relevance', 'temporal_coverage']
    for column in continuous:
        scale = table[column].std(ddof=0)
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError(f'{column} has no finite variation')
        table[f'z_{column}'] = (table[column] - table[column].mean()) / scale
    coefficients = fit_clustered_logistic(table, continuous)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'controlled_regression_input.parquet', index=False)
    coefficients.to_csv(args.output_dir / 'controlled_logistic_coefficients.csv', index=False)
    (args.output_dir / 'controlled_logistic_summary.txt').write_text(
        'Unpenalized logistic regression; source-video cluster sandwich covariance.\n' +
        coefficients.to_string(index=False) + '\n')
    debt_row = coefficients[coefficients.term == 'z_interaction_debt'].iloc[0]
    print(json.dumps({'sets': len(table), 'source_video_clusters': int(table.source_video_id.nunique()),
                      'debt_coefficient_per_sd': float(debt_row.coefficient),
                      'debt_p_value_clustered': float(debt_row.p_value)}))


if __name__ == '__main__':
    main()
