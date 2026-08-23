#!/usr/bin/env python3
"""Test whether evidence-pair interaction exceeds utility-matched controls."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def clustered_ols(table, outcome, predictors, cluster):
    clean = table[[outcome, cluster] + predictors].dropna().copy()
    x_raw = clean[predictors].to_numpy(float)
    means = x_raw.mean(axis=0)
    scales = x_raw.std(axis=0)
    scales[scales == 0] = 1.0
    x = np.column_stack([np.ones(len(clean)), (x_raw - means) / scales])
    y = clean[outcome].to_numpy(float)
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    groups = clean[cluster].astype(str).to_numpy()
    unique = np.unique(groups)
    for group in unique:
        score = x[groups == group].T @ residual[groups == group]
        meat += np.outer(score, score)
    n, p, g = len(clean), x.shape[1], len(unique)
    correction = (g / (g - 1)) * ((n - 1) / (n - p)) if g > 1 and n > p else 1.0
    covariance = correction * bread @ meat @ bread
    se = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    t_value = beta / se
    p_value = 2 * stats.t.sf(np.abs(t_value), df=max(1, g - 1))
    focal_scale = scales[0]
    focal_coefficient = beta[1] / focal_scale
    focal_se = se[1] / focal_scale
    critical = stats.t.ppf(.975, max(1, g - 1))
    return {
        'records': n,
        'source_video_clusters': g,
        'coefficient': focal_coefficient,
        'cluster_se': focal_se,
        'ci_low': focal_coefficient - critical * focal_se,
        'ci_high': focal_coefficient + critical * focal_se,
        'p_two_sided': p_value[1],
        'predictor_scale': scales[0],
    }


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(1.0, adjusted)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-root', required=True, type=Path)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    adjusted_rows, record_tables = [], []
    for path in sorted(args.analysis_root.glob('*/per_pair_context.parquet')):
        pairs = pd.read_parquet(path)
        pairs = pairs[pairs.pair_type.isin(['evidence_evidence', 'outside_outside'])].copy()
        model, benchmark = pairs.model.iloc[0], pairs.benchmark.iloc[0]
        design = pd.read_parquet(args.input_root / benchmark / 'e4' / 'all_conditions.parquet')
        source = design.drop_duplicates(['item_id', 'context_index', 'pair_type'])[
            ['item_id', 'context_index', 'pair_type', 'source_video_id',
             'candidate_a_time', 'candidate_b_time']]
        pairs = pairs.merge(source, on=['item_id', 'context_index', 'pair_type'], validate='one_to_one')
        pairs['evidence_pair'] = (pairs.pair_type == 'evidence_evidence').astype(float)
        pairs['utility_a'] = pairs.a - pairs.baseline
        pairs['utility_b'] = pairs.b - pairs.baseline
        pairs['main_abs_sum'] = pairs.utility_a.abs() + pairs.utility_b.abs()
        pairs['main_abs_max'] = pairs[['utility_a', 'utility_b']].abs().max(axis=1)
        pairs['main_abs_difference'] = (pairs.utility_a.abs() - pairs.utility_b.abs()).abs()
        pairs['main_signed_sum'] = pairs.utility_a + pairs.utility_b
        pairs['baseline_magnitude'] = pairs.baseline.abs()
        pairs['temporal_distance'] = (pairs.candidate_a_time - pairs.candidate_b_time).abs()
        pairs['main_abs_sum_squared'] = pairs.main_abs_sum ** 2
        outside = pairs[pairs.evidence_pair == 0].main_abs_sum
        low, high = outside.quantile([.01, .99])
        pairs['outside_common_support'] = pairs.main_abs_sum.between(low, high)
        specifications = {
            'unadjusted': (['evidence_pair'], pairs),
            'singleton_sum': (['evidence_pair', 'main_abs_sum'], pairs),
            'singleton_rich': ([
                'evidence_pair', 'main_abs_sum', 'main_abs_max',
                'main_abs_difference', 'main_signed_sum', 'main_abs_sum_squared'], pairs),
            'full': ([
                'evidence_pair', 'main_abs_sum', 'main_abs_max', 'main_abs_difference',
                'main_signed_sum', 'baseline_magnitude', 'temporal_distance',
                'main_abs_sum_squared'], pairs),
            'full_common_support': ([
                'evidence_pair', 'main_abs_sum', 'main_abs_max', 'main_abs_difference',
                'main_signed_sum', 'baseline_magnitude', 'temporal_distance',
                'main_abs_sum_squared'], pairs[pairs.outside_common_support]),
        }
        for specification, (predictors, analysis_table) in specifications.items():
            result = clustered_ols(
                analysis_table, 'absolute_interaction', predictors, 'source_video_id')
            adjusted_rows.append({
                'model': model,
                'benchmark': benchmark,
                'specification': specification,
                'estimand': 'evidence_pair_coefficient_adjusted_for_singleton_scale',
                **result,
                'raw_gap': (
                    pairs.loc[pairs.evidence_pair == 1, 'absolute_interaction'].mean()
                    - pairs.loc[pairs.evidence_pair == 0, 'absolute_interaction'].mean()
                ),
                'main_abs_sum_gap': (
                    pairs.loc[pairs.evidence_pair == 1, 'main_abs_sum'].mean()
                    - pairs.loc[pairs.evidence_pair == 0, 'main_abs_sum'].mean()
                ),
                'outside_support_low': low,
                'outside_support_high': high,
            })
        record_tables.append(pairs)
    records = pd.concat(record_tables, ignore_index=True)
    adjusted = pd.DataFrame(adjusted_rows).sort_values(['model', 'benchmark'])
    adjusted['p_bh_within_specification'] = adjusted.groupby(
        'specification', group_keys=False).p_two_sided.transform(bh)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(args.output_dir / 'per_pair_adjustment_variables.parquet', index=False)
    adjusted.to_csv(args.output_dir / 'adjusted_setting_results.csv', index=False)
    print(adjusted.to_string(index=False))


if __name__ == '__main__':
    main()
