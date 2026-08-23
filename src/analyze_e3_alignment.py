#!/usr/bin/env python3
"""Compute frozen proxy-to-measured-utility alignment and pointwise regret."""
import argparse
from pathlib import Path

import pandas as pd


def rank_correlation(group, proxy):
    return group.utility_mean.corr(group[proxy], method='spearman')


def item_metrics(group, proxy, k):
    observed = group.sort_values(proxy, ascending=False, kind='mergesort').head(k)
    oracle = group.sort_values('utility_mean', ascending=False, kind='mergesort').head(k)
    recall = len(set(observed.candidate_index) & set(oracle.candidate_index)) / k
    regret = oracle.utility_mean.mean() - observed.utility_mean.mean()
    return pd.Series({'utility_recall_at_k': recall, 'pointwise_regret': regret})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--utilities', required=True, type=Path,
                        help='aggregate_causal.py E3 per_candidate.parquet')
    parser.add_argument('--proxies', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--k', type=int, default=2)
    args = parser.parse_args()
    utilities, proxies = pd.read_parquet(args.utilities), pd.read_parquet(args.proxies)
    keys = ['item_id', 'candidate_index']
    table = utilities.merge(proxies, on=keys, validate='one_to_one', suffixes=('', '_proxy'))
    if 'benchmark_proxy' in table and not (table.benchmark == table.benchmark_proxy).all():
        raise ValueError('proxy benchmark mismatch')
    proxy_columns = ['siglip_relevance', 'dino_local_change', 'temporal_centrality',
                     'visual_spectral_saliency', 'annotated_membership']
    missing_proxies = set(proxy_columns) - set(table)
    if missing_proxies:
        raise ValueError(f'proxy table lacks {sorted(missing_proxies)}')
    rows, per_item = [], []
    for (model, benchmark), cell in table.groupby(['model', 'benchmark'], sort=False):
        for proxy in proxy_columns:
            correlation = rank_correlation(cell, proxy)
            item = cell.groupby('item_id', group_keys=False).apply(item_metrics, proxy=proxy, k=args.k).reset_index()
            item['model'], item['benchmark'], item['proxy'] = model, benchmark, proxy
            item['spearman_pooled'] = correlation
            per_item.append(item)
            rows.append({'model': model, 'benchmark': benchmark, 'proxy': proxy,
                         'candidates': len(cell), 'items': item.item_id.nunique(),
                         'spearman_pooled': correlation,
                         'mean_utility_recall_at_k': item.utility_recall_at_k.mean(),
                         'mean_pointwise_regret': item.pointwise_regret.mean()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / 'utility_proxy_join.parquet', index=False)
    pd.concat(per_item, ignore_index=True).to_parquet(args.output_dir / 'per_item_selector_metrics.parquet', index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / 'alignment_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
