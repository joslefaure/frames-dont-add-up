#!/usr/bin/env python3
"""Generate descriptive, paper-ready tables from completed causal holdout data.

These tables deliberately never mix the causal holdout with the all-eligible
census. Means are equally weighted over completed model--benchmark cells.
"""
import argparse
from pathlib import Path

import pandas as pd


CONDITION_NAMES = {
    'uniform8': 'Uniform', 'random8': 'Seeded random',
    'siglip_relevance8': 'SigLIP relevance', 'dino_change8': 'DINO change',
    'dino_diversity8': 'DINO diversity',
    'siglip_dino_submodular8': 'Relevance-diversity',
    'aks_fixedk8': 'AKS (alias of relevance)', 'bolt_its8': 'BOLT-ITS',
    'mdp3_8': 'MDP3', 'plain_dpp_8': 'Plain DPP',
}


def write_table(frame: pd.DataFrame, output: Path, title: str):
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix('.csv'), index=False)
    markdown = f'## {title}\n\n' + frame.to_markdown(index=False) + '\n'
    output.with_suffix('.md').write_text(markdown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e2-root', type=Path, default=Path('artifacts/e2_causal_analysis'))
    parser.add_argument('--selector-extension-root', type=Path,
                        default=Path('artifacts/mdp3_dpp_question_only_analysis'))
    parser.add_argument('--e3-summary', type=Path, default=Path('artifacts/e3_alignment/all_summary.csv'))
    parser.add_argument('--e4-root', type=Path, default=Path('artifacts/e4_topology_analysis'))
    parser.add_argument('--output-dir', type=Path, default=Path('reproduced/tables'))
    args = parser.parse_args()

    e2 = pd.read_csv(args.e2_root / 'all_summary.csv')
    extension_paths = sorted(args.selector_extension_root.glob('*/summary.csv'))
    if extension_paths:
        e2 = pd.concat([e2, *(pd.read_csv(path) for path in extension_paths)], ignore_index=True)
    selector = (e2.groupby('condition', as_index=False)
                  .agg(completed_cells=('model', 'size'), accuracy=('accuracy', 'mean'),
                       log_odds=('mean_logodds', 'mean'), evidence_recall=('evidence_recall_at_k', 'mean'),
                       temporal_coverage=('temporal_coverage', 'mean'), seconds=('mean_seconds', 'mean')))
    selector.insert(0, 'selector', selector.condition.map(CONDITION_NAMES).fillna(selector.condition))
    selector = selector.drop(columns='condition').sort_values('log_odds', ascending=False)
    selector[['accuracy', 'log_odds', 'evidence_recall', 'temporal_coverage', 'seconds']] = selector[
        ['accuracy', 'log_odds', 'evidence_recall', 'temporal_coverage', 'seconds']].round(3)
    write_table(selector, args.output_dir / 'causal_selector_audit',
                'Causal holdout selector audit (equal-cell macro averages)')

    e3 = pd.read_csv(args.e3_summary)
    proxy = (e3.groupby('proxy', as_index=False)
               .agg(completed_cells=('model', 'size'), items=('items', 'sum'),
                    pooled_spearman=('spearman_pooled', 'median'),
                    utility_recall_at_k=('mean_utility_recall_at_k', 'mean'),
                    pointwise_regret=('mean_pointwise_regret', 'mean'))
               .sort_values('pointwise_regret', ascending=False))
    proxy[['pooled_spearman', 'utility_recall_at_k', 'pointwise_regret']] = proxy[
        ['pooled_spearman', 'utility_recall_at_k', 'pointwise_regret']].round(3)
    write_table(proxy, args.output_dir / 'proxy_utility_alignment',
                'Causal pointwise utility alignment (LVBench and HERBench only)')

    paths = sorted(args.e4_root.glob('*/topology_summary.csv'))
    topology = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    topology = (topology.groupby('topology', as_index=False)
                .agg(completed_cells=('model', 'size'), pair_context_cells=('cells', 'sum'),
                     signed_interaction=('mean_interaction', 'mean'),
                     absolute_interaction=('mean_absolute_interaction', 'mean'),
                     negative_rate=('negative_rate', 'mean'))
                .sort_values('absolute_interaction', ascending=False))
    topology[['signed_interaction', 'absolute_interaction', 'negative_rate']] = topology[
        ['signed_interaction', 'absolute_interaction', 'negative_rate']].round(3)
    write_table(topology, args.output_dir / 'factorial_topology',
                'Pairwise factorial topology audit (equal-cell macro averages)')
    print(f'Wrote three causal-holdout tables to {args.output_dir}')


if __name__ == '__main__':
    main()
