#!/usr/bin/env python3
"""Render the preregistered E3 proxy-to-utility alignment figure."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ['siglip_relevance', 'dino_local_change', 'temporal_centrality',
         'visual_spectral_saliency', 'annotated_membership']
LABELS = ['SigLIP relevance', 'DINO local change', 'Temporal centrality',
          'Visual saliency', 'Annotation membership']
COLORS = ['#24577A', '#4D8C57', '#9A6A36', '#815A9B', '#B34444']


def draw(ax, table, column, xlabel):
    rng = np.random.default_rng(20260731)
    for index, proxy in enumerate(ORDER):
        values = table.loc[table.proxy == proxy, column].to_numpy()
        jitter = rng.uniform(-.13, .13, size=len(values))
        ax.scatter(values, np.full(len(values), index) + jitter, color=COLORS[index], s=25,
                   alpha=.75, linewidths=0, zorder=2)
        median = np.median(values)
        ax.plot([median, median], [index - .25, index + .25], color='#202020', lw=2.5, zorder=3)
    ax.axvline(0, color='#777777', lw=.8, zorder=1)
    ax.set_yticks(range(len(ORDER)), LABELS)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    if set(ORDER) != set(table.proxy.unique()):
        raise ValueError('input does not contain the complete fixed proxy suite')
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True, constrained_layout=True)
    draw(axes[0], table, 'spearman_pooled', 'Pooled Spearman with measured utility')
    draw(axes[1], table, 'mean_pointwise_regret', 'Selection regret vs. utility oracle (log odds)')
    axes[0].set_title('Rank alignment', fontsize=11, fontweight='semibold')
    axes[1].set_title('Selection regret', fontsize=11, fontweight='semibold')
    axes[1].axvline(0, color='#cccccc', lw=.8, ls='--', zorder=0)
    axes[1].tick_params(labelleft=False)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=220, bbox_inches='tight')
    print(f'Wrote {args.output_prefix.with_suffix(".pdf")} and .png')


if __name__ == '__main__':
    main()
