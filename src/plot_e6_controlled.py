#!/usr/bin/env python3
"""Plot controlled E6 selector predictors with clustered uncertainty bars."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TERMS = ['z_interaction_debt', 'z_evidence_recall', 'z_summed_siglip_relevance', 'z_temporal_coverage']
LABELS = ['Interaction debt', 'Evidence recall', 'Summed SigLIP relevance', 'Temporal coverage']
COLORS = ['#B34444', '#24577A', '#4D8C57', '#9A6A36']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.input).set_index('term').loc[TERMS].reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 3.5), constrained_layout=True)
    positions = list(range(len(table)))
    for position, row, color in zip(positions, table.itertuples(index=False), COLORS):
        ax.errorbar(row.coefficient, position, xerr=1.96 * row.std_error_clustered_source_video,
                    fmt='o', ms=7, color=color, capsize=3, lw=1.8)
    ax.axvline(0, color='#777777', lw=.9)
    ax.set_yticks(positions, LABELS)
    ax.invert_yaxis()
    ax.set_xlabel('Controlled logistic coefficient per SD (95% clustered interval)')
    ax.set_title('Does interaction debt predict selector correctness?')
    ax.grid(axis='x', alpha=.2)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=220, bbox_inches='tight')
    print(f'Wrote {args.output_prefix.with_suffix(".pdf")} and .png')


if __name__ == '__main__':
    main()
