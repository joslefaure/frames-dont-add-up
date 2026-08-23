#!/usr/bin/env python3
"""Render same-versus-distinct annotated-interval interaction contrasts."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {'qwen3_vl_4b': '#4E79A7', 'qwen3_vl_32b': '#1F4E79',
          'internvl35_8b': '#59A14F', 'internvl35_38b': '#2F6B2F',
          'perception_lm_8b': '#E15759', 'videollama3_7b': '#9C755F'}
LABELS = {'qwen3_vl_4b': 'Qwen 4B', 'qwen3_vl_32b': 'Qwen 32B',
          'internvl35_8b': 'InternVL 8B', 'internvl35_38b': 'InternVL 38B',
          'perception_lm_8b': 'PLM 8B', 'videollama3_7b': 'VideoLLaMA3'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.input_root.glob('*/topology_summary.csv'))
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    table = table[table.topology.isin(['evidence_evidence_same_interval',
                                       'evidence_evidence_distinct_interval'])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), sharey=True, constrained_layout=True)
    for ax, benchmark, title in zip(axes, ['nextgqa', 'herbench'], ['NExT-GQA', 'HERBench']):
        cell = table[table.benchmark == benchmark]
        for position, (_, group) in enumerate(cell.groupby('model', sort=True)):
            group = group.set_index('topology')
            if len(group) != 2:
                continue
            jitter = (position - 2.5) * .035
            values = [group.loc['evidence_evidence_same_interval', 'mean_interaction'],
                      group.loc['evidence_evidence_distinct_interval', 'mean_interaction']]
            color = COLORS[group.model.iloc[0]]
            ax.plot(np.array([0, 1]) + jitter, values, marker='o', color=color, lw=1.6,
                    label=LABELS[group.model.iloc[0]])
        ax.axhline(0, color='#777777', lw=.8)
        ax.set_xticks([0, 1], ['Same\ninterval', 'Distinct\ninterval'])
        ax.set_title(title, fontsize=11, fontweight='semibold')
        ax.grid(axis='y', alpha=.2)
        # Label the zero line with direction cues at print size
        ax.text(1.02, 0.02, 'complementarity ▲', fontsize=6.5, color='#666666',
                ha='left', va='bottom', transform=ax.get_yaxis_transform())
        ax.text(1.02, -0.02, 'saturation / interference ▼', fontsize=6.5, color='#666666',
                ha='left', va='top', transform=ax.get_yaxis_transform())
    axes[0].set_ylabel('Mean signed pair interaction (log odds)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, -.16), frameon=False)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=220, bbox_inches='tight')
    print(f'Wrote {args.output_prefix.with_suffix(".pdf")} and .png')


if __name__ == '__main__':
    main()
