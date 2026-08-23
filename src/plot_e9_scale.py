#!/usr/bin/env python3
"""Render paired model-scale comparisons from the completed E9 table."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


METRICS = [
    ('evidence_minus_outside_utility', 'Evidence − outside utility'),
    ('evidence_pair_absolute_interaction', 'Evidence-pair\nabsolute interaction'),
    ('selector_accuracy', 'Best selector accuracy'),
]
COLORS = {'internvl': '#F58518', 'qwen': '#4C78A8'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.input).sort_values(['family', 'benchmark'])
    if set(data.family) != {'internvl', 'qwen'} or len(data) != 6:
        raise ValueError('E9 expects the frozen Qwen and InternVL three-benchmark scale pairs')
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.5, 4.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes, METRICS):
        for row_index, (_, row) in enumerate(data.iterrows()):
            small, large = row[f'{metric}_small'], row[f'{metric}_large']
            y = row_index
            color = COLORS[row.family]
            axis.plot([small, large], [y, y], color=color, linewidth=2)
            axis.scatter([small], [y], color=color, marker='o', s=52, zorder=3)
            axis.scatter([large], [y], color=color, marker='s', s=52, zorder=3)
        axis.axvline(0 if metric != 'selector_accuracy' else .25, color='#888888', linewidth=.8, zorder=0)
        axis.set_yticks(range(len(data)), [f'{row.family}: {row.benchmark}' for _, row in data.iterrows()], fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=10)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Larger models do not uniformly reduce contextual interaction', fontsize=13)
    fig.legend(handles=[Line2D([0], [0], marker='o', color='#444444', label='smaller model', linewidth=0),
                        Line2D([0], [0], marker='s', color='#444444', label='larger model', linewidth=0),
                        Line2D([0], [0], color=COLORS['qwen'], label='Qwen family'),
                        Line2D([0], [0], color=COLORS['internvl'], label='InternVL family')],
               loc='lower center', ncols=4, bbox_to_anchor=(.5, -.09))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
