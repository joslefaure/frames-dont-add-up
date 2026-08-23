#!/usr/bin/env python3
"""Render E8 topology-contrast robustness controls from frozen summaries."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ORDER = ['reversed', 'slot_swap', 'extra_context', 'hard_distractor', 'k16', 'direct_rgb_png']
LABELS = {'reversed': 'Reversed order', 'slot_swap': 'Candidate slot swap',
          'extra_context': 'Two new contexts', 'hard_distractor': 'Hard distractors',
          'k16': 'K = 16', 'direct_rgb_png': 'Direct RGB PNG'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    # The combined summary also carries a boundary-offset contrast without a
    # robustness-condition label; it is reported elsewhere and is not one of
    # the six controls in this panel.
    table = table[table.robustness_condition.isin(ORDER)].copy()
    value = 'evidence_minus_outside_absolute_interaction'
    if set(table.robustness_condition) != set(ORDER):
        raise ValueError('E8 controls are incomplete or contain an unexpected condition')
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    positions, values = [], []
    for index, condition in enumerate(ORDER, 1):
        group = table[table.robustness_condition == condition][value].to_numpy()
        if len(group) != 9 or not (group > 0).all():
            raise ValueError(f'{condition}: expected nine positive frozen contrasts')
        positions.append(index)
        values.append(group)
    violin = ax.violinplot(values, positions=positions, widths=.72, showmeans=False, showmedians=True)
    for body in violin['bodies']:
        body.set_facecolor('#4C78A8')
        body.set_alpha(.45)
    for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
        violin[key].set_color('#1F4E79')
    for index, group in zip(positions, values):
        jitter = ((pd.Series(range(len(group))) % 3) - 1) * .06
        ax.scatter(index + jitter, group, color='#1F4E79', s=23, zorder=3)
    ax.axhline(0, color='#555555', linewidth=1)
    ax.set_xticks(positions, [LABELS[key] for key in ORDER], rotation=22, ha='right')
    ax.set_ylabel('Evidence/evidence − outside/outside\nabsolute interaction')
    ax.set_title('Non-additivity persists across context, budget, and codec controls')
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(.01, .02, 'Each point: one representative model × benchmark cell (n = 9/control).',
            transform=ax.transAxes, fontsize=8)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
