#!/usr/bin/env python3
"""Render E5's fixed-budget functional-ANOVA decomposition."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODELS = ['qwen3_vl_32b', 'internvl35_38b', 'perception_lm_8b']
BENCHMARKS = ['lvbench', 'nextgqa', 'herbench']
MODEL_LABELS = {'qwen3_vl_32b': 'Qwen3-VL 32B', 'internvl35_38b': 'InternVL3.5 38B',
                'perception_lm_8b': 'Perception-LM 8B'}
BENCH_LABELS = {'lvbench': 'LVBench', 'nextgqa': 'NExT-GQA', 'herbench': 'HERBench'}
COLORS = {'1': '#4C78A8', '2': '#F58518', '3': '#54A24B', '4': '#B279A2'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    table['order'] = table.order.astype(str)
    required = set(MODELS) | set()
    if set(table.model) != required or set(table.benchmark) != set(BENCHMARKS):
        raise ValueError('E5 input does not contain the frozen 3 × 3 model/benchmark matrix')
    fig, ax = plt.subplots(figsize=(10.6, 4.6), constrained_layout=True)
    labels, positions, high_order_labels = [], [], []
    for model_index, model in enumerate(MODELS):
        for bench_index, benchmark in enumerate(BENCHMARKS):
            position = model_index * 4 + bench_index
            positions.append(position)
            labels.append(f'{MODEL_LABELS[model]}\n{BENCH_LABELS[benchmark]}')
            cell = table[(table.model == model) & (table.benchmark == benchmark)].set_index('order')
            bottom = 0.
            for order in ('1', '2', '3', '4'):
                value = float(cell.loc[order, 'fraction_absolute_component'])
                ax.bar(position, value, bottom=bottom, width=.76, color=COLORS[order],
                       label=f'order {order}' if position == 0 else None)
                bottom += value
            high = cell.loc['3_plus_4']
            high_order_labels.append((position, float(high.fraction_absolute_component),
                                      float(high.ci_low), float(high.ci_high)))
    for position, estimate, low, high in high_order_labels:
        ax.text(position, 1.025, f'3+4: {estimate:.2f}\n[{low:.2f}, {high:.2f}]',
                ha='center', va='bottom', fontsize=7)
    for model_index, model in enumerate(MODELS):
        if model_index > 0:
            sep_x = model_index * 4 - 0.6
            ax.axvline(sep_x, color='#cccccc', lw=.7, ls='--', zorder=0)
    ax.set_xticks(positions, labels, rotation=21, ha='right', fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Fraction of absolute nonconstant set components')
    ax.set_title('Higher-order interactions remain under a fixed frame budget',
                 fontsize=11, fontweight='semibold')
    ax.legend(title='Möbius order', ncols=2, loc='lower right',
              fontsize=8, title_fontsize=8, framealpha=.92, edgecolor='#cccccc')
    ax.spines[['top', 'right']].set_visible(False)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
