#!/usr/bin/env python3
"""Render the primary causal contrasts as a paper-ready heatmap."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


MODELS = ['qwen3_vl_4b', 'qwen3_vl_32b', 'internvl35_8b', 'internvl35_38b',
          'perception_lm_8b', 'videollama3_7b']
BENCHMARKS = ['lvbench', 'nextgqa', 'herbench']
LABELS = {
    'qwen3_vl_4b': 'Qwen3-VL 4B', 'qwen3_vl_32b': 'Qwen3-VL 32B',
    'internvl35_8b': 'InternVL3.5 8B', 'internvl35_38b': 'InternVL3.5 38B',
    'perception_lm_8b': 'Perception-LM 8B', 'videollama3_7b': 'VideoLLaMA3 7B',
    'lvbench': 'LVBench', 'nextgqa': 'NExT-GQA', 'herbench': 'HERBench',
}


def panel(ax, table, title, cmap, norm):
    value = table.pivot(index='model', columns='benchmark', values='estimate').reindex(
        index=MODELS, columns=BENCHMARKS)
    pval = table.pivot(index='model', columns='benchmark', values='p_bh_within_experiment').reindex(
        index=MODELS, columns=BENCHMARKS)
    color_map = plt.get_cmap(cmap)
    image = ax.imshow(value.to_numpy(dtype=float), cmap=color_map, norm=norm, aspect='auto')
    ax.set_title(title, fontsize=13, fontweight='semibold', pad=12)
    ax.set_xticks(range(len(BENCHMARKS)), [LABELS[key] for key in BENCHMARKS], rotation=20, ha='right')
    ax.set_yticks(range(len(MODELS)), [LABELS[key] for key in MODELS])
    ax.tick_params(axis='both', labelsize=10)
    for row in range(len(MODELS)):
        for col in range(len(BENCHMARKS)):
            x = value.iloc[row, col]
            if pd.isna(x):
                ax.text(col, row, 'not run', ha='center', va='center', color='#555555', fontsize=8)
                continue
            star = '*' if pval.iloc[row, col] < .05 else ''
            # Choose annotation color from the actual rendered cell luminance.
            # The previous distance-from-midpoint heuristic incorrectly used
            # white on near-white low-valued cells.
            red, green, blue, _ = color_map(norm(x))
            luminance = .2126 * red + .7152 * green + .0722 * blue
            color = '#FFFFFF' if luminance < .48 else '#111111'
            ax.text(col, row, f'{x:.2f}{star}', ha='center', va='center', color=color,
                    fontsize=10, fontweight='semibold')
    ax.set_xticks(np.arange(-.5, len(BENCHMARKS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    e3 = data[(data.experiment == 'e3') & (data.contrast == 'evidence_minus_outside_utility')]
    e4 = data[(data.experiment == 'e4') &
              (data.contrast == 'evidence_minus_outside_absolute_interaction')]
    if len(e3) != 18 or len(e4) != 18:
        raise ValueError(f'expected 18 E3 and 18 E4 complete contrast cells, got {len(e3)} and {len(e4)}')
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    e3_image = panel(axes[0], e3,
                     'Individual evidence utility (17/18 settings ↑)\nEvidence minus matched outside frames',
                     'RdBu_r', TwoSlopeNorm(vmin=-.1, vcenter=0, vmax=2.8))
    e4_image = panel(axes[1], e4,
                     'Pairwise non-additivity (18/18 settings ↑)\nEvidence pairs minus outside-frame pairs',
                     'YlGnBu', plt.Normalize(vmin=0, vmax=2.4))
    fig.colorbar(e3_image, ax=axes[0], shrink=.84, label='Utility contrast (log odds)')
    fig.colorbar(e4_image, ax=axes[1], shrink=.84, label='Absolute-interaction contrast')
    fig.text(.5, .005, '* BH-adjusted paired source-video sign-flip p < .05',
             ha='center', fontsize=9)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=240, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
