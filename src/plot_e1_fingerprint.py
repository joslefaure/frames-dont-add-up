#!/usr/bin/env python3
"""Render the completed E1 causal sampling fingerprints.

The left panel reports the per-setting accuracy contrasts directly implied by the
factorial interventions.  The right panel keeps setting-level heterogeneity
visible rather than presenting an unqualified pooled effect.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MEASURES = [
    ('oracle_headroom_accuracy_delta', 'Annotated oracle\n− uniform 8'),
    ('evidence_dose_8_vs_1_accuracy_delta', '8 evidence\n− 1 evidence'),
    ('context_interference_accuracy_delta', 'Evidence + context\ninteraction'),
    ('reverse_order_sensitivity_accuracy_delta', 'Reverse-order\nsensitivity'),
    ('shuffle_order_sensitivity_accuracy_delta', 'Shuffle-order\nsensitivity'),
    ('uniform8_vs_prior_accuracy_delta', 'Uniform 8\n− prior'),
    ('coverage_32_vs_4_accuracy_delta', 'Uniform 32\n− uniform 4'),
]


def cell_label(row):
    names = {
        'qwen3_vl_4b': 'Qwen 4B', 'qwen3_vl_32b': 'Qwen 32B',
        'internvl35_8b': 'InternVL 8B', 'internvl35_38b': 'InternVL 38B',
        'perception_lm_8b': 'PLM 8B', 'videollama3_7b': 'VideoLLaMA3',
        'lvbench': 'LVBench', 'nextgqa': 'NExT-GQA', 'herbench': 'HERBench',
    }
    return f"{names.get(row.model, row.model)} · {names.get(row.benchmark, row.benchmark)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()

    paths = sorted(args.input_root.glob('*/fingerprint_summary.csv'))
    if not paths:
        raise FileNotFoundError(f'No fingerprint summaries under {args.input_root}')
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    columns = [column for column, _ in MEASURES]
    table = table.drop_duplicates(['model', 'benchmark']).sort_values(['benchmark', 'model'])
    values = table[columns].to_numpy(float)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12.2, 5.7), gridspec_kw={'width_ratios': [1.05, 1.35]},
        constrained_layout=True,
    )
    means = np.nanmean(values, axis=0)
    # Across-setting spread is descriptive, not a resampling confidence interval.
    lo, hi = np.nanpercentile(values, [25, 75], axis=0)
    positions = np.arange(len(MEASURES))
    ax0.errorbar(means, positions, xerr=np.vstack([means - lo, hi - means]), fmt='o',
                 color='#24577A', ecolor='#24577A', capsize=3, lw=1.6, ms=6)
    ax0.axvline(0, color='#777777', lw=.8)
    ax0.set_yticks(positions, [label for _, label in MEASURES])
    ax0.invert_yaxis()
    ax0.set_xlabel('Accuracy contrast (mean; bar = setting IQR)')
    ax0.set_title('Evidence-access contrasts')
    ax0.grid(axis='x', alpha=.22)

    vmax = max(.05, float(np.nanmax(np.abs(values))))
    image = ax1.imshow(values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.set_xticks(range(len(MEASURES)), ['Oracle', 'Dose', 'Context', 'Reverse', 'Shuffle',
                                           'Prior', 'Coverage'], rotation=38, ha='right')
    ax1.set_yticks(range(len(table)), [cell_label(row) for row in table.itertuples(index=False)])
    ax1.set_title('Evaluation-setting fingerprints (accuracy contrast)')
    for y in np.where(table.benchmark.to_numpy()[1:] != table.benchmark.to_numpy()[:-1])[0] + .5:
        ax1.axhline(y, color='white', lw=2)
    colorbar = fig.colorbar(image, ax=ax1, fraction=.047, pad=.02)
    colorbar.set_label('Accuracy contrast')

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=220, bbox_inches='tight')
    print(f'Wrote {args.output_prefix.with_suffix(".pdf")} and .png from {len(table)} settings')


if __name__ == '__main__':
    main()
