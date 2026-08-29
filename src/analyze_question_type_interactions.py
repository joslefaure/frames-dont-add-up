#!/usr/bin/env python3
"""Stratify the frozen E4 factorial audit by benchmark-provided question type.

The analysis specification is frozen in STANFORD_REVIEW_REVISION_PLAN.md.  All
uncertainty resamples source-video aggregates, never individual questions or
pair-context rows.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_TYPES = ("evidence_evidence", "outside_outside")
NEXT_COARSE = {
    "CW": "causal",
    "CH": "causal",
    "TN": "temporal",
    "TC": "temporal",
    "TP": "temporal",
}
LV_TAGS = (
    "temporal grounding",
    "reasoning",
    "event understanding",
    "entity recognition",
    "key information retrieval",
    "summarization",
)


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:16], 16) % (2**32)


def load_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    keep = ["item_id", "source_video_id", "question_type"]
    return pd.DataFrame(rows)[keep]


def item_strata(manifest, benchmark):
    rows = []
    for row in manifest.itertuples(index=False):
        qtype = str(row.question_type)
        if benchmark == "nextgqa":
            rows.append((row.item_id, row.source_video_id, "native", qtype))
            if qtype not in NEXT_COARSE:
                raise ValueError(f"unmapped NExT-GQA question type: {qtype}")
            rows.append((row.item_id, row.source_video_id, "coarse", NEXT_COARSE[qtype]))
        elif benchmark == "herbench":
            rows.append((row.item_id, row.source_video_id, "native", qtype))
        elif benchmark == "lvbench":
            tags = {part.strip() for part in qtype.split("|")}
            unknown = tags.difference(LV_TAGS)
            if unknown:
                raise ValueError(f"unmapped LVBench tags for {row.item_id}: {sorted(unknown)}")
            for tag in LV_TAGS:
                if tag in tags:
                    rows.append((row.item_id, row.source_video_id, "tag_present", tag))
        else:
            raise ValueError(benchmark)
    return pd.DataFrame(rows, columns=["item_id", "source_video_id", "scheme", "stratum"])


def bootstrap_ci(values, draws, seed):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    # Chunk draws to bound memory even if a future manifest has many clusters.
    means = []
    remaining = draws
    while remaining:
        take = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(take, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= take
    samples = np.concatenate(means)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def summarize_setting(table, model, benchmark, scheme, stratum, draws):
    # First average repeated pairs/contexts and questions within each source video.
    source_pair = (table.groupby(["source_video_id", "pair_type"], as_index=False)
                        .agg(abs_interaction=("absolute_interaction", "mean"),
                             signed_interaction=("interaction", "mean"),
                             negative_rate=("interaction", lambda x: float((x < 0).mean()))))
    abs_wide = source_pair.pivot(index="source_video_id", columns="pair_type",
                                 values="abs_interaction").dropna(subset=list(PAIR_TYPES))
    gap = abs_wide["evidence_evidence"] - abs_wide["outside_outside"]
    ee = source_pair[source_pair.pair_type == "evidence_evidence"].set_index("source_video_id")
    shared = gap.index.intersection(ee.index)
    gap = gap.loc[shared]
    signed = ee.loc[shared, "signed_interaction"]
    negative = ee.loc[shared, "negative_rate"]
    base = (model, benchmark, scheme, stratum)
    gap_ci = bootstrap_ci(gap.values, draws, stable_seed(*base, "gap"))
    signed_ci = bootstrap_ci(signed.values, draws, stable_seed(*base, "signed"))
    negative_ci = bootstrap_ci(negative.values, draws, stable_seed(*base, "negative"))
    return {
        "model": model,
        "benchmark": benchmark,
        "scheme": scheme,
        "stratum": stratum,
        "questions": int(table.item_id.nunique()),
        "source_videos": int(len(shared)),
        "pair_context_records": int(len(table)),
        "absolute_gap": float(gap.mean()),
        "absolute_gap_ci_low": float(gap_ci[0]),
        "absolute_gap_ci_high": float(gap_ci[1]),
        "signed_evidence_interaction": float(signed.mean()),
        "signed_evidence_ci_low": float(signed_ci[0]),
        "signed_evidence_ci_high": float(signed_ci[1]),
        "evidence_negative_rate": float(negative.mean()),
        "evidence_negative_rate_ci_low": float(negative_ci[0]),
        "evidence_negative_rate_ci_high": float(negative_ci[1]),
    }


def nextgqa_coarse_contrast(table, model, draws):
    source_pair = (table.groupby(["source_video_id", "stratum", "pair_type"], as_index=False)
                   .absolute_interaction.mean())
    wide = source_pair.pivot(index=["source_video_id", "stratum"], columns="pair_type",
                             values="absolute_interaction").dropna(subset=list(PAIR_TYPES))
    wide["gap"] = wide["evidence_evidence"] - wide["outside_outside"]
    source_gap = wide["gap"].unstack("stratum")
    if set(source_gap.columns) != {"causal", "temporal"}:
        raise ValueError(f"unexpected NExT-GQA coarse strata: {list(source_gap.columns)}")
    estimate = float(source_gap.temporal.mean() - source_gap.causal.mean())
    values = source_gap[["causal", "temporal"]].to_numpy(float)
    rng = np.random.default_rng(stable_seed(model, "nextgqa", "temporal_minus_causal"))
    samples = []
    remaining = draws
    while remaining:
        take = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(take, len(values)))
        sampled = values[indices]
        with np.errstate(invalid="ignore"):
            means = np.nanmean(sampled, axis=1)
        samples.append(means[:, 1] - means[:, 0])
        remaining -= take
    samples = np.concatenate(samples)
    ci = np.quantile(samples, [0.025, 0.975])
    return {
        "model": model,
        "benchmark": "nextgqa",
        "contrast": "temporal_minus_causal_absolute_gap",
        "union_source_videos": int(len(source_gap)),
        "causal_source_videos": int(source_gap.causal.notna().sum()),
        "temporal_source_videos": int(source_gap.temporal.notna().sum()),
        "shared_source_videos": int(source_gap.dropna().shape[0]),
        "estimate": estimate,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/causal_analysis/e4"))
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests/v12/causal_holdout"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_question_type_analysis"))
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_tables = {}
    stratum_tables = {}
    for benchmark in ("herbench", "lvbench", "nextgqa"):
        manifest = load_manifest(args.manifest_root / f"{benchmark}.jsonl")
        manifest_tables[benchmark] = manifest
        stratum_tables[benchmark] = item_strata(manifest, benchmark)
    pd.concat([table.assign(benchmark=benchmark) for benchmark, table in stratum_tables.items()],
              ignore_index=True).to_csv(args.output_dir / "item_strata.csv", index=False)

    rows = []
    contrasts = []
    paths = sorted(args.analysis_root.glob("*/per_pair_context.parquet"))
    if len(paths) != 18:
        raise ValueError(f"expected 18 E4 setting tables, found {len(paths)}")
    for path in paths:
        table = pd.read_parquet(path)
        model = str(table.model.iloc[0])
        benchmark = str(table.benchmark.iloc[0])
        table = table[table.pair_type.isin(PAIR_TYPES)].merge(
            stratum_tables[benchmark], on="item_id", how="inner", validate="many_to_many")
        for (scheme, stratum), group in table.groupby(["scheme", "stratum"], sort=True):
            rows.append(summarize_setting(group, model, benchmark, scheme, stratum, args.draws))
        if benchmark == "nextgqa":
            contrasts.append(nextgqa_coarse_contrast(table[table.scheme == "coarse"], model,
                                                      args.draws))

    settings = pd.DataFrame(rows).sort_values(["benchmark", "scheme", "stratum", "model"])
    settings.to_csv(args.output_dir / "setting_strata.csv", index=False)
    macro = (settings.groupby(["benchmark", "scheme", "stratum"], as_index=False)
             .agg(settings=("model", "nunique"),
                  min_source_videos=("source_videos", "min"),
                  median_source_videos=("source_videos", "median"),
                  median_absolute_gap=("absolute_gap", "median"),
                  min_absolute_gap=("absolute_gap", "min"),
                  max_absolute_gap=("absolute_gap", "max"),
                  positive_settings=("absolute_gap", lambda x: int((x > 0).sum())),
                  ci_positive_settings=("absolute_gap_ci_low", lambda x: int((x > 0).sum())),
                  median_signed_evidence=("signed_evidence_interaction", "median"),
                  negative_signed_settings=("signed_evidence_interaction", lambda x: int((x < 0).sum())),
                  median_negative_rate=("evidence_negative_rate", "median")))
    macro.to_csv(args.output_dir / "macro_summary.csv", index=False)
    contrast_table = pd.DataFrame(contrasts).sort_values("model")
    contrast_table.to_csv(args.output_dir / "planned_contrasts.csv", index=False)
    provenance = {
        "schema_version": 1,
        "draws": args.draws,
        "resampling_unit": "source_video_id",
        "primary_estimand": "source-video-equal mean absolute interaction: evidence_evidence minus outside_outside",
        "nextgqa_coarse_mapping": NEXT_COARSE,
        "lvbench_tag_present_strata": LV_TAGS,
        "setting_tables": len(paths),
        "setting_stratum_rows": len(settings),
        "planned_contrasts": len(contrast_table),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({**provenance, "macro_rows": len(macro)}, indent=2))


if __name__ == "__main__":
    main()
