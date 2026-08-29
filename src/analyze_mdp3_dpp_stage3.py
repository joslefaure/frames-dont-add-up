#!/usr/bin/env python3
"""Prespecified Stage-3 analysis for MDP3 and matched plain DPP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEW = ("mdp3_8", "plain_dpp_8")
COMPARATORS = ("uniform8", "siglip_relevance8", "siglip_dino_submodular8", "bolt_its8")
GATE_COMPARATORS = ("siglip_dino_submodular8", "bolt_its8")
METRICS = ("gold_vs_rest_logodds", "correct")
NAMES = {
    "mdp3_8": "MDP3", "plain_dpp_8": "Plain DPP", "uniform8": "Uniform",
    "siglip_relevance8": "SigLIP relevance",
    "siglip_dino_submodular8": "Relevance-diversity", "bolt_its8": "BOLT-ITS",
    "random8": "Seeded random", "dino_change8": "DINO change",
    "dino_diversity8": "DINO diversity", "aks_fixedk8": "AKS",
}


def bh(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = np.minimum.accumulate((ranked * len(array) / np.arange(1, len(array) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def infer(values: np.ndarray, draws: int, rng: np.random.Generator):
    observed = float(values.mean())
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    signs = rng.choice((-1.0, 1.0), (draws, len(values)))
    null = (values[None, :] * signs).mean(axis=1)
    p = float((1 + (np.abs(null) >= abs(observed)).sum()) / (draws + 1))
    return observed, float(np.quantile(boot, .025)), float(np.quantile(boot, .975)), p


def load_all(new_analysis_root: Path) -> pd.DataFrame:
    frames = []
    for root in (ROOT / "artifacts/e2_causal_analysis", new_analysis_root):
        for path in sorted(root.glob("*/per_item.parquet")):
            frames.append(pd.read_parquet(path)[
                ["model", "benchmark", "item_id", "condition", *METRICS,
                 "annotated_evidence_recall_at_k", "temporal_coverage"]])
    table = pd.concat(frames, ignore_index=True)
    table = table.drop_duplicates(["model", "benchmark", "item_id", "condition"])
    sources = []
    for benchmark in ("herbench", "lvbench", "nextgqa"):
        refined = ROOT / f"manifests/refined/{benchmark}.jsonl"
        manifest_path = refined if refined.is_file() else ROOT / f"manifests/causal_holdout/{benchmark}.jsonl"
        manifest = pd.read_json(manifest_path, lines=True)
        manifest["source_video_id"] = manifest.source_video_id.astype(str)
        sources.append(manifest.drop_duplicates("item_id")[["benchmark", "item_id", "source_video_id"]])
    table = table.merge(pd.concat(sources), on=["benchmark", "item_id"], validate="many_to_one")
    return table


def macro_bootstrap(video_differences, draws, seed):
    rng = np.random.default_rng(seed)
    observed = float(np.mean([values.mean() for values in video_differences.values()]))
    boot = np.zeros(draws)
    # Resample the same benchmark clusters jointly across models.
    for benchmark in ("herbench", "lvbench", "nextgqa"):
        settings = {key: value for key, value in video_differences.items() if key[1] == benchmark}
        source_ids = sorted(set.intersection(*(set(value.index) for value in settings.values())))
        sampled = rng.integers(0, len(source_ids), (draws, len(source_ids)))
        for values in settings.values():
            array = values.loc[source_ids].to_numpy(float)
            boot += array[sampled].mean(axis=1) / len(video_differences)
    return observed, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def absolute_macro_uncertainty(table, draws, seed):
    """Equal-setting item means with joint benchmark-cluster resampling."""
    rng = np.random.default_rng(seed)
    conditions = sorted(table.condition.unique())
    accum = {(metric, condition): np.zeros(draws) for metric in METRICS for condition in conditions}
    rows = []
    for benchmark in ("herbench", "lvbench", "nextgqa"):
        subset = table[table.benchmark == benchmark]
        source_ids = sorted(subset.source_video_id.unique())
        sampled = rng.integers(0, len(source_ids), (draws, len(source_ids)))
        for model in sorted(subset.model.unique()):
            cell = subset[subset.model == model]
            for condition in conditions:
                selected = cell[cell.condition == condition]
                if selected.empty:
                    continue
                for metric in METRICS:
                    grouped = selected.groupby("source_video_id")[metric].agg(["sum", "size"]).reindex(source_ids)
                    sums, counts = grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float)
                    accum[(metric, condition)] += (
                        sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1) / 18)
    for metric in METRICS:
        for condition in conditions:
            selected = table[table.condition == condition]
            setting_means = selected.groupby(["model", "benchmark"])[metric].mean()
            values = accum[(metric, condition)]
            rows.append({"metric": metric, "condition": condition, "settings": len(setting_means),
                         "equal_setting_item_mean": float(setting_means.mean()),
                         "ci_low": float(np.quantile(values, .025)),
                         "ci_high": float(np.quantile(values, .975))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reproduced/mdp3_dpp_question_only_stage3_analysis")
    parser.add_argument("--new-analysis-root", type=Path,
                        default=ROOT / "artifacts/mdp3_dpp_question_only_analysis",
                        help="Per-cell analysis root for the MDP3/DPP run.")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = load_all(args.new_analysis_root)

    required = set(NEW + COMPARATORS)
    coverage = table.groupby(["model", "benchmark"]).condition.agg(lambda x: set(x))
    if not coverage.map(lambda x: required.issubset(x)).all():
        raise ValueError("not every setting contains the complete prespecified family")

    setting_rows, macro_rows = [], []
    macro_video = {}
    counter = 0
    for metric in METRICS:
        for new in NEW:
            for comparator in COMPARATORS:
                macro_video[(metric, new, comparator)] = {}
                for (model, benchmark), group in table.groupby(["model", "benchmark"], sort=True):
                    pivot = group.pivot(index=["source_video_id", "item_id"], columns="condition", values=metric)
                    paired = pivot.dropna(subset=[new, comparator])
                    diff = (paired[new].astype(float) - paired[comparator].astype(float)).groupby("source_video_id").mean()
                    estimate, low, high, p = infer(diff.to_numpy(), args.draws,
                                                   np.random.default_rng(args.seed + counter))
                    counter += 1
                    setting_rows.append({
                        "model": model, "benchmark": benchmark, "metric": metric,
                        "new_selector": new, "comparator": comparator,
                        "items": len(paired), "source_video_clusters": len(diff),
                        "equal_video_estimate": estimate, "ci_low": low, "ci_high": high,
                        "p_two_sided": p,
                    })
                    macro_video[(metric, new, comparator)][(model, benchmark)] = diff
                estimate, low, high = macro_bootstrap(
                    macro_video[(metric, new, comparator)], args.draws,
                    args.seed + 10000 + counter)
                macro_rows.append({"metric": metric, "new_selector": new, "comparator": comparator,
                                   "settings": 18, "macro_estimate": estimate,
                                   "ci_low": low, "ci_high": high})

    setting = pd.DataFrame(setting_rows)
    setting["p_bh_within_metric"] = setting.groupby("metric", group_keys=False).p_two_sided.apply(bh)
    setting.to_csv(args.output_dir / "setting_paired_contrasts.csv", index=False)
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(args.output_dir / "macro_paired_contrasts.csv", index=False)

    # Absolute source-video-equal selector outcomes and rankings.
    outcome_rows = []
    for (model, benchmark, condition), group in table.groupby(["model", "benchmark", "condition"]):
        for metric in METRICS:
            video = group.groupby("source_video_id")[metric].mean()
            outcome_rows.append({"model": model, "benchmark": benchmark, "condition": condition,
                                 "metric": metric, "source_video_clusters": len(video),
                                 "equal_video_mean": float(video.mean())})
    outcomes = pd.DataFrame(outcome_rows)
    outcomes.to_csv(args.output_dir / "setting_selector_outcomes.csv", index=False)
    rankings = outcomes.groupby(["condition", "metric"], as_index=False).agg(
        settings=("model", "size"), macro_mean=("equal_video_mean", "mean"))
    rankings["rank"] = rankings.groupby("metric").macro_mean.rank(method="min", ascending=False).astype(int)
    rankings["selector_name"] = rankings.condition.map(NAMES)
    rankings.sort_values(["metric", "rank"]).to_csv(args.output_dir / "macro_selector_rankings.csv", index=False)

    absolute = absolute_macro_uncertainty(table, args.draws, args.seed + 20000)
    absolute["selector_name"] = absolute.condition.map(NAMES)
    absolute.to_csv(args.output_dir / "macro_absolute_uncertainty.csv", index=False)

    selection = table.groupby("condition", as_index=False).agg(
        evaluated_item_answerer_rows=("model", "size"),
        evidence_fraction=("annotated_evidence_recall_at_k", "mean"),
        temporal_coverage=("temporal_coverage", "mean"))
    selection["selector_name"] = selection.condition.map(NAMES)
    selection.to_csv(args.output_dir / "selector_selection_properties.csv", index=False)

    gate = {}
    for new in NEW:
        passed_on = []
        details = {}
        for primary, secondary in (("gold_vs_rest_logodds", "correct"),
                                   ("correct", "gold_vs_rest_logodds")):
            rank = int(rankings[(rankings.condition == new) & (rankings.metric == primary)].iloc[0]["rank"])
            primary_rows = macro[(macro.metric == primary) & (macro.new_selector == new) &
                                 macro.comparator.isin(GATE_COMPARATORS)].set_index("comparator")
            secondary_rows = macro[(macro.metric == secondary) & (macro.new_selector == new) &
                                   macro.comparator.isin(GATE_COMPARATORS)].set_index("comparator")
            improved_one = bool((primary_rows.ci_low > 0).any())
            nonnegative_other = bool((primary_rows.macro_estimate >= 0).all())
            no_supported_harm = bool((secondary_rows.ci_high >= 0).all())
            passed = rank == 1 and improved_one and nonnegative_other and no_supported_harm
            details[primary] = {"rank": rank, "interval_improvement_over_at_least_one": improved_one,
                                "nonnegative_against_both": nonnegative_other,
                                "no_secondary_interval_supported_harm": no_supported_harm,
                                "passed": passed}
            if passed:
                passed_on.append(primary)
        gate[new] = {"passed": bool(passed_on), "passed_on": passed_on, "details": details}
    report = {"paper_inclusion_gate_passed": any(v["passed"] for v in gate.values()),
              "selectors": gate, "draws": args.draws, "seed": args.seed,
              "calls": 17940, "items": 1495}
    (args.output_dir / "inclusion_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nRankings:\n", rankings.sort_values(["metric", "rank"]).to_string(index=False))
    print("\nGate contrasts:\n", macro[macro.comparator.isin(GATE_COMPARATORS)].to_string(index=False))


if __name__ == "__main__":
    main()
