#!/usr/bin/env python3
"""Compare matched question-only MDP3/DPP runs with richer-query sensitivity runs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARKS = ("herbench", "lvbench", "nextgqa")
CONDITIONS = ("plain_dpp_8", "mdp3_8")
METRICS = ("gold_vs_rest_logodds", "correct")


def load_analysis(root: Path, sources: pd.DataFrame) -> pd.DataFrame:
    paths = sorted(root.glob("*/per_item.parquet"))
    if len(paths) != 18:
        raise ValueError(f"expected 18 analysis cells under {root}, found {len(paths)}")
    table = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    table = table[["model", "benchmark", "item_id", "condition", *METRICS]]
    return table.merge(sources, on=["benchmark", "item_id"], validate="many_to_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-analysis-root", type=Path,
                        default=Path("artifacts/mdp3_dpp_question_only_analysis"))
    parser.add_argument("--sensitivity-analysis-root", type=Path,
                        default=Path("artifacts/mdp3_dpp_rich_query_analysis"))
    parser.add_argument("--primary-selection-root", type=Path,
                        default=Path("artifacts/mdp3_dpp_question_only_selections"))
    parser.add_argument("--sensitivity-selection-root", type=Path,
                        default=Path("artifacts/mdp3_dpp_rich_query_selections"))
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reproduced/mdp3_dpp_query_sensitivity"))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_tables = []
    for benchmark in BENCHMARKS:
        refined = args.manifest_root / "refined" / f"{benchmark}.jsonl"
        manifest_path = (refined if refined.is_file() else
                         args.manifest_root / "causal_holdout" / f"{benchmark}.jsonl")
        manifest = pd.read_json(manifest_path, lines=True)
        manifest["source_video_id"] = manifest.source_video_id.astype(str)
        source_tables.append(manifest.drop_duplicates("item_id")[
            ["benchmark", "item_id", "source_video_id"]])
    sources = pd.concat(source_tables, ignore_index=True)
    primary = load_analysis(args.primary_analysis_root, sources)
    sensitivity = load_analysis(args.sensitivity_analysis_root, sources)

    overlap_rows = []
    for benchmark in BENCHMARKS:
        left = pd.read_csv(args.primary_selection_root / f"{benchmark}.selections.csv")
        right = pd.read_csv(args.sensitivity_selection_root / f"{benchmark}.selections.csv")
        merged = left.merge(right, on=["benchmark", "item_id", "condition"],
                            validate="one_to_one", suffixes=("_primary", "_sensitivity"))
        for condition, group in merged.groupby("condition"):
            overlaps = []
            for row in group.itertuples():
                first = set(ast.literal_eval(row.selected_candidate_indices_primary))
                second = set(ast.literal_eval(row.selected_candidate_indices_sensitivity))
                overlaps.append(len(first & second))
            overlap_rows.append({
                "benchmark": benchmark, "condition": condition, "items": len(group),
                "exact_set_fraction": float(np.mean(np.asarray(overlaps) == 8)),
                "mean_overlap_of_8": float(np.mean(overlaps)),
                "median_overlap_of_8": float(np.median(overlaps)),
                "minimum_overlap_of_8": int(np.min(overlaps)),
            })
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(args.output_dir / "selection_overlap.csv", index=False)
    aggregate_overlap = (overlap.groupby("condition")
                         .apply(lambda x: pd.Series({
                             "items": int(x["items"].sum()),
                             "exact_set_fraction": float(np.average(
                                 x["exact_set_fraction"], weights=x["items"])),
                             "mean_overlap_of_8": float(np.average(
                                 x["mean_overlap_of_8"], weights=x["items"])),
                         }), include_groups=False).reset_index())
    aggregate_overlap.to_csv(args.output_dir / "selection_overlap_aggregate.csv", index=False)

    paired = primary.merge(sensitivity, on=["model", "benchmark", "item_id", "condition",
                                            "source_video_id"], validate="one_to_one",
                           suffixes=("_primary", "_sensitivity"))
    rng = np.random.default_rng(args.seed)
    outcome_rows = []
    for metric in METRICS:
        for condition in CONDITIONS:
            differences = {}
            for (model, benchmark), group in paired[paired.condition == condition].groupby(
                    ["model", "benchmark"], sort=True):
                item_difference = (group[f"{metric}_primary"].astype(float)
                                   - group[f"{metric}_sensitivity"].astype(float))
                differences[(model, benchmark)] = item_difference.groupby(
                    group.source_video_id).mean()
            observed = float(np.mean([values.mean() for values in differences.values()]))
            bootstrap = np.zeros(args.draws)
            for benchmark in BENCHMARKS:
                settings = {key: value for key, value in differences.items()
                            if key[1] == benchmark}
                source_ids = sorted(set.intersection(*(set(value.index)
                                                       for value in settings.values())))
                sampled = rng.integers(0, len(source_ids), (args.draws, len(source_ids)))
                for values in settings.values():
                    array = values.loc[source_ids].to_numpy(float)
                    bootstrap += array[sampled].mean(axis=1) / len(differences)
            outcome_rows.append({
                "metric": metric, "condition": condition, "settings": len(differences),
                "primary_minus_sensitivity": observed,
                "ci_low": float(np.quantile(bootstrap, .025)),
                "ci_high": float(np.quantile(bootstrap, .975)),
            })
    outcome = pd.DataFrame(outcome_rows)
    outcome.to_csv(args.output_dir / "outcome_differences.csv", index=False)
    provenance = {
        "primary_query": "question_only",
        "sensitivity_query": "question_and_answer_choices",
        "draws": args.draws,
        "seed": args.seed,
        "paired_answerer_rows": len(paired),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(overlap.to_string(index=False))
    print("\nAggregate overlap:\n", aggregate_overlap.to_string(index=False))
    print("\nOutcome differences:\n", outcome.to_string(index=False))


if __name__ == "__main__":
    main()
