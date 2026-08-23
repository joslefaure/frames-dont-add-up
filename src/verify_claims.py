#!/usr/bin/env python3
"""Recompute headline paper claims from the released analysis records."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"


def close(actual, expected, tolerance=5e-4):
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"expected {expected}, found {actual}")


def main():
    e3_differences = []
    e4_gaps = []
    for path in sorted((A / "causal_analysis/e3").glob("*/summary.csv")):
        table = pd.read_csv(path).set_index("candidate_stratum")
        e3_differences.append(table.loc["evidence", "mean_utility"] - table.loc["outside", "mean_utility"])
    for path in sorted((A / "causal_analysis/e4").glob("*/summary.csv")):
        table = pd.read_csv(path).set_index("pair_type")
        e4_gaps.append(table.loc["evidence_evidence", "mean_absolute_interaction"] -
                       table.loc["outside_outside", "mean_absolute_interaction"])
    assert len(e3_differences) == 18 and sum(x > 0 for x in e3_differences) == 17
    assert len(e4_gaps) == 18 and all(x > 0 for x in e4_gaps)
    primary = pd.read_csv(A / "causal_analysis/bootstrap_primary_contrasts.csv")
    utility = primary[(primary.experiment == "e3") &
                      (primary.contrast == "evidence_minus_outside_utility")]
    assert len(utility) == 18 and ((utility.ci_low > 0) | (utility.ci_high < 0)).sum() == 16
    interaction = primary[(primary.experiment == "e4") &
                          (primary.contrast == "evidence_minus_outside_absolute_interaction")]
    assert len(interaction) == 18 and (interaction.estimate > 0).all()
    close(float(interaction.estimate.median()), 0.513, tolerance=0.001)

    chrono = pd.read_csv(A / "e8_chronological_analysis/primary_contrasts.csv")
    assert len(chrono) == 18 and (chrono.equal_video_absolute_interaction_gap > 0).all()

    higher = pd.read_csv(A / "e5_expanded_analysis/bootstrap_order_fractions.csv")
    mass = higher[higher.order == "3_plus_4"].fraction_absolute_component
    close(float(mass.min()), 0.235, tolerance=0.001)
    close(float(mass.max()), 0.277, tolerance=0.001)

    recovery = pd.read_csv(A / "e5_expanded_selection_regret_analysis/macro_summary.csv")
    b3 = recovery[recovery.budget == 3].set_index("method")
    close(float(b3.loc["singleton", "optimal_set_recovery"]), 0.552, tolerance=0.0006)
    close(float(b3.loc["pairwise", "optimal_set_recovery"]), 0.686, tolerance=0.0006)
    reduction = 1 - b3.loc["pairwise", "mean_logodds_regret"] / b3.loc["singleton", "mean_logodds_regret"]
    close(float(reduction), 0.582, tolerance=0.0006)

    winners = pd.read_csv(A / "e2_census_analysis/winner_comparison.csv")
    grounded = winners[winners.benchmark != "videomme"]
    assert len(grounded) == 18 and not grounded.recall_winner_overlaps_logodds_winner.astype(bool).any()

    adjusted = pd.read_csv(A / "e4_utility_adjusted_analysis/adjusted_setting_results.csv")
    full = adjusted[adjusted.specification == "full"]
    assert len(full) == 18 and (full.coefficient > 0).all() and (full.ci_low > 0).all()

    print("PASS: 17/18 positive singleton-utility settings; 16/18 clustered CIs exclude zero")
    print("PASS: evidence-pair interaction gap positive in 18/18 settings; median +0.513")
    print("PASS: chronological replication positive in 18/18 settings")
    print("PASS: higher-order mass, selector recovery/regret, census mismatch, and adjusted audit")


if __name__ == "__main__":
    main()
