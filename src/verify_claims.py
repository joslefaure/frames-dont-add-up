#!/usr/bin/env python3
"""Recompute headline paper claims from the released analysis records."""
from pathlib import Path
import json

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

    # Evidence-access and order claims in the main text.
    condition_tables = []
    for path in sorted((A / "e1_causal_analysis").glob("*/condition_summary.csv")):
        table = pd.read_csv(path).set_index("condition")
        condition_tables.append(table)
    assert len(condition_tables) == 18
    oracle_uniform = np.array([t.loc["oracle8", "mean_logodds"] -
                               t.loc["uniform8", "mean_logodds"] for t in condition_tables])
    oracle_mixed = np.array([t.loc["oracle8", "mean_logodds"] -
                             t.loc["oracle4_distractor4", "mean_logodds"]
                             for t in condition_tables])
    oracle_reverse = np.array([t.loc["oracle8", "mean_logodds"] -
                               t.loc["oracle8_reversed", "mean_logodds"]
                               for t in condition_tables])
    reverse_accuracy = np.array([t.loc["oracle8", "accuracy"] -
                                 t.loc["oracle8_reversed", "accuracy"]
                                 for t in condition_tables])
    assert (oracle_uniform > 0).all() and (oracle_mixed > 0).all()
    assert (oracle_reverse > 0).all() and (reverse_accuracy > 0).all()
    close(float(np.median(oracle_uniform)), 1.04, tolerance=0.005)
    close(float(np.median(oracle_reverse)), 0.34, tolerance=0.005)

    # Pointwise-proxy claims use a median setting-level pooled Spearman but an
    # equal-setting mean regret; keeping both aggregators explicit prevents a
    # silent mean/median substitution in prose or tables.
    proxy = pd.read_csv(A / "e3_alignment/all_summary.csv")
    spectral = proxy[proxy.proxy == "visual_spectral_saliency"]
    assert len(spectral) == 12
    close(float(spectral.spearman_pooled.median()), 0.0001, tolerance=0.0002)
    close(float(spectral.mean_pointwise_regret.median()), 0.961, tolerance=0.001)
    assert spectral.spearman_pooled.min() > -0.048 and spectral.spearman_pooled.max() < 0.044
    relevance = proxy[proxy.proxy == "siglip_relevance"]
    assert len(relevance) == 12 and (relevance.mean_pointwise_regret > 0).all()
    annotation = proxy[proxy.proxy == "annotated_membership"]
    close(float(annotation.spearman_pooled.median()), 0.121, tolerance=0.001)
    close(float(annotation.mean_pointwise_regret.mean()), 0.537, tolerance=0.001)

    spectral_ci = pd.read_csv(A / "reviewer_response_analysis/spectral_saliency_cluster_bootstrap.csv")
    assert len(spectral_ci) == 12
    assert ((spectral_ci.ci_low <= 0) & (spectral_ci.ci_high >= 0)).sum() == 11

    # Bounded score scales reproduce the direction of the primary contrast.
    scales = pd.read_csv(A / "e4_score_scale_analysis/all_summary.csv")
    for score_scale, expected in (("probability", 0.041), ("accuracy", 0.063)):
        pivot = scales[scales.score_scale == score_scale].pivot(
            index=["model", "benchmark"], columns="pair_type",
            values="mean_absolute_interaction")
        gap = pivot.evidence_evidence - pivot.outside_outside
        assert len(gap) == 18 and (gap > 0).all()
        close(float(gap.median()), expected, tolerance=0.001)

    # Question-type wording is intentionally about the equal-setting median in
    # each stratum, not every individual model-by-stratum cell.
    question_types = pd.read_csv(A / "stage2_question_type_analysis/macro_summary.csv")
    assert len(question_types) == 19 and (question_types.median_absolute_gap > 0).all()
    assert question_types.groupby("benchmark").size().to_dict() == {
        "herbench": 6, "lvbench": 6, "nextgqa": 7}
    next_contrasts = pd.read_csv(A / "stage2_question_type_analysis/planned_contrasts.csv")
    assert len(next_contrasts) == 6 and (next_contrasts.estimate > 0).all()
    assert (next_contrasts.ci_low > 0).sum() == 3

    chrono = pd.read_csv(A / "e8_chronological_analysis/primary_contrasts.csv")
    assert len(chrono) == 18 and (chrono.equal_video_absolute_interaction_gap > 0).all()
    assert (chrono.ci_low > 0).all() and (chrono.p_bh < .05).all()
    close(float(chrono.equal_video_absolute_interaction_gap.median()), 0.543, tolerance=0.001)

    higher = pd.read_csv(A / "e5_expanded_analysis/bootstrap_order_fractions.csv")
    mass = higher[higher.order == "3_plus_4"].fraction_absolute_component
    close(float(mass.min()), 0.235, tolerance=0.001)
    close(float(mass.max()), 0.277, tolerance=0.001)
    first_order = higher[higher.order.astype(str) == "1"].fraction_absolute_component
    assert first_order.min() >= .34 and first_order.max() < .405

    recovery = pd.read_csv(A / "e5_expanded_selection_regret_analysis/macro_summary.csv")
    b3 = recovery[recovery.budget == 3].set_index("method")
    close(float(b3.loc["singleton", "optimal_set_recovery"]), 0.552, tolerance=0.0006)
    close(float(b3.loc["pairwise", "optimal_set_recovery"]), 0.686, tolerance=0.0006)
    reduction = 1 - b3.loc["pairwise", "mean_logodds_regret"] / b3.loc["singleton", "mean_logodds_regret"]
    close(float(reduction), 0.582, tolerance=0.0006)
    close(float(b3.loc["singleton", "mean_logodds_regret"]), 0.326, tolerance=0.001)
    close(float(b3.loc["pairwise", "mean_logodds_regret"]), 0.136, tolerance=0.001)
    close(float(b3.loc["singleton", "accuracy"]), 0.692, tolerance=0.001)
    close(float(b3.loc["pairwise", "accuracy"]), 0.709, tolerance=0.001)
    close(float(b3.loc["exhaustive", "accuracy"]), 0.719, tolerance=0.001)
    b2 = recovery[recovery.budget == 2].set_index("method")
    close(float(b2.loc["singleton", "optimal_set_recovery"]), 0.546, tolerance=0.001)
    close(float(b2.loc["singleton", "mean_logodds_regret"]), 0.271, tolerance=0.001)
    paired_recovery = pd.read_csv(A / "e5_expanded_selection_regret_analysis/paired_method_contrasts.csv")
    budget3 = paired_recovery[(paired_recovery.budget == 3) &
                              (paired_recovery.contrast == "pairwise_minus_singleton")]
    regret_rows = budget3[budget3.metric == "logodds_regret_reduction"]
    recovery_rows = budget3[budget3.metric == "optimal_set_recovery_gain"]
    accuracy_rows = budget3[budget3.metric == "accuracy_gain"]
    assert len(regret_rows) == len(recovery_rows) == len(accuracy_rows) == 9
    assert (regret_rows.ci_low > 0).all()
    assert (recovery_rows.estimate > 0).all() and (recovery_rows.ci_low > 0).sum() == 8
    assert (accuracy_rows.estimate > 0).sum() == 7

    winners = pd.read_csv(A / "e2_census_analysis/winner_comparison.csv")
    grounded = winners[winners.benchmark != "videomme"]
    assert len(grounded) == 18 and not grounded.recall_winner_overlaps_logodds_winner.astype(bool).any()

    k16_winners = pd.read_csv(A / "e2_causal_k16_analysis/winner_comparison.csv")
    assert len(k16_winners) == 18
    assert k16_winners.recall_winner_overlaps_logodds_winner.astype(bool).sum() == 2

    adjusted = pd.read_csv(A / "e4_utility_adjusted_analysis/adjusted_setting_results.csv")
    full = adjusted[adjusted.specification == "full"]
    assert len(full) == 18 and (full.coefficient > 0).all() and (full.ci_low > 0).all()
    assert (full.p_bh_within_specification < .05).all()
    close(float(full.coefficient.median()), 0.07799, tolerance=0.0001)
    close(float(full.coefficient.min()), 0.01044, tolerance=0.0001)
    close(float(full.coefficient.max()), 0.17536, tolerance=0.0001)

    # Pair topology and descriptive outcome taxonomy.
    topology_tables = [pd.read_csv(path) for path in
                       sorted((A / "e4_topology_analysis").glob("*/topology_summary.csv"))]
    topology = pd.concat(topology_tables, ignore_index=True)
    topology_pivot = topology.pivot(index=["model", "benchmark"], columns="topology",
                                    values="mean_absolute_interaction")
    matched = topology_pivot.dropna(subset=["evidence_evidence_same_interval",
                                            "evidence_evidence_distinct_interval"])
    same_minus_distinct = (matched.evidence_evidence_same_interval -
                           matched.evidence_evidence_distinct_interval)
    assert len(same_minus_distinct) == 12 and (same_minus_distinct > 0).sum() == 7
    close(float(same_minus_distinct.median()), 0.043, tolerance=0.001)
    macro_topology = topology.groupby("topology").mean_absolute_interaction.mean()
    close(float(macro_topology["evidence_evidence_same_interval"]), 0.943, tolerance=0.001)

    mechanisms = pd.read_csv(A / "e4_mechanism_analysis/macro_summary.csv").set_index("pair_type")
    close(float(mechanisms.loc["evidence_evidence", "both_help_but_joint_worse"]), .1489)
    close(float(mechanisms.loc["outside_outside", "both_help_but_joint_worse"]), .0998)
    close(float(mechanisms.loc["evidence_evidence", "complementary_rescue"]), .0159)
    close(float(mechanisms.loc["outside_outside", "complementary_rescue"]), .0092)

    selector = pd.read_csv(A / "mdp3_dpp_question_only_stage3_analysis/macro_paired_contrasts.csv")
    dpp = selector[(selector.new_selector == "plain_dpp_8") &
                   selector.comparator.isin(["siglip_dino_submodular8", "bolt_its8"])]
    logodds = dpp[dpp.metric == "gold_vs_rest_logodds"].set_index("comparator")
    accuracy = dpp[dpp.metric == "correct"].set_index("comparator")
    close(float(logodds.loc["siglip_dino_submodular8", "macro_estimate"]), 0.124148)
    close(float(logodds.loc["bolt_its8", "macro_estimate"]), 0.129869)
    assert (logodds.ci_low > 0).all() and (accuracy.ci_low < 0).all() and (accuracy.ci_high > 0).all()
    gate = json.loads((A / "mdp3_dpp_question_only_stage3_analysis/inclusion_gate.json").read_text())
    assert gate["paper_inclusion_gate_passed"] and gate["selectors"]["plain_dpp_8"]["passed"]

    query = pd.read_csv(A / "mdp3_dpp_query_sensitivity/outcome_differences.csv")
    assert len(query) == 4 and (query.ci_low < 0).all() and (query.ci_high > 0).all()
    overlap = pd.read_csv(A / "mdp3_dpp_query_sensitivity/selection_overlap_aggregate.csv").set_index("condition")
    close(float(overlap.loc["plain_dpp_8", "mean_overlap_of_8"]), 4.532441)
    close(float(overlap.loc["mdp3_8", "mean_overlap_of_8"]), 4.391304)

    # Remaining stress-test and accounting claims.
    annotation_width = pd.read_csv(A / "e4_annotation_sensitivity/coverage_thresholds.csv")
    width_medians = annotation_width.groupby("maximum_sampling_coverage").equal_video_gap.median()
    for threshold, expected in ((.10, .536), (.25, .513), (.50, .508)):
        values = annotation_width[annotation_width.maximum_sampling_coverage == threshold]
        assert len(values) == 18 and (values.equal_video_gap > 0).all()
        close(float(width_medians.loc[threshold]), expected, tolerance=.001)

    extensions = pd.read_csv(A / "e8_extension_analysis/topology_contrasts.csv")
    expected_controls = {"extra_context": .454, "hard_distractor": .344,
                         "k16": .359, "direct_rgb_png": .631}
    for condition, expected in expected_controls.items():
        values = extensions[extensions.robustness_condition == condition]
        assert len(values) == 9 and (values.evidence_minus_outside_absolute_interaction > 0).all()
        close(float(values.evidence_minus_outside_absolute_interaction.median()), expected,
              tolerance=.001)

    boundary_tables = [pd.read_csv(path) for path in
                       sorted((A / "e8_boundary_offset_analysis").glob("*/summary.csv"))]
    boundary = pd.concat(boundary_tables, ignore_index=True).pivot(
        index=["model", "benchmark"], columns="pair_type",
        values="mean_absolute_interaction")
    boundary_gap = boundary.evidence_evidence - boundary.outside_outside
    assert len(boundary_gap) == 9 and (boundary_gap > 0).all()
    close(float(boundary_gap.median()), .596, tolerance=.001)

    repeats = pd.concat([pd.read_csv(path) for path in
                         sorted((A / "e8_repeat_analysis").glob("*.csv"))], ignore_index=True)
    assert repeats.rows.sum() == 1350
    assert repeats.max_logodds_abs_delta.max() == 0
    assert repeats.max_probability_abs_delta.max() == 0 and repeats.changed_correct.sum() == 0
    noop = pd.read_csv(A / "e8_analysis/all_summary.csv")
    noop = noop[noop.robustness_condition == "noop"]
    assert len(noop) == 9 and noop.max_absolute_interaction.max() == 0

    extremes = pd.read_csv(A / "reviewer_response_analysis/interaction_extreme_summary.csv").set_index("pair_type")
    assert int(extremes.loc["evidence_evidence", "records"]) == 27000
    assert int(extremes.loc["outside_outside", "records"]) == 27000
    close(float(extremes.loc["evidence_evidence", "fraction_gt_10"]), .0102, tolerance=.00005)
    close(float(extremes.loc["outside_outside", "fraction_gt_10"]), .0006, tolerance=.00005)
    extreme_robustness = pd.read_csv(
        A / "reviewer_response_analysis/interaction_extreme_robustness_summary.csv")
    assert len(extreme_robustness) == 4 and (extreme_robustness.positive == 18).all()

    debt = pd.read_csv(A / "e6_controlled_analysis/controlled_logistic_coefficients.csv")
    debt = debt[debt.term == "z_interaction_debt"].iloc[0]
    close(float(debt.coefficient), .055, tolerance=.001)
    close(float(debt.std_error_clustered_source_video), .085, tolerance=.001)
    close(float(debt.p_value), .515, tolerance=.001)

    scale = pd.read_csv(A / "e9_scale_analysis/scale_pairs.csv")
    assert len(scale) == 6 and (scale.selector_accuracy_large_minus_small > 0).all()
    qwen = scale[scale.family == "qwen"]
    internvl = scale[scale.family == "internvl"]
    assert (qwen.evidence_pair_absolute_interaction_large_minus_small > 0).sum() == 2
    assert (internvl.evidence_pair_absolute_interaction_large_minus_small > 0).sum() == 2

    registry = pd.read_csv(A / "qualitative_registry.csv")
    assert len(registry) == 148
    assert registry.selection_group.str.startswith("stress_").sum() == 100
    assert registry.selection_group.str.startswith("seeded_typical_").sum() == 48

    assert sum((124500, 243000, 540000, 57600, 107640, 36000)) == 1108740

    print("PASS: 17/18 positive singleton-utility settings; 16/18 clustered CIs exclude zero")
    print("PASS: evidence-pair interaction gap positive in 18/18 settings; median +0.513")
    print("PASS: chronological replication positive in 18/18 settings")
    print("PASS: E1 access/order, pointwise proxies, score scales, and question-type strata")
    print("PASS: higher-order mass, selector recovery/regret, census/K16 mismatch, and adjusted audit")
    print("PASS: matched-query DPP gate and richer-query sensitivity claims")
    print("PASS: topology, mechanisms, controls, deterministic checks, scale, registry, and call total")


if __name__ == "__main__":
    main()
