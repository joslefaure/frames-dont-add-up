# Artifact map

The release uses analysis-ready records as the reproducibility boundary. These retain the intervention identities and downstream quantities used in the paper while avoiding duplicated RGB frames and scheduler-specific logs.

| Paper result | Released evidence |
|---|---|
| Evidence access, context, and order | `artifacts/e1_causal_analysis/` |
| Fixed-pool selector audit and census | `artifacts/e2_causal_analysis/`, `artifacts/e2_census_analysis/` |
| Question-type stratification | `artifacts/stage2_question_type_analysis/` |
| $K=16$ selector diagnostic | `artifacts/e2_causal_k16_analysis/` |
| Matched-query MDP$^3$/DPP extension and interface sensitivity | `artifacts/mdp3_dpp_question_only_*`, `artifacts/mdp3_dpp_rich_query_*`, `artifacts/mdp3_dpp_query_sensitivity/` |
| Measured singleton utility and proxy alignment | `artifacts/causal_analysis/e3/`, `artifacts/e3_alignment/` |
| Pair factorial and robustness analyses | `artifacts/causal_analysis/e4/`, `artifacts/e4_*`, `artifacts/e8_*` |
| Higher-order Möbius audit and recovery | `artifacts/e5_expanded_*` |
| Controlled selector-debt regression | `artifacts/e6_controlled_analysis/` |
| Model-family and scale comparison | `artifacts/e9_scale_analysis/` |
| Qualitative selection registry | `artifacts/qualitative_registry.csv` |

Parquet is used for repeated records and CSV for compact summaries. `ARTIFACT_MANIFEST.json` records byte sizes and SHA-256 digests. The principal automated cross-checks are in `src/verify_claims.py`.
