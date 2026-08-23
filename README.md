# Frames Don't Add Up

Official anonymous artifact for **“Frames Don't Add Up: A Factorial Causal Audit of Contextual Evidence Utility in Video-Language Models.”**

The paper asks a simple question: when a video-language model receives several selected frames, do their individually measured values add up? This repository contains the frozen study manifests, analysis-ready outcomes, exact evaluation configuration, and code needed to verify the reported results or rerun the audit with locally obtained benchmark videos.

## Start here

```bash
bash bin/setup.sh
bash bin/verify.sh
```

`verify.sh` checks artifact integrity, anonymity, schemas, factorial coverage, and the paper's headline numerical claims. It does not download models or run GPU inference.

To regenerate the principal plots from the released records:

```bash
bash bin/reproduce_figures.sh
```

Generated files are written to `reproduced/`; the checked-in paper versions remain unchanged.

## What is included

| Path | Contents |
|---|---|
| `artifacts/causal_analysis/` | Per-candidate utility and per-pair factorial records for the 18 grounded settings |
| `artifacts/e1_*`–`artifacts/e9_*` | Analysis-ready records and compact summaries for the interventions, selector census, higher-order audit, controls, and scale analysis |
| `artifacts/qualitative_registry.csv` | Frozen 148-case registry metadata; benchmark frames are not redistributed |
| `manifests/` | Frozen causal holdouts, census manifests, and documented exclusions |
| `configs/` | Exact model revisions, prompt, decoding policy, seed, and portable dataset paths |
| `src/` | Preparation, deterministic evaluation, aggregation, statistics, and plotting code used by the paper |
| `figures/` | Reference copies of the principal paper figures |
| `ARTIFACT_MANIFEST.json` | Size and SHA-256 digest for every released file |

The release is deliberately analysis-first. It includes the records needed to recompute the scientific quantities in the paper, but excludes redundant prepared RGB copies and voluminous scheduler logs.

## Rerun inference

Source videos and model weights are governed by their original licenses and are not redistributed. Obtain the benchmark data, edit the relative paths in `configs/registry.json`, and install the optional inference dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

Prepare an intervention table with the relevant `src/prepare_*.py` program, then run deterministic constrained multiple-choice inference:

```bash
bash bin/run_inference.sh \
  qwen3_vl_4b \
  path/to/all_conditions.parquet \
  reproduced/qwen3_vl_4b_results.jsonl
```

The input table must contain prepared RGB frame paths and the intervention columns expected by `src/evaluate_mcq.py`. Preparation commands expose `--help`; the exact common configuration is in `configs/reproducibility.json`. Large models require hardware appropriate to their published checkpoints. The repository makes no runtime or GPU-hour claim.

## Reproducibility levels

1. **Verify claims (CPU, minutes):** `bash bin/verify.sh`.
2. **Regenerate plots (CPU, minutes):** `bash bin/reproduce_figures.sh`.
3. **Repeat model calls (GPU, benchmark access required):** prepare inputs and use `bin/run_inference.sh`.

All resampling uses seed `20260720`. Primary confidence intervals use 10,000 source-video-cluster bootstrap resamples; paired randomization tests use 10,000 source-video sign-flip draws. Model outputs use deterministic constrained one-token decoding.

