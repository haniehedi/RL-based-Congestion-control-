# Time-Above-Threshold Validation Report

## Executive Summary

This is the first alternative reward family that showed a meaningful improvement over `buffer_default_step` in multi-seed validation.

Main result:

- the time-above-threshold idea is **better than `buffer_default_step`**
- but it still does **not** beat `baseline` overall on the primary queue metrics
- it won `1 / 3` seeds on average peak buffer utilization

So this idea is more promising than the peak-only reward family, but it is still not a clear final answer.

## Setup

- Training seeds: `123, 231, 312`
- Training timesteps per run: `90112`
- Training phases per run: `1`
- PPO rollout / batch: `8192 / 2048`
- Evaluation episodes per run: `20`

This validation tests a new reward family that penalizes the fraction of each control interval spent above a buffer threshold.

## Aggregate Results

| Model | Return Mean +/- Std | Avg Buffer Mean +/- Std | Avg Peak Buffer Mean +/- Std | Time Above Threshold Mean +/- Std | Max Buffer Mean +/- Std |
|---|---:|---:|---:|---:|---:|
| baseline | `555.485 +/- 109.805` | `0.362979 +/- 0.185449` | `0.417388 +/- 0.173162` | `0.449084 +/- 0.185574` | `0.697091 +/- 0.242297` |
| buffer_default_step | `504.004 +/- 292.442` | `0.405861 +/- 0.188537` | `0.488356 +/- 0.190327` | `0.553316 +/- 0.153140` | `0.727227 +/- 0.130670` |
| buffer_time_threshold_step | `528.154 +/- 126.252` | `0.367307 +/- 0.128754` | `0.446083 +/- 0.095883` | `0.492920 +/- 0.075678` | `0.642454 +/- 0.122756` |

## Seed-Level Peak Buffer Results

| Seed | baseline | buffer_default_step | buffer_time_threshold_step | Best |
|---|---:|---:|---:|---|
| 123 | `0.217438` | `0.297842` | `0.429272` | `baseline` |
| 231 | `0.517564` | `0.488728` | `0.359717` | `buffer_time_threshold_step` |
| 312 | `0.517162` | `0.678497` | `0.549260` | `baseline` |

## Verdict

The time-above-threshold idea beat `buffer_default_step`, but still did not beat baseline on peak-buffer control.

- `buffer_time_threshold_step` vs baseline avg peak buffer: `+6.87%`
- `buffer_time_threshold_step` vs baseline avg buffer: `+1.19%`
- `buffer_time_threshold_step` vs baseline time above threshold: `+9.76%`
- `buffer_time_threshold_step` vs `buffer_default_step` avg peak buffer: `-8.66%`
- `buffer_time_threshold_step` wins on avg peak buffer in `1 / 3` seeds

## Interpretation

Compared with the earlier peak-aware attempts, this is a better direction.

Why:

- it reduced average peak buffer utilization relative to `buffer_default_step`
- it also reduced time spent above the threshold relative to `buffer_default_step`
- and it lowered mean max buffer utilization relative to both `baseline` and `buffer_default_step`

But it still falls short because:

- its average peak buffer utilization is still higher than baseline
- its average time above threshold is still higher than baseline
- its average buffer utilization is only slightly worse than baseline, not clearly better

So the current evidence says:

- peak-only reward shaping was not a good idea
- time-above-threshold reward shaping is a better idea
- but it still needs more tuning before it can be called successful

## Artifacts

- Summary JSON: `experiments/time_threshold_validation_20260429/summary.json`
- Each run directory contains `model.zip`, `train.log`, `eval.log`, and `eval.json`.
