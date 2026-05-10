# Peak-Aware Idea Validation Report

## Executive Summary

This is the strongest validation run so far because it uses multiple independent training seeds instead of a single run.

Main answer:

- the current peak-aware idea is **not validated**
- `buffer_peak_linear_step` did **not** beat baseline on peak-buffer control
- it also did **not** beat `buffer_default_step`
- it won `0 / 3` seeds on the primary metric

So the idea is still only a hypothesis, not a demonstrated improvement.

## Setup

- Training seeds: `123, 231, 312`
- Training timesteps per run: `90112`
- Training phases per run: `1`
- PPO rollout / batch: `8192 / 2048`
- Evaluation episodes per run: `20`

This validation compares the original baseline, the earlier queue-aware `buffer_default_step`, and the best improved peak-aware candidate across multiple training seeds.

## Aggregate Results

| Model | Return Mean +/- Std | Legacy Mean +/- Std | Avg Buffer Mean +/- Std | Avg Peak Buffer Mean +/- Std | Max Buffer Mean +/- Std |
|---|---:|---:|---:|---:|---:|
| baseline | `555.485 +/- 109.805` | `1.389 +/- 0.275` | `0.362979 +/- 0.185449` | `0.417388 +/- 0.173162` | `0.697091 +/- 0.242297` |
| buffer_default_step | `504.004 +/- 292.442` | `1.301 +/- 0.715` | `0.405861 +/- 0.188537` | `0.488356 +/- 0.190327` | `0.727227 +/- 0.130670` |
| buffer_peak_linear_step | `565.286 +/- 208.772` | `1.501 +/- 0.546` | `0.465504 +/- 0.175795` | `0.550607 +/- 0.198402` | `0.785942 +/- 0.262435` |

## Seed-Level Peak Buffer Results

| Seed | baseline | buffer_default_step | buffer_peak_linear_step | Best |
|---|---:|---:|---:|---|
| 123 | `0.217438` | `0.297842` | `0.321646` | `baseline` |
| 231 | `0.517564` | `0.488728` | `0.658292` | `buffer_default_step` |
| 312 | `0.517162` | `0.678497` | `0.671883` | `baseline` |

## Verdict

The improved peak-aware idea did not beat either baseline or `buffer_default_step` on the primary peak-buffer metric.

- `buffer_peak_linear_step` vs baseline avg peak buffer: `+31.92%`
- `buffer_peak_linear_step` vs baseline avg buffer: `+28.25%`
- `buffer_peak_linear_step` vs `buffer_default_step` avg peak buffer: `+12.75%`
- `buffer_peak_linear_step` wins on avg peak buffer in `0 / 3` seeds

## Interpretation

The multi-seed result is more important than any earlier single run.

What it shows:

- the peak-aware reward can sometimes achieve good return
- but it does not consistently reduce queue spikes
- and it does not provide a reliable improvement over the simpler alternatives

In other words, the current design is not robust enough to call it a good idea for this project yet.

## Artifacts

- Summary JSON: `experiments/peak_validation_20260429/summary.json`
- Each run directory contains `model.zip`, `train.log`, `eval.log`, and `eval.json`.
