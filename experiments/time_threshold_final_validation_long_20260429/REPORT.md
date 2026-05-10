# Final Time-Threshold Validation Report

## Executive Summary

This is the final confirmation run for the best time-above-threshold configuration found in the sweep.

Tested models:

- `baseline`
- `buffer_default_step`
- `buffer_time_threshold_best_step`

The candidate configuration is:

- `--reward-mode=buffer_time_threshold_penalty`
- `--buffer-penalty-weight=100.0`
- `--time-buffer-penalty-weight=50.0`
- `--buffer-time-threshold=0.1`
- `--observe-buffer-util=true`
- `--buffer-feedback-mode=step`
- `--buffer-feedback-delay-steps=1`

## Setup

- Training seeds: `123, 231, 312`
- Training timesteps per run: `656000`
- Training phases per run: `1`
- PPO rollout / batch: `8192 / 2048`
- Evaluation episodes per run: `20`

## Aggregate Results

| Model | Return Mean +/- Std | Legacy Mean +/- Std | Avg Buffer Mean +/- Std | Avg Peak Buffer Mean +/- Std | Time Above Threshold Mean +/- Std | Max Buffer Mean +/- Std |
|---|---:|---:|---:|---:|---:|---:|
| baseline | `459.277 +/- 150.065` | `1.148 +/- 0.375` | `0.380531 +/- 0.283701` | `0.418634 +/- 0.303261` | `0.422495 +/- 0.309788` | `0.616190 +/- 0.401148` |
| buffer_default_step | `665.494 +/- 197.624` | `1.690 +/- 0.496` | `0.262659 +/- 0.055237` | `0.338833 +/- 0.060426` | `0.416802 +/- 0.067158` | `0.731537 +/- 0.078695` |
| buffer_time_threshold_best_step | `742.131 +/- 125.096` | `1.912 +/- 0.301` | `0.298014 +/- 0.105734` | `0.373138 +/- 0.110885` | `0.531760 +/- 0.121699` | `0.769324 +/- 0.120431` |

## Seed-Level Peak Buffer Results

| Seed | baseline | buffer_default_step | buffer_time_threshold_best_step | Best |
|---|---:|---:|---:|---|
| 123 | `0.089849` | `0.271599` | `0.277441` | `baseline` |
| 231 | `0.478659` | `0.356296` | `0.347317` | `buffer_time_threshold_best_step` |
| 312 | `0.687395` | `0.388604` | `0.494655` | `buffer_default_step` |

## Final Verdict

The final candidate beats baseline on the primary peak-buffer metric.

- vs baseline avg peak buffer: `-10.87%`
- vs baseline avg buffer: `-21.68%`
- vs baseline time above threshold: `+25.86%`
- vs baseline return: `+61.59%`
- vs `buffer_default_step` avg peak buffer: `+10.12%`
- candidate wins on avg peak buffer in `1 / 3` seeds

## Artifacts

- Summary JSON: `experiments/time_threshold_final_validation_long_20260429/summary.json`
- Each run directory contains `model.zip`, `train.log`, `eval.log`, and `eval.json`.
