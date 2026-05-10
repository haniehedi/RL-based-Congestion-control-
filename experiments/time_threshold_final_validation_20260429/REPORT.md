# Final Time-Threshold Validation Report

## Executive Summary

This is the final confirmation run for the best time-above-threshold configuration found in the sweep.

What happened in short:

- the earlier peak-only reward designs did not work reliably
- switching to a reward that penalizes **time spent above a queue threshold** worked much better
- after tuning that reward, the final `0.1 / 50` setting beat both `baseline` and `buffer_default_step` on the main queue-control metrics
- it also improved mean return over baseline

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

Why this version is different:

- instead of punishing only instantaneous queue usage or rare spikes
- it punishes how much of the control interval is spent in a congested queue state
- that gives the agent a smoother and more learnable signal

## Setup

- Training seeds: `123, 231, 312`
- Training timesteps per run: `90112`
- Training phases per run: `1`
- PPO rollout / batch: `8192 / 2048`
- Evaluation episodes per run: `20`

## Aggregate Results

| Model | Return Mean +/- Std | Legacy Mean +/- Std | Avg Buffer Mean +/- Std | Avg Peak Buffer Mean +/- Std | Time Above Threshold Mean +/- Std | Max Buffer Mean +/- Std |
|---|---:|---:|---:|---:|---:|---:|
| baseline | `555.485 +/- 109.805` | `1.389 +/- 0.275` | `0.362979 +/- 0.185449` | `0.417388 +/- 0.173162` | `0.449084 +/- 0.185574` | `0.697091 +/- 0.242297` |
| buffer_default_step | `504.004 +/- 292.442` | `1.301 +/- 0.715` | `0.405861 +/- 0.188537` | `0.488356 +/- 0.190327` | `0.553316 +/- 0.153140` | `0.727227 +/- 0.130670` |
| buffer_time_threshold_best_step | `608.440 +/- 151.728` | `1.564 +/- 0.389` | `0.216027 +/- 0.075646` | `0.294835 +/- 0.083043` | `0.429929 +/- 0.100490` | `0.609112 +/- 0.165230` |

## What Improved

Compared with `baseline`, the final model improved:

- average peak buffer utilization by `29.36%`
- average buffer utilization by `40.49%`
- time above threshold by `4.27%`
- mean return by `9.53%`
- legacy reward from `1.389` to `1.564`
- mean max buffer utilization from `0.697091` to `0.609112`

Compared with `buffer_default_step`, the final model improved:

- average peak buffer utilization by `39.63%`
- average buffer utilization by `46.77%`
- time above threshold from `0.553316` to `0.429929`
- mean return from `504.004` to `608.440`

So this final version was not just a queue-improvement tradeoff. It improved both queue behavior and overall return.

## Seed-Level Peak Buffer Results

| Seed | baseline | buffer_default_step | buffer_time_threshold_best_step | Best |
|---|---:|---:|---:|---|
| 123 | `0.217438` | `0.297842` | `0.275612` | `baseline` |
| 231 | `0.517564` | `0.488728` | `0.385804` | `buffer_time_threshold_best_step` |
| 312 | `0.517162` | `0.678497` | `0.223089` | `buffer_time_threshold_best_step` |

## Final Verdict

The final candidate beats baseline on the primary peak-buffer metric.

- vs baseline avg peak buffer: `-29.36%`
- vs baseline avg buffer: `-40.49%`
- vs baseline time above threshold: `-4.27%`
- vs baseline return: `+9.53%`
- vs `buffer_default_step` avg peak buffer: `-39.63%`
- candidate wins on avg peak buffer in `2 / 3` seeds

## Interpretation

The main lesson is that the reward shape mattered more than simply increasing penalty strength.

What failed earlier:

- direct peak-only penalties were too unstable
- overly strong queue penalties made learning worse

What worked here:

- a lower threshold: `0.1`
- a lighter time-above-threshold penalty: `50`
- keeping the delayed buffer observation from the earlier step-based model

This produced a controller that was more conservative about sustained queue buildup without collapsing transport performance.

## Artifacts

- Summary JSON: `experiments/time_threshold_final_validation_20260429/summary.json`
- Each run directory contains `model.zip`, `train.log`, `eval.log`, and `eval.json`.
