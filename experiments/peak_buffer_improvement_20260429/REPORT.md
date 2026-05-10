# Peak Reward Improvement Report

## Executive Summary

This follow-up experiment tested whether a gentler peak-aware reward could improve on the failed `buffer_peak_step` model.

Main outcome:

- `buffer_peak_linear_step` was the **best new peak-aware variant**
- it substantially improved over the original `buffer_peak_step`
- but it still did **not** beat `baseline` on queue metrics
- it also did **not** beat `buffer_default_step` on average or peak buffer usage

So the spike-aware idea became better than before, but it is still not the best overall design.

## Setup

- Training timesteps per model: `90112`
- Training phases: `1`
- PPO rollout / batch: `8192 / 2048`
- Training seed: `123`
- Evaluation episodes: `20`
- Evaluation seed: `2123`

This report combines the previous matched-budget sweep with newly trained peak-aware variants.

## Results

| Model | Mean Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Avg Peak Buffer Util | Max Buffer Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `484.438` | `1.211` | `2207475.686` | `0.550860` | `0.038804` | `0.148993` | `0.217438` | `0.438291` |
| buffer_default_step | `639.706` | `1.623` | `2815954.502` | `0.637032` | `0.043472` | `0.233879` | `0.297842` | `0.577989` |
| buffer_light_step | `654.511` | `1.643` | `3134466.146` | `0.855309` | `0.056667` | `0.285398` | `0.383985` | `0.586741` |
| buffer_strong_step | `-725.265` | `-1.536` | `2826635.782` | `3.517540` | `0.187021` | `0.692781` | `0.711326` | `0.997482` |
| buffer_default_rtt | `453.015` | `1.175` | `2663808.361` | `0.889651` | `0.077433` | `0.427869` | `0.526449` | `0.788017` |
| buffer_reward_only | `191.360` | `0.546` | `2931694.968` | `1.336163` | `0.280295` | `0.679265` | `0.748934` | `0.998306` |
| buffer_peak_step | `492.611` | `1.375` | `2867246.072` | `0.751651` | `0.131141` | `0.477593` | `0.585988` | `0.737790` |
| buffer_peak_linear_step | `378.127` | `0.993` | `2762390.327` | `1.209732` | `0.049414` | `0.264955` | `0.321646` | `0.492083` |
| buffer_peak_linear_low_threshold_step | `441.096` | `1.204` | `3621066.956` | `1.542228` | `0.135693` | `0.622078` | `0.741823` | `0.922507` |

## New Variants

### `buffer_peak_linear_step`

- vs baseline avg peak buffer util: `+47.93%`
- vs baseline avg buffer util: `+77.83%`
- vs `buffer_default_step` avg peak buffer util: `+7.99%`
- vs original `buffer_peak_step` avg peak buffer util: `-45.11%`
- vs original `buffer_peak_step` avg buffer util: `-44.52%`
- vs original `buffer_peak_step` max buffer util: `-33.30%`
- mean return: `378.127`
- legacy reward: `0.993`

### `buffer_peak_linear_low_threshold_step`

- vs baseline avg peak buffer util: `+241.17%`
- vs baseline avg buffer util: `+317.52%`
- vs `buffer_default_step` avg peak buffer util: `+149.07%`
- vs original `buffer_peak_step` avg peak buffer util: `+26.59%`
- mean return: `441.096`
- legacy reward: `1.204`

## Main Findings

### 1. The best improved variant was `buffer_peak_linear_step`

This variant was the only new design that clearly improved on the original `buffer_peak_step`.

Compared with `buffer_peak_step`, it reduced:

- average buffer utilization from `0.477593` to `0.264955`
- average peak buffer utilization from `0.585988` to `0.321646`
- max buffer utilization from `0.737790` to `0.492083`

So the gentler linear spike penalty was much better than the original harsh quadratic version.

### 2. It still did not beat baseline

Even the improved linear version remained worse than baseline on all queue metrics:

- baseline avg buffer util: `0.148993`
- `buffer_peak_linear_step` avg buffer util: `0.264955`
- baseline avg peak buffer util: `0.217438`
- `buffer_peak_linear_step` avg peak buffer util: `0.321646`
- baseline max buffer util: `0.438291`
- `buffer_peak_linear_step` max buffer util: `0.492083`

So the improvement was real, but not enough.

### 3. It also did not beat `buffer_default_step`

Compared with `buffer_default_step`, the improved linear variant still had:

- higher average buffer utilization
- higher average peak buffer utilization
- lower return

So among the queue-aware designs tested so far, `buffer_default_step` still remains the stronger overall model.

### 4. The lower-threshold linear variant did not help

`buffer_peak_linear_low_threshold_step` performed badly.

Its queue metrics were worse than:

- baseline
- `buffer_default_step`
- `buffer_peak_linear_step`

That suggests the low-threshold setting pushed the policy too hard and destabilized behavior again.

## Conclusion

The improved reward did partially fix the first peak-aware design, but not enough to become the new best model.

Final ranking of the new peak-aware attempts:

1. `buffer_peak_linear_step`
2. `buffer_peak_step`
3. `buffer_peak_linear_low_threshold_step`

Best overall conclusion:

- the harsh quadratic peak penalty was too unstable
- the gentler linear peak penalty was a meaningful improvement
- but the current peak-aware family still does not beat baseline on queue control or `buffer_default_step` on overall queue-aware performance

## Artifacts

- Summary JSON: `experiments/peak_buffer_improvement_20260429/summary.json`
- New model directories contain `model.zip`, `train.log`, `eval.log`, and `eval.json`.
