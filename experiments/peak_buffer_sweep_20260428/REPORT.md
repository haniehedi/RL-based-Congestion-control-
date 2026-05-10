# Peak-Aware Reward Variant Report

## Executive Summary

This report reflects the newest matched-budget sweep that included the new `buffer_peak_step` model.

Main outcome:

- the new peak-aware reward did **not** improve peak queue control
- `buffer_peak_step` was worse than both `baseline` and `buffer_default_step` on queue metrics
- in this sweep, `baseline` remained the best model for:
  - average buffer utilization
  - average peak buffer utilization
  - maximum buffer utilization
- the highest return came from `buffer_light_step`, but it achieved that with worse queue behavior than baseline

## Objective

This experiment tested a new buffer-aware variant designed to reduce **peak queue occupancy**, not just average queue occupancy.

The earlier `buffer_default_step` design penalized buffer usage, but it still allowed large queue spikes. To address that, a new model called `buffer_peak_step` was added on top of the same basic design.

The main question was:

Can a peak-aware reward improve worst-case queue behavior relative to both the baseline and the earlier `buffer_default_step` model?

## Experimental Setup

All models in this experiment were trained and evaluated under the same matched budget:

- Training timesteps per model: `90112`
- Training phases: `1`
- PPO rollout / batch: `8192 / 2048`
- Training seed: `123`
- Evaluation: `20` deterministic seeded episodes
- Evaluation seed: `2123`

This was a shorter-budget comparison intended to test the new reward idea quickly and fairly across all variants.

This report supersedes any earlier expectation that the first peak-aware reward might improve queue spikes. Under this experiment, it did not.

## Models Compared

- `baseline`
- `buffer_default_step`
- `buffer_light_step`
- `buffer_strong_step`
- `buffer_default_rtt`
- `buffer_reward_only`
- `buffer_peak_step`

## Variant Configurations

| Variant | Reward Mode | Buffer Penalty | Peak Penalty | Peak Threshold | Observe Buffer | Feedback Mode | Delay |
|---|---|---:|---:|---:|---|---|---:|
| `baseline` | `baseline` | `0` | `0` | N/A | No | N/A | N/A |
| `buffer_default_step` | `buffer_penalty` | `100` | `0` | N/A | Yes | `step` | `1` |
| `buffer_light_step` | `buffer_penalty` | `25` | `0` | N/A | Yes | `step` | `1` |
| `buffer_strong_step` | `buffer_penalty` | `400` | `0` | N/A | Yes | `step` | `1` |
| `buffer_default_rtt` | `buffer_penalty` | `100` | `0` | N/A | Yes | `rtt` | `1` minimum |
| `buffer_reward_only` | `buffer_penalty` | `100` | `0` | N/A | No | N/A | N/A |
| `buffer_peak_step` | `buffer_peak_penalty` | `100` | `400` | `0.3` | Yes | `step` | `1` |

## What Changed In `buffer_peak_step`

`buffer_peak_step` keeps the same delayed buffer observation as `buffer_default_step`, but changes the reward.

Instead of only subtracting a penalty based on current buffer utilization, it also adds a **quadratic penalty on interval peak buffer utilization above a threshold**.

Conceptually:

- `buffer_default_step` tries to reduce queue usage in a general sense
- `buffer_peak_step` tries to make high-occupancy spikes much more expensive

So this new model is the first direct attempt to target **queue spikes**.

## Metrics

The models were compared on:

- **Mean Return**
- **Legacy Reward**
- **Throughput**
- **Latency**
- **Loss**
- **Average Buffer Utilization**
- **Average Peak Buffer Utilization**
- **Maximum Buffer Utilization**

The most important new metric in this experiment is **Average Peak Buffer Utilization**, because it directly reflects whether the controller reduced within-interval queue spikes.

## Results

| Model | Mean Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Avg Peak Buffer Util | Max Buffer Util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | `484.438` | `1.211` | `2207475.686` | `0.550860` | `0.038804` | `0.148993` | `0.217438` | `0.438291` |
| `buffer_default_step` | `639.706` | `1.623` | `2815954.502` | `0.637032` | `0.043472` | `0.233879` | `0.297842` | `0.577989` |
| `buffer_light_step` | `654.511` | `1.643` | `3134466.146` | `0.855309` | `0.056667` | `0.285398` | `0.383985` | `0.586741` |
| `buffer_strong_step` | `-725.265` | `-1.536` | `2826635.782` | `3.517540` | `0.187021` | `0.692781` | `0.711326` | `0.997482` |
| `buffer_default_rtt` | `453.015` | `1.175` | `2663808.361` | `0.889651` | `0.077433` | `0.427869` | `0.526449` | `0.788017` |
| `buffer_reward_only` | `191.360` | `0.546` | `2931694.968` | `1.336163` | `0.280295` | `0.679265` | `0.748934` | `0.998306` |
| `buffer_peak_step` | `492.611` | `1.375` | `2867246.072` | `0.751651` | `0.131141` | `0.477593` | `0.585988` | `0.737790` |

## Key Comparison: `buffer_peak_step` vs Baseline

Relative to baseline, `buffer_peak_step` changed as follows:

- return: `+1.69%`
- legacy reward: `+13.57%`
- throughput: `+29.89%`
- latency: `+36.45%`
- loss: `+237.95%`
- average buffer utilization: `+220.55%`
- average peak buffer utilization: `+169.50%`
- max buffer utilization: `+68.33%`

This means the new peak-aware model slightly improved reward and throughput, but **worsened every queue-related metric**.

## Key Comparison: `buffer_peak_step` vs `buffer_default_step`

Relative to `buffer_default_step`, `buffer_peak_step` changed as follows:

- return: `-22.99%`
- legacy reward: `-15.23%`
- throughput: `+1.82%`
- latency: `+17.99%`
- loss: `+201.66%`
- average buffer utilization: `+104.20%`
- average peak buffer utilization: `+96.74%`
- max buffer utilization: `+27.65%`

So compared with the earlier default step design, the new peak-aware reward was clearly worse on both:

- transport quality
- queue-control quality

## Main Findings

### 1. The new peak-aware variant did not work

The central result of this experiment is that `buffer_peak_step` did **not** improve peak queue control.

Instead, compared with baseline, it produced:

- much higher average buffer utilization
- much higher average peak buffer utilization
- much higher maximum buffer utilization

So the first attempt at spike-aware reward shaping failed under this budget.

### 2. The new reward did not beat `buffer_default_step`

The new variant was intended as an improvement over `buffer_default_step`, but it did not achieve that.

It had:

- lower return
- lower legacy reward
- higher latency
- much higher loss
- much worse average and peak queue usage

This means the added peak penalty, in its current form, did not guide the policy toward better queue behavior.

### 3. The strongest model in this sweep was still not the peak-aware one

Among the tested models, the highest return came from `buffer_light_step`, but it also had worse queue behavior than baseline.

The best queue behavior in this matched-budget run was actually the original `baseline`, which had:

- the lowest average buffer utilization
- the lowest average peak buffer utilization
- the lowest maximum buffer utilization

So under this shorter-budget experiment, none of the buffer-aware variants improved queue behavior over baseline.

### 4. What this means for the original idea

The original idea was to improve `buffer_default_step` by making queue spikes more expensive.

That did not happen here. The first peak-aware implementation did not produce a cleaner queueing policy. Instead, it increased:

- average queue occupancy
- spike-related occupancy
- worst-case occupancy

So the idea is still plausible, but this particular reward formulation is not yet the right one.

## Interpretation

The peak-aware reward idea was reasonable, but this specific implementation did not succeed.

A likely explanation is that the controller was still able to gain enough throughput to offset the extra penalty, while also suffering unstable behavior from the stronger reward shaping. In practice, that produced:

- more loss
- more latency
- more queue buildup

rather than less.

So the current peak penalty appears to be either:

- too weak in the right places, or
- too destabilizing in the wrong places

for this training budget and threshold choice.

## Conclusion

This experiment does **not** support the current `buffer_peak_step` design.

The new peak-aware reward:

- did not reduce average queueing
- did not reduce queue spikes
- did not reduce worst-case queue occupancy
- did not beat `buffer_default_step`

In short:

The first peak-focused reward variant did **not work**.

## Recommended Next Step

The next revision should keep the idea of targeting spikes, but change the reward formulation rather than simply increasing penalty strength.

The most reasonable next options are:

- penalize **max buffer utilization directly** rather than current utilization plus thresholded peak excess
- penalize **time spent above a threshold** instead of only the interval peak
- use a **smaller peak penalty weight** with a **lower threshold**, to avoid unstable overreaction

The current result suggests that spike control will need a more targeted and better-balanced objective.

## Where To See The Results

- Main report:
  - `experiments/peak_buffer_sweep_20260428/REPORT.md`
- Summary JSON:
  - `experiments/peak_buffer_sweep_20260428/summary.json`
- Evaluation files:
  - `experiments/peak_buffer_sweep_20260428/baseline/eval.json`
  - `experiments/peak_buffer_sweep_20260428/buffer_default_step/eval.json`
  - `experiments/peak_buffer_sweep_20260428/buffer_light_step/eval.json`
  - `experiments/peak_buffer_sweep_20260428/buffer_strong_step/eval.json`
  - `experiments/peak_buffer_sweep_20260428/buffer_default_rtt/eval.json`
  - `experiments/peak_buffer_sweep_20260428/buffer_reward_only/eval.json`
  - `experiments/peak_buffer_sweep_20260428/buffer_peak_step/eval.json`
