# Full-Budget Buffer-Aware vs Baseline Report

## Setup

- Training budget per phase: `656000` timesteps
- Training phases: `6`
- Total training timesteps per model: `3,936,000`
- PPO rollout / batch: `8192 / 2048`
- Training seed: `123`
- Evaluation: `20` deterministic seeded episodes
- Evaluation seed: `2123`

## Models Compared

### Buffer-aware

- Flags:
  - `--reward-mode=buffer_penalty`
  - `--buffer-penalty-weight=25.0`
  - `--observe-buffer-util=true`
  - `--buffer-feedback-mode=step`
  - `--buffer-feedback-delay-steps=1`
- Model:
  - `experiments/full_budget_buffer_light_20260425/model.zip`

### Baseline

- Flags:
  - `--reward-mode=baseline`
  - `--observe-buffer-util=false`
- Model:
  - `experiments/full_budget_baseline_20260425/model.zip`

## Evaluation Summary

| Model | Mean Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Max Buffer Util |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buffer-aware | `833.255` | `2.091` | `3344617.853` | `0.637848` | `0.029153` | `0.315580` | `0.699951` |
| Baseline | `552.391` | `1.381` | `2486795.953` | `0.610534` | `0.040409` | `0.238540` | `0.432300` |

## Relative Comparison

Buffer-aware vs baseline:

- Mean return: `+50.86%`
- Legacy reward: `+51.41%`
- Throughput: `+34.50%`
- Latency: `+4.47%` worse
- Loss: `-27.86%` lower
- Average buffer utilization: `+32.30%` worse
- Max buffer utilization: `+61.91%` worse

## Interpretation

The full-budget run changed the tradeoff relative to the short-budget stage.

- The buffer-aware model clearly beat the baseline on the optimization objective:
  - much higher mean return
  - much higher legacy reward
  - materially higher throughput
  - lower packet loss
- But it did **not** beat the baseline on queue usage:
  - higher average buffer utilization
  - much higher peak buffer utilization
  - slightly higher latency

So the second-stage result is:

- If the question is "did the buffer-aware model outperform baseline overall on reward and throughput?" the answer is **yes**.
- If the question is "did it achieve lower switch-buffer usage than baseline at full budget?" the answer is **no**.

## Bottom Line

At full default training budget, the chosen buffer-aware configuration (`buffer_light_step`) learned a stronger, more aggressive policy than baseline. It improved return, throughput, and loss, but it did so while using **more** buffer on average and at peak.

That means this specific reward shaping setup is **not yet sufficient** if your primary goal is to reduce switch buffer occupancy. It is helping the policy optimize a better overall transport objective, but not enforcing queue restraint strongly enough over long training.

## Where To See The Results

- Full-budget report:
  - `experiments/full_budget_comparison_20260426.md`
- Buffer-aware evaluation JSON:
  - `experiments/full_budget_buffer_light_20260425/eval.json`
- Baseline evaluation JSON:
  - `experiments/full_budget_baseline_20260425/eval.json`
- Buffer-aware model:
  - `experiments/full_budget_buffer_light_20260425/model.zip`
- Baseline model:
  - `experiments/full_budget_baseline_20260425/model.zip`
