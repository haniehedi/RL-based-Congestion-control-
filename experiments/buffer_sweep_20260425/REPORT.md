# PCC-RL Buffer-Aware Training Report

## Setup

- Training budget per model: `45056` timesteps
- PPO rollout / batch: `4096 / 512`
- Training phases: `1`
- Training seed: `123`
- Evaluation: `20` seeded episodes, deterministic policy
- Baseline model path: `experiments/baseline_20260425/model.zip`

## Buffer-Aware Sweep

| Model | Flags | Mean Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Max Buffer Util |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `buffer_default_step` | `weight=100, observe=true, mode=step, delay=1` | 453.235 | 1.196 | 3151324.932 | 0.926929 | 0.251781 | 0.625265 | 0.961745 |
| `buffer_light_step` | `weight=25, observe=true, mode=step, delay=1` | 575.731 | 1.446 | 2903845.881 | 0.875088 | 0.049325 | 0.272259 | 0.845308 |
| `buffer_strong_step` | `weight=400, observe=true, mode=step, delay=1` | 94.758 | 0.341 | 2773339.398 | 1.814417 | 0.077913 | 0.259942 | 0.664241 |
| `buffer_default_rtt` | `weight=100, observe=true, mode=rtt, delay=1` | -483.404 | -1.139 | 3467839.421 | 3.470530 | 0.279143 | 0.695611 | 0.955194 |
| `buffer_reward_only` | `weight=100, observe=false` | 515.376 | 1.306 | 2334222.815 | 0.523411 | 0.057948 | 0.174378 | 0.300077 |

## Baseline

| Model | Mean Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Max Buffer Util |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 478.787 | 1.197 | 3231846.369 | 1.182749 | 0.156744 | 0.428736 | 0.668766 |

## Comparison Against Baseline

| Model | Return vs Baseline | Throughput vs Baseline | Latency vs Baseline | Loss vs Baseline | Avg Buffer Util vs Baseline |
|---|---:|---:|---:|---:|---:|
| `buffer_default_step` | -5.34% | -2.49% | -21.63% | +60.63% | +45.84% |
| `buffer_light_step` | +20.25% | -10.15% | -26.01% | -68.53% | -36.50% |
| `buffer_strong_step` | -80.21% | -14.19% | +53.41% | -50.29% | -39.37% |
| `buffer_default_rtt` | -200.96% | +7.30% | +193.43% | +78.09% | +62.25% |
| `buffer_reward_only` | +7.64% | -27.77% | -55.75% | -63.03% | -59.33% |

## Conclusions

1. The best observation-enabled configuration in this budget was `buffer_light_step`.
   It improved mean return and legacy reward, while reducing mean latency, mean loss, and mean average buffer utilization relative to baseline.

2. The strongest raw buffer-control result was `buffer_reward_only`.
   It achieved the largest reduction in average and peak buffer utilization, but paid for it with a large throughput drop.

3. The original default recommendation `buffer_default_step` was not the best setting.
   Under this budget, `weight=100` with delayed observation was too aggressive and increased buffer usage and loss.

4. `buffer_default_rtt` performed poorly.
   The RTT-delayed observation path appears harder to train and likely needs either a smaller penalty weight, more training budget, or both.

## Recommendation

- If you want the best result **while exposing delayed buffer utilization to the agent**, use:
  - `--reward-mode=buffer_penalty`
  - `--buffer-penalty-weight=25.0`
  - `--observe-buffer-util=true`
  - `--buffer-feedback-mode=step`
  - `--buffer-feedback-delay-steps=1`

- If you want the strongest queue/buffer suppression regardless of throughput cost, use:
  - `--reward-mode=buffer_penalty`
  - `--buffer-penalty-weight=100.0`
  - `--observe-buffer-util=false`

## Artifacts

- Sweep summary: `experiments/buffer_sweep_20260425/summary.json`
- Baseline eval: `experiments/baseline_20260425/eval.json`
- Per-run logs and models are stored under each experiment directory.
