# Full-Budget Variant Comparison Report

## Objective

The goal of this study was to determine whether a reinforcement-learning congestion controller can be improved by making it aware of switch buffer usage.

The original baseline controller learns only from standard transport signals such as throughput, latency, and loss. The proposed variants extend this design by incorporating buffer-awareness into the reward, and in some cases also into the observation seen by the agent.

The main question was:

Can a buffer-aware RL congestion controller outperform the original baseline while also reducing switch buffer occupancy?

## Experimental Setup

All models were trained under the same full-budget setting:

- Training budget per phase: `656000` timesteps
- Training phases: `6`
- Total training timesteps per model: `3936000`
- PPO rollout / batch: `8192 / 2048`
- Training seed: `123`
- Evaluation: `20` deterministic seeded episodes
- Evaluation seed: `2123`

This ensured that all models were compared fairly under identical training and evaluation conditions.

## Models Compared

The experiment compared six models:

- `baseline`
- `buffer_default_step`
- `buffer_light_step`
- `buffer_strong_step`
- `buffer_default_rtt`
- `buffer_reward_only`

These variants differ in how buffer-awareness is incorporated:

- how strongly buffer usage is penalized
- whether the agent can observe delayed buffer utilization
- how that delay is modeled

## Baseline Design

### Baseline Reward Model

The baseline reward combines three transport objectives:

- higher throughput is rewarded
- higher latency is penalized
- higher packet loss is penalized

So the baseline controller is trained to maximize delivery performance while keeping delay and packet drops low.

Importantly, the baseline reward does **not** explicitly penalize switch buffer occupancy.

### Baseline Observation Space

The baseline observation consists of a short history of transport-level signals. The agent observes:

- latency inflation
- latency ratio
- send ratio

These features summarize how current network behavior compares to recent behavior and minimum-path conditions. They allow the controller to infer congestion indirectly from transport statistics, but they do not give it direct queue information.

In short:

- **baseline reward** = throughput reward - latency penalty - loss penalty
- **baseline observation** = recent transport-level features only

## Proposed Buffer-Aware Design

### Proposed Reward Model

The proposed reward starts from the same baseline objective and adds a penalty term for switch buffer utilization.

So compared with the baseline:

- throughput is still rewarded
- latency is still penalized
- loss is still penalized
- buffer usage is now also penalized

This means the new controller is explicitly trained to avoid queue buildup instead of relying only on latency and loss as indirect signals.

### Proposed Observation Space

The proposed observation space keeps the original transport-level features, but can optionally add:

- delayed buffer utilization

This delay is important because in a realistic system the sender would not know switch queue occupancy instantly. Instead, queue information would arrive only after some feedback delay, for example through the receiver path.

So the proposed design has two levels:

- **baseline observation** = transport-only signals
- **proposed observation** = transport signals + delayed buffer signal

### Legacy Reward vs Proposed Reward

Two reward notions are important in this study:

- **Legacy reward**: the original baseline objective, used as a common comparison metric across all models
- **Proposed reward**: the modified training objective that includes a buffer-utilization penalty

This distinction matters because it tells us whether a new buffer-aware controller is only better under its own modified objective, or whether it also improves under the original project objective.

### Buffer-Aware Variant Configurations

| Variant | Buffer Penalty Weight | Observe Buffer Utilization | Feedback Mode | Delay Setting | High-Level Meaning |
|---|---:|---|---|---|---|
| `buffer_default_step` | `100` | Yes | `step` | `1` step | Default queue-aware controller with fixed delayed queue feedback |
| `buffer_light_step` | `25` | Yes | `step` | `1` step | Same as default step, but with a weaker queue penalty |
| `buffer_strong_step` | `400` | Yes | `step` | `1` step | Same as default step, but with a much stronger queue penalty |
| `buffer_default_rtt` | `100` | Yes | `rtt` | minimum `1` step, RTT-shaped delay | Queue-aware controller with more realistic RTT-like delayed queue feedback |
| `buffer_reward_only` | `100` | No | N/A | N/A | Queue-aware reward, but the agent does not directly observe queue utilization |

The only difference between these variants is how queue-awareness is introduced:

- penalty strength
- whether queue state is exposed to the agent
- whether delay is modeled as fixed-step or RTT-like

## Results

| Model | Mean Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Max Buffer Util |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | `552.391` | `1.381` | `2486795.953` | `0.610534` | `0.040409` | `0.238540` | `0.432300` |
| `buffer_default_step` | `963.272` | `2.427` | `3764123.563` | `0.652018` | `0.029009` | `0.185525` | `0.583296` |
| `buffer_light_step` | `833.255` | `2.091` | `3344617.853` | `0.637848` | `0.029153` | `0.315580` | `0.699951` |
| `buffer_strong_step` | `793.114` | `2.058` | `3218861.797` | `0.567584` | `0.028175` | `0.189168` | `0.718003` |
| `buffer_default_rtt` | `780.024` | `1.989` | `3283205.060` | `0.650541` | `0.048395` | `0.386127` | `0.713102` |
| `buffer_reward_only` | `713.974` | `1.821` | `3102276.558` | `0.647135` | `0.058749` | `0.356625` | `0.585050` |

## Relative To Baseline

| Model | Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Max Buffer Util |
|---|---:|---:|---:|---:|---:|---:|---:|
| `buffer_default_step` | `+74.38%` | `+75.73%` | `+51.36%` | `+6.79%` | `-28.21%` | `-22.23%` | `+34.93%` |
| `buffer_light_step` | `+50.85%` | `+51.42%` | `+34.50%` | `+4.47%` | `-27.86%` | `+32.30%` | `+61.91%` |
| `buffer_strong_step` | `+43.58%` | `+49.06%` | `+29.44%` | `-7.03%` | `-30.28%` | `-20.70%` | `+66.09%` |
| `buffer_default_rtt` | `+41.21%` | `+44.00%` | `+32.03%` | `+6.55%` | `+19.76%` | `+61.87%` | `+64.96%` |
| `buffer_reward_only` | `+29.25%` | `+31.83%` | `+24.75%` | `+5.99%` | `+45.38%` | `+49.50%` | `+35.33%` |

## Main Findings

### 1. Best Overall Model

The strongest model in this full-budget comparison was `buffer_default_step`.

It achieved:

- the highest mean return
- the highest legacy reward
- the highest throughput

So under both the new optimization objective and the original baseline objective, it was the best-performing model.

### 2. Average Buffer Usage Improved

`buffer_default_step` also improved the main queue metric on average:

- baseline average buffer utilization: `0.238540`
- `buffer_default_step` average buffer utilization: `0.185525`

This corresponds to a reduction of about `22.23%`.

This is an important result, because it shows that a well-tuned buffer-aware controller can improve transport performance while also reducing average switch buffer occupancy.

### 3. Peak Buffer Usage Did Not Improve

Although `buffer_default_step` improved average buffer usage, it did not improve the peak case:

- baseline max buffer utilization: `0.432300`
- `buffer_default_step` max buffer utilization: `0.583296`

So the controller reduced average queueing, but still allowed larger queue spikes than the baseline.

### 4. Other Variants

- `buffer_strong_step` also reduced average buffer utilization versus baseline and improved latency/loss, but it was weaker than `buffer_default_step` on return and throughput.
- `buffer_light_step`, which looked best in the earlier short-budget sweep, did not remain the strongest option at full budget.
- `buffer_default_rtt` and `buffer_reward_only` were not good queue-control choices in this experiment, because both worsened average and peak buffer usage relative to baseline.

## Interpretation

The results show that buffer-awareness is useful, but only if it is tuned carefully.

The study suggests:

- weak penalties are not enough
- some delayed-feedback designs are too noisy or ineffective
- a stronger but still balanced queue penalty performs better

This means the basic idea is sound: explicitly accounting for queue occupancy can improve the controller.

However, the remaining weakness is clear:

- the current design improves **average** queue behavior
- it does not yet control **peak** queue buildup well enough

## Final Conclusion

This experiment supports four main conclusions:

1. A buffer-aware RL congestion controller can outperform the original baseline.
2. It is possible to improve both transport performance and average queue occupancy at the same time.
3. A delayed queue signal can be useful when combined with the right reward strength.
4. Peak queue control remains an open problem in the current design.

So the proposed method is promising, but not yet complete.

## Recommendation

If the goal is the best current tradeoff between transport performance and lower **average** switch buffer usage, the recommended configuration is:

- `--reward-mode=buffer_penalty`
- `--buffer-penalty-weight=100.0`
- `--observe-buffer-util=true`
- `--buffer-feedback-mode=step`
- `--buffer-feedback-delay-steps=1`

This is the `buffer_default_step` configuration.

Why this is the recommended model:

- best overall return
- best legacy reward
- best throughput
- lower loss than baseline
- lower average buffer utilization than baseline

Main caveat:

- it still increases peak buffer utilization relative to baseline
- if the real requirement is strict control of worst-case queue spikes, more targeted reward shaping is still needed

## Suggested Next Step

The next improvement should focus specifically on peak queue occupancy.

The current reward is able to reduce average queueing, but it is not punishing large queue spikes strongly enough. A future design should therefore place more emphasis on high-occupancy events, not just mean occupancy.

## Where To See The Results

- This report:
  - `experiments/full_budget_suite_20260426/REPORT.md`
- Baseline evaluation:
  - `experiments/full_budget_suite_20260426/baseline/eval.json`
- Buffer-aware evaluations:
  - `experiments/full_budget_suite_20260426/buffer_default_step/eval.json`
  - `experiments/full_budget_suite_20260426/buffer_light_step/eval.json`
  - `experiments/full_budget_suite_20260426/buffer_strong_step/eval.json`
  - `experiments/full_budget_suite_20260426/buffer_default_rtt/eval.json`
  - `experiments/full_budget_suite_20260426/buffer_reward_only/eval.json`
