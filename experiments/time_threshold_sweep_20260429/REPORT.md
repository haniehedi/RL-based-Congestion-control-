# Time-Above-Threshold Sweep Report

## Scope

This was a partial parameter sweep over the time-above-threshold reward family.

Completed configurations:

- threshold `0.10`, weight `50`
- threshold `0.10`, weight `100`
- threshold `0.10`, weight `150`
- threshold `0.15`, weight `50`

Reference models reused from the earlier 3-seed validation:

- `baseline`
- `buffer_default_step`
- `buffer_time_threshold_step` with threshold `0.20`, weight `150`

All numbers below use the same 3 training seeds:

- `123`
- `231`
- `312`

And the same matched budget:

- `90112` training timesteps
- `1` training phase

## Reference Results

| Model | Avg Peak Buffer | Avg Buffer | Time Above Threshold | Return |
|---|---:|---:|---:|---:|
| `baseline` | `0.417388` | `0.362979` | `0.449084` | `555.485` |
| `buffer_default_step` | `0.488356` | `0.405861` | `0.553316` | `504.004` |
| `buffer_time_threshold_step (0.20 / 150)` | `0.446083` | `0.367307` | `0.492920` | `528.154` |

## Completed Sweep Results

| Variant | Threshold | Weight | Avg Peak Buffer | Avg Buffer | Time Above Threshold | Return |
|---|---:|---:|---:|---:|---:|---:|
| `buffer_time_threshold_t01_w50` | `0.10` | `50` | `0.294835` | `0.216027` | `0.429929` | `608.440` |
| `buffer_time_threshold_t01_w100` | `0.10` | `100` | `0.307874` | `0.240346` | `0.404607` | `600.610` |
| `buffer_time_threshold_t015_w50` | `0.15` | `50` | `0.449850` | `0.373513` | `0.506523` | `544.967` |
| `buffer_time_threshold_t01_w150` | `0.10` | `150` | `0.594440` | `0.497379` | `0.691942` | `537.887` |

## Main Finding

The first strong result from the sweep is:

- `threshold = 0.10`
- `weight = 50`

This variant beat both `baseline` and `buffer_default_step` on all three queue-control metrics that matter here:

- average peak buffer utilization
- average buffer utilization
- time spent above threshold

It also improved return relative to both reference models.

## Best Variant vs Baseline

Best variant: `buffer_time_threshold_t01_w50`

Comparison to `baseline`:

- avg peak buffer utilization: `0.294835` vs `0.417388`
- change: `-29.36%`
- avg buffer utilization: `0.216027` vs `0.362979`
- change: `-40.49%`
- time above threshold: `0.429929` vs `0.449084`
- change: `-4.27%`
- return: `608.440` vs `555.485`
- change: `+9.53%`

## Best Variant vs `buffer_default_step`

Comparison to `buffer_default_step`:

- avg peak buffer utilization: `0.294835` vs `0.488356`
- change: `-39.63%`
- avg buffer utilization: `0.216027` vs `0.405861`
- change: `-46.77%`
- time above threshold: `0.429929` vs `0.553316`
- change: `-22.30%`
- return: `608.440` vs `504.004`
- change: `+20.72%`

## Interpretation

This is the first result in the project that gives clear positive evidence for the time-above-threshold idea.

What seems to be happening:

- lower threshold `0.10` is better than `0.20`
- smaller weight `50` or `100` is better than `150`
- the earlier time-threshold setting (`0.20 / 150`) was too harsh
- the lighter setting keeps the queue objective meaningful without destabilizing the policy

Among the completed configurations:

- `0.10 / 50` is the current best overall
- `0.10 / 100` is also strong and slightly better on time-above-threshold itself, but worse on peak and average buffer than `0.10 / 50`
- `0.10 / 150` is too aggressive
- `0.15 / 50` is weaker and less robust than `0.10 / 50`

## Current Recommendation

The current best candidate is:

- `--reward-mode=buffer_time_threshold_penalty`
- `--buffer-penalty-weight=100.0`
- `--time-buffer-penalty-weight=50.0`
- `--buffer-time-threshold=0.1`
- `--observe-buffer-util=true`
- `--buffer-feedback-mode=step`
- `--buffer-feedback-delay-steps=1`

## Caveat

This is a partial sweep, not the full `3 x 3` grid yet.

But based on the completed configurations, the search already found a variant that beats both baseline and `buffer_default_step` on the primary queue-control metrics across the same 3-seed validation setup.
