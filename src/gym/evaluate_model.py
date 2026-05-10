import inspect
import json
import os
import statistics
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(parentdir, ".mplconfig"))
sys.path.insert(0, parentdir)

try:
    import gymnasium as gym
except ImportError:
    import gym

import network_sim

from stable_baselines3 import PPO

from common.simple_arg_parse import arg_or_default


BYTES_PER_PACKET = 1500
REWARD_SCALE = 0.001


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(statistics.mean(values))


def stdev_or_zero(values):
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def summarize_events(events):
    if not events:
        return {
            "avg_throughput": 0.0,
            "avg_latency": 0.0,
            "avg_loss": 0.0,
            "avg_buffer_utilization": 0.0,
            "avg_peak_buffer_utilization": 0.0,
            "avg_buffer_time_above_threshold_fraction": 0.0,
            "avg_observed_buffer_utilization": 0.0,
            "max_buffer_utilization": 0.0,
            "legacy_reward": 0.0,
        }

    throughputs = [float(event["Throughput"]) for event in events]
    latencies = [float(event["Latency"]) for event in events]
    losses = [float(event["Loss Rate"]) for event in events]
    buffer_utils = [float(event.get("Buffer Utilization", 0.0)) for event in events]
    peak_buffer_utils = [float(event.get("Peak Buffer Utilization", event.get("Buffer Utilization", 0.0))) for event in events]
    buffer_time_threshold_fracs = [float(event.get("Buffer Time Above Threshold Fraction", 0.0)) for event in events]
    observed_buffer_utils = [float(event.get("Observed Buffer Utilization", 0.0)) for event in events]

    avg_throughput = mean_or_zero(throughputs)
    avg_latency = mean_or_zero(latencies)
    avg_loss = mean_or_zero(losses)
    avg_buffer_util = mean_or_zero(buffer_utils)

    legacy_reward = REWARD_SCALE * (
        10.0 * avg_throughput / (8 * BYTES_PER_PACKET)
        - 1e3 * avg_latency
        - 2e3 * avg_loss
    )

    return {
        "avg_throughput": avg_throughput,
        "avg_latency": avg_latency,
        "avg_loss": avg_loss,
        "avg_buffer_utilization": avg_buffer_util,
        "avg_peak_buffer_utilization": mean_or_zero(peak_buffer_utils),
        "avg_buffer_time_above_threshold_fraction": mean_or_zero(buffer_time_threshold_fracs),
        "avg_observed_buffer_utilization": mean_or_zero(observed_buffer_utils),
        "max_buffer_utilization": max(buffer_utils) if buffer_utils else 0.0,
        "legacy_reward": legacy_reward,
    }


def main():
    model_path = arg_or_default("--model-path", default="")
    output_path = arg_or_default("--output-path", default="")
    episodes = arg_or_default("--episodes", default=20)
    seed = arg_or_default("--seed", default=123)
    deterministic = arg_or_default("--deterministic", default=True)

    if model_path == "":
        raise ValueError("--model-path is required")

    model = PPO.load(model_path)
    env = gym.make("PccNs-v0")

    episode_returns = []
    episode_lengths = []
    episode_summaries = []

    for episode_idx in range(episodes):
        reset_result = env.reset(seed=seed + episode_idx)
        if isinstance(reset_result, tuple):
            obs, _info = reset_result
        else:
            obs = reset_result
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not done and not truncated:
            action, _state = model.predict(obs, deterministic=deterministic)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, _info = step_result
            else:
                obs, reward, done, _info = step_result
                truncated = False
            episode_return += float(reward)
            episode_length += 1

        events = list(env.unwrapped.event_record["Events"])
        episode_summary = summarize_events(events)
        episode_summary["episode_return"] = episode_return
        episode_summary["episode_length"] = episode_length

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_summaries.append(episode_summary)

    result = {
        "model_path": os.path.abspath(model_path),
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "mean_episode_return": mean_or_zero(episode_returns),
        "stdev_episode_return": stdev_or_zero(episode_returns),
        "mean_episode_length": mean_or_zero(episode_lengths),
        "mean_avg_throughput": mean_or_zero([item["avg_throughput"] for item in episode_summaries]),
        "mean_avg_latency": mean_or_zero([item["avg_latency"] for item in episode_summaries]),
        "mean_avg_loss": mean_or_zero([item["avg_loss"] for item in episode_summaries]),
        "mean_avg_buffer_utilization": mean_or_zero([item["avg_buffer_utilization"] for item in episode_summaries]),
        "mean_avg_peak_buffer_utilization": mean_or_zero([item["avg_peak_buffer_utilization"] for item in episode_summaries]),
        "mean_avg_buffer_time_above_threshold_fraction": mean_or_zero([item["avg_buffer_time_above_threshold_fraction"] for item in episode_summaries]),
        "mean_avg_observed_buffer_utilization": mean_or_zero([item["avg_observed_buffer_utilization"] for item in episode_summaries]),
        "mean_max_buffer_utilization": mean_or_zero([item["max_buffer_utilization"] for item in episode_summaries]),
        "mean_legacy_reward": mean_or_zero([item["legacy_reward"] for item in episode_summaries]),
        "episode_summaries": episode_summaries,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
