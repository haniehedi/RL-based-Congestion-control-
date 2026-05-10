import inspect
import json
import os
import subprocess
import sys
from datetime import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
repo_root = os.path.dirname(os.path.dirname(currentdir))

sys.path.insert(0, os.path.dirname(currentdir))
from common.simple_arg_parse import arg_or_default


EXPERIMENTS = [
    {
        "name": "baseline",
        "flags": {
            "--reward-mode": "baseline",
            "--observe-buffer-util": "false",
        },
    },
    {
        "name": "buffer_default_step",
        "flags": {
            "--reward-mode": "buffer_penalty",
            "--buffer-penalty-weight": "100.0",
            "--observe-buffer-util": "true",
            "--buffer-feedback-mode": "step",
            "--buffer-feedback-delay-steps": "1",
        },
    },
    {
        "name": "buffer_light_step",
        "flags": {
            "--reward-mode": "buffer_penalty",
            "--buffer-penalty-weight": "25.0",
            "--observe-buffer-util": "true",
            "--buffer-feedback-mode": "step",
            "--buffer-feedback-delay-steps": "1",
        },
    },
    {
        "name": "buffer_strong_step",
        "flags": {
            "--reward-mode": "buffer_penalty",
            "--buffer-penalty-weight": "400.0",
            "--observe-buffer-util": "true",
            "--buffer-feedback-mode": "step",
            "--buffer-feedback-delay-steps": "1",
        },
    },
    {
        "name": "buffer_default_rtt",
        "flags": {
            "--reward-mode": "buffer_penalty",
            "--buffer-penalty-weight": "100.0",
            "--observe-buffer-util": "true",
            "--buffer-feedback-mode": "rtt",
            "--buffer-feedback-delay-steps": "1",
        },
    },
    {
        "name": "buffer_reward_only",
        "flags": {
            "--reward-mode": "buffer_penalty",
            "--buffer-penalty-weight": "100.0",
            "--observe-buffer-util": "false",
        },
    },
    {
        "name": "buffer_peak_step",
        "flags": {
            "--reward-mode": "buffer_peak_penalty",
            "--buffer-penalty-weight": "100.0",
            "--peak-buffer-penalty-weight": "400.0",
            "--peak-buffer-threshold": "0.3",
            "--observe-buffer-util": "true",
            "--buffer-feedback-mode": "step",
            "--buffer-feedback-delay-steps": "1",
        },
    },
]


def run_command(command, log_path):
    with open(log_path, "w") as log_file:
        subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def build_flag_list(flag_dict):
    result = []
    for key, value in flag_dict.items():
        result.append("%s=%s" % (key, value))
    return result


def render_table_rows(summary):
    rows = []
    for experiment in summary["experiments"]:
        metrics = experiment["metrics"]
        rows.append(
            "| %s | `%0.3f` | `%0.3f` | `%0.3f` | `%0.6f` | `%0.6f` | `%0.6f` | `%0.6f` | `%0.6f` |"
            % (
                experiment["name"],
                metrics["mean_episode_return"],
                metrics["mean_legacy_reward"],
                metrics["mean_avg_throughput"],
                metrics["mean_avg_latency"],
                metrics["mean_avg_loss"],
                metrics["mean_avg_buffer_utilization"],
                metrics["mean_avg_peak_buffer_utilization"],
                metrics["mean_max_buffer_utilization"],
            )
        )
    return "\n".join(rows)


def render_relative_rows(summary):
    baseline = None
    for experiment in summary["experiments"]:
        if experiment["name"] == "baseline":
            baseline = experiment["metrics"]
            break
    rows = []
    for experiment in summary["experiments"]:
        if experiment["name"] == "baseline":
            continue
        metrics = experiment["metrics"]
        rows.append(
            "| %s | `%+.2f%%` | `%+.2f%%` | `%+.2f%%` | `%+.2f%%` | `%+.2f%%` | `%+.2f%%` | `%+.2f%%` |"
            % (
                experiment["name"],
                100.0 * (metrics["mean_episode_return"] - baseline["mean_episode_return"]) / baseline["mean_episode_return"],
                100.0 * (metrics["mean_avg_throughput"] - baseline["mean_avg_throughput"]) / baseline["mean_avg_throughput"],
                100.0 * (metrics["mean_avg_latency"] - baseline["mean_avg_latency"]) / baseline["mean_avg_latency"],
                100.0 * (metrics["mean_avg_loss"] - baseline["mean_avg_loss"]) / baseline["mean_avg_loss"],
                100.0 * (metrics["mean_avg_buffer_utilization"] - baseline["mean_avg_buffer_utilization"]) / baseline["mean_avg_buffer_utilization"],
                100.0 * (metrics["mean_avg_peak_buffer_utilization"] - baseline["mean_avg_peak_buffer_utilization"]) / baseline["mean_avg_peak_buffer_utilization"],
                100.0 * (metrics["mean_max_buffer_utilization"] - baseline["mean_max_buffer_utilization"]) / baseline["mean_max_buffer_utilization"],
            )
        )
    return "\n".join(rows)


def write_report(output_dir, summary):
    report_path = os.path.join(output_dir, "REPORT.md")
    lines = [
        "# Full-Budget Suite Report",
        "",
        "## Setup",
        "",
        "- Training budget per phase: `%d` timesteps" % summary["train_timesteps"],
        "- Training phases: `%d`" % summary["num_models"],
        "- Total training timesteps per model: `%d`" % (summary["train_timesteps"] * summary["num_models"]),
        "- PPO rollout / batch: `%d / %d`" % (summary["n_steps"], summary["batch_size"]),
        "- Training seed: `%d`" % summary["seed"],
        "- Evaluation episodes: `%d`" % summary["eval_episodes"],
        "",
        "## Results",
        "",
        "| Model | Mean Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Avg Peak Buffer Util | Max Buffer Util |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        render_table_rows(summary),
        "",
        "## Relative To Baseline",
        "",
        "| Model | Return | Throughput | Latency | Loss | Avg Buffer Util | Avg Peak Buffer Util | Max Buffer Util |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        render_relative_rows(summary),
        "",
        "## Artifacts",
        "",
        "- Summary JSON: `%s`" % os.path.join(output_dir, "summary.json"),
        "- Each model directory contains `model.zip`, `train.log`, `eval.log`, and `eval.json`.",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    python_executable = sys.executable
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = arg_or_default("--output-dir", default=os.path.join(repo_root, "experiments", "full_budget_suite_%s" % timestamp))
    train_timesteps = arg_or_default("--train-timesteps", default=656000)
    n_steps = arg_or_default("--n-steps", default=8192)
    batch_size = arg_or_default("--batch-size", default=2048)
    num_models = arg_or_default("--num-models", default=6)
    eval_episodes = arg_or_default("--eval-episodes", default=20)
    seed = arg_or_default("--seed", default=123)
    eval_seed = arg_or_default("--eval-seed", default=2123)

    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "output_dir": os.path.abspath(output_dir),
        "train_timesteps": train_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "num_models": num_models,
        "eval_episodes": eval_episodes,
        "seed": seed,
        "eval_seed": eval_seed,
        "experiments": [],
    }

    print("Writing full-budget suite outputs to %s" % os.path.abspath(output_dir), flush=True)

    for experiment in EXPERIMENTS:
        exp_dir = os.path.join(output_dir, experiment["name"])
        os.makedirs(exp_dir, exist_ok=True)
        model_path = os.path.join(exp_dir, "model.zip")
        train_log = os.path.join(exp_dir, "train.log")
        eval_log = os.path.join(exp_dir, "eval.log")
        eval_json = os.path.join(exp_dir, "eval.json")

        train_command = [
            python_executable,
            os.path.join(repo_root, "src", "gym", "stable_solve.py"),
            "--timesteps=%d" % train_timesteps,
            "--n-steps=%d" % n_steps,
            "--batch-size=%d" % batch_size,
            "--num-models=%d" % num_models,
            "--seed=%d" % seed,
            "--skip-checkpoints",
            "--model-dir=%s" % model_path,
        ] + build_flag_list(experiment["flags"])

        print("[train] %s" % experiment["name"], flush=True)
        run_command(train_command, train_log)

        eval_command = [
            python_executable,
            os.path.join(repo_root, "src", "gym", "evaluate_model.py"),
            "--model-path=%s" % model_path,
            "--output-path=%s" % eval_json,
            "--episodes=%d" % eval_episodes,
            "--seed=%d" % eval_seed,
        ] + build_flag_list(experiment["flags"])

        print("[eval] %s" % experiment["name"], flush=True)
        run_command(eval_command, eval_log)

        with open(eval_json) as f:
            eval_result = json.load(f)

        summary["experiments"].append({
            "name": experiment["name"],
            "flags": experiment["flags"],
            "model_path": model_path,
            "train_log": train_log,
            "eval_log": eval_log,
            "eval_json": eval_json,
            "metrics": eval_result,
        })

        print(
            "[done] %s return=%0.3f legacy=%0.3f avg_buffer=%0.6f"
            % (
                experiment["name"],
                eval_result["mean_episode_return"],
                eval_result["mean_legacy_reward"],
                eval_result["mean_avg_buffer_utilization"],
            ),
            flush=True,
        )

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    write_report(output_dir, summary)
    print("Finished full-budget suite. Summary: %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
