import inspect
import json
import os
import subprocess
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
repo_root = os.path.dirname(os.path.dirname(currentdir))

sys.path.insert(0, os.path.dirname(currentdir))
from common.simple_arg_parse import arg_or_default


REFERENCE_SUMMARY = arg_or_default(
    "--reference-summary",
    default=os.path.join(repo_root, "experiments", "peak_buffer_sweep_20260428", "summary.json"),
)


NEW_EXPERIMENTS = [
    {
        "name": "buffer_peak_linear_step",
        "flags": {
            "--reward-mode": "buffer_peak_penalty",
            "--buffer-penalty-weight": "100.0",
            "--peak-buffer-penalty-weight": "100.0",
            "--peak-buffer-threshold": "0.2",
            "--peak-buffer-penalty-shape": "linear",
            "--observe-buffer-util": "true",
            "--buffer-feedback-mode": "step",
            "--buffer-feedback-delay-steps": "1",
        },
    },
    {
        "name": "buffer_peak_linear_low_threshold_step",
        "flags": {
            "--reward-mode": "buffer_peak_penalty",
            "--buffer-penalty-weight": "100.0",
            "--peak-buffer-penalty-weight": "60.0",
            "--peak-buffer-threshold": "0.1",
            "--peak-buffer-penalty-shape": "linear",
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


def load_reference_summary():
    with open(REFERENCE_SUMMARY) as f:
        return json.load(f)


def render_rows(experiments):
    rows = []
    for experiment in experiments:
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


def percent_change(new_value, base_value):
    if base_value == 0.0:
        return 0.0
    return 100.0 * (new_value - base_value) / base_value


def write_report(output_dir, summary):
    report_path = os.path.join(output_dir, "REPORT.md")
    baseline = next(experiment["metrics"] for experiment in summary["experiments"] if experiment["name"] == "baseline")
    default_step = next(experiment["metrics"] for experiment in summary["experiments"] if experiment["name"] == "buffer_default_step")
    old_peak = next(experiment["metrics"] for experiment in summary["experiments"] if experiment["name"] == "buffer_peak_step")

    lines = [
        "# Peak Reward Improvement Report",
        "",
        "## Setup",
        "",
        "- Training timesteps per model: `%d`" % summary["train_timesteps"],
        "- Training phases: `%d`" % summary["num_models"],
        "- PPO rollout / batch: `%d / %d`" % (summary["n_steps"], summary["batch_size"]),
        "- Training seed: `%d`" % summary["seed"],
        "- Evaluation episodes: `%d`" % summary["eval_episodes"],
        "- Evaluation seed: `%d`" % summary["eval_seed"],
        "",
        "This report combines the previous matched-budget sweep with newly trained peak-aware variants.",
        "",
        "## Results",
        "",
        "| Model | Mean Return | Legacy Reward | Throughput | Latency | Loss | Avg Buffer Util | Avg Peak Buffer Util | Max Buffer Util |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        render_rows(summary["experiments"]),
        "",
        "## New Variants",
        "",
    ]

    for experiment in summary["experiments"]:
        if experiment["name"] in {"buffer_peak_linear_step", "buffer_peak_linear_low_threshold_step"}:
            metrics = experiment["metrics"]
            lines.extend([
                "### `%s`" % experiment["name"],
                "",
                "- vs baseline avg peak buffer util: `%+.2f%%`" % percent_change(metrics["mean_avg_peak_buffer_utilization"], baseline["mean_avg_peak_buffer_utilization"]),
                "- vs baseline avg buffer util: `%+.2f%%`" % percent_change(metrics["mean_avg_buffer_utilization"], baseline["mean_avg_buffer_utilization"]),
                "- vs `buffer_default_step` avg peak buffer util: `%+.2f%%`" % percent_change(metrics["mean_avg_peak_buffer_utilization"], default_step["mean_avg_peak_buffer_utilization"]),
                "- vs original `buffer_peak_step` avg peak buffer util: `%+.2f%%`" % percent_change(metrics["mean_avg_peak_buffer_utilization"], old_peak["mean_avg_peak_buffer_utilization"]),
                "- mean return: `%0.3f`" % metrics["mean_episode_return"],
                "- legacy reward: `%0.3f`" % metrics["mean_legacy_reward"],
                "",
            ])

    lines.extend([
        "## Conclusion",
        "",
        "Use the table above to compare whether the gentler linear spike penalties improved queue metrics over the original `buffer_peak_step`, the earlier `buffer_default_step`, and baseline.",
        "",
        "## Artifacts",
        "",
        "- Summary JSON: `%s`" % os.path.join(output_dir, "summary.json"),
        "- New model directories contain `model.zip`, `train.log`, `eval.log`, and `eval.json`.",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    python_executable = sys.executable
    output_dir = arg_or_default(
        "--output-dir",
        default=os.path.join(repo_root, "experiments", "peak_buffer_improvement_20260429"),
    )
    train_timesteps = arg_or_default("--train-timesteps", default=90112)
    n_steps = arg_or_default("--n-steps", default=8192)
    batch_size = arg_or_default("--batch-size", default=2048)
    num_models = arg_or_default("--num-models", default=1)
    eval_episodes = arg_or_default("--eval-episodes", default=20)
    seed = arg_or_default("--seed", default=123)
    eval_seed = arg_or_default("--eval-seed", default=2123)

    os.makedirs(output_dir, exist_ok=True)
    reference_summary = load_reference_summary()
    experiments = list(reference_summary["experiments"])

    for experiment in NEW_EXPERIMENTS:
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

        experiments.append({
            "name": experiment["name"],
            "flags": experiment["flags"],
            "model_path": model_path,
            "train_log": train_log,
            "eval_log": eval_log,
            "eval_json": eval_json,
            "metrics": eval_result,
        })

        print(
            "[done] %s return=%0.3f avg_peak=%0.6f avg_buffer=%0.6f"
            % (
                experiment["name"],
                eval_result["mean_episode_return"],
                eval_result["mean_avg_peak_buffer_utilization"],
                eval_result["mean_avg_buffer_utilization"],
            ),
            flush=True,
        )

    summary = {
        "output_dir": os.path.abspath(output_dir),
        "train_timesteps": train_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "num_models": num_models,
        "eval_episodes": eval_episodes,
        "eval_seed": eval_seed,
        "seed": seed,
        "experiments": experiments,
        "reference_summary": os.path.abspath(REFERENCE_SUMMARY),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    write_report(output_dir, summary)
    print("Finished peak improvement suite. Summary: %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
