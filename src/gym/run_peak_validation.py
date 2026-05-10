import inspect
import json
import os
import statistics
import subprocess
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
repo_root = os.path.dirname(os.path.dirname(currentdir))

sys.path.insert(0, os.path.dirname(currentdir))
from common.simple_arg_parse import arg_or_default


VARIANTS = [
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
]


METRICS = [
    "mean_episode_return",
    "mean_legacy_reward",
    "mean_avg_throughput",
    "mean_avg_latency",
    "mean_avg_loss",
    "mean_avg_buffer_utilization",
    "mean_avg_peak_buffer_utilization",
    "mean_max_buffer_utilization",
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


def mean(values):
    return float(statistics.mean(values)) if values else 0.0


def stdev(values):
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def summarize_variant(runs):
    summary = {"runs": runs}
    for metric in METRICS:
        values = [run["metrics"][metric] for run in runs]
        summary[metric + "_mean"] = mean(values)
        summary[metric + "_stdev"] = stdev(values)
    return summary


def percent_change(new_value, base_value):
    if base_value == 0.0:
        return 0.0
    return 100.0 * (new_value - base_value) / base_value


def write_report(output_dir, summary):
    report_path = os.path.join(output_dir, "REPORT.md")
    baseline = summary["variants"]["baseline"]
    default_step = summary["variants"]["buffer_default_step"]
    peak_linear = summary["variants"]["buffer_peak_linear_step"]

    baseline_peak = baseline["mean_avg_peak_buffer_utilization_mean"]
    baseline_avg = baseline["mean_avg_buffer_utilization_mean"]
    default_peak = default_step["mean_avg_peak_buffer_utilization_mean"]
    peak_linear_peak = peak_linear["mean_avg_peak_buffer_utilization_mean"]

    lines = [
        "# Peak-Aware Idea Validation Report",
        "",
        "## Setup",
        "",
        "- Training seeds: `%s`" % ", ".join(str(seed) for seed in summary["seeds"]),
        "- Training timesteps per run: `%d`" % summary["train_timesteps"],
        "- Training phases per run: `%d`" % summary["num_models"],
        "- PPO rollout / batch: `%d / %d`" % (summary["n_steps"], summary["batch_size"]),
        "- Evaluation episodes per run: `%d`" % summary["eval_episodes"],
        "",
        "This validation compares the original baseline, the earlier queue-aware `buffer_default_step`, and the best improved peak-aware candidate across multiple training seeds.",
        "",
        "## Aggregate Results",
        "",
        "| Model | Return Mean +/- Std | Legacy Mean +/- Std | Avg Buffer Mean +/- Std | Avg Peak Buffer Mean +/- Std | Max Buffer Mean +/- Std |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in ["baseline", "buffer_default_step", "buffer_peak_linear_step"]:
        variant = summary["variants"][name]
        lines.append(
            "| %s | `%0.3f +/- %0.3f` | `%0.3f +/- %0.3f` | `%0.6f +/- %0.6f` | `%0.6f +/- %0.6f` | `%0.6f +/- %0.6f` |"
            % (
                name,
                variant["mean_episode_return_mean"],
                variant["mean_episode_return_stdev"],
                variant["mean_legacy_reward_mean"],
                variant["mean_legacy_reward_stdev"],
                variant["mean_avg_buffer_utilization_mean"],
                variant["mean_avg_buffer_utilization_stdev"],
                variant["mean_avg_peak_buffer_utilization_mean"],
                variant["mean_avg_peak_buffer_utilization_stdev"],
                variant["mean_max_buffer_utilization_mean"],
                variant["mean_max_buffer_utilization_stdev"],
            )
        )

    lines.extend([
        "",
        "## Seed-Level Peak Buffer Results",
        "",
        "| Seed | baseline | buffer_default_step | buffer_peak_linear_step | Best |",
        "|---|---:|---:|---:|---|",
    ])

    seeds = summary["seeds"]
    for index, seed in enumerate(seeds):
        seed_values = {
            "baseline": baseline["runs"][index]["metrics"]["mean_avg_peak_buffer_utilization"],
            "buffer_default_step": default_step["runs"][index]["metrics"]["mean_avg_peak_buffer_utilization"],
            "buffer_peak_linear_step": peak_linear["runs"][index]["metrics"]["mean_avg_peak_buffer_utilization"],
        }
        best_name = min(seed_values, key=seed_values.get)
        lines.append(
            "| %d | `%0.6f` | `%0.6f` | `%0.6f` | `%s` |"
            % (
                seed,
                seed_values["baseline"],
                seed_values["buffer_default_step"],
                seed_values["buffer_peak_linear_step"],
                best_name,
            )
        )

    lines.extend([
        "",
        "## Verdict",
        "",
    ])

    if peak_linear_peak < baseline_peak:
        verdict = "The peak-aware idea worked against baseline on the primary peak-buffer metric."
    elif peak_linear_peak < default_peak:
        verdict = "The improved peak-aware idea beat `buffer_default_step`, but still did not beat baseline on peak-buffer control."
    else:
        verdict = "The improved peak-aware idea did not beat either baseline or `buffer_default_step` on the primary peak-buffer metric."

    lines.extend([
        verdict,
        "",
        "- `buffer_peak_linear_step` vs baseline avg peak buffer: `%+.2f%%`" % percent_change(peak_linear_peak, baseline_peak),
        "- `buffer_peak_linear_step` vs baseline avg buffer: `%+.2f%%`" % percent_change(peak_linear["mean_avg_buffer_utilization_mean"], baseline_avg),
        "- `buffer_peak_linear_step` vs `buffer_default_step` avg peak buffer: `%+.2f%%`" % percent_change(peak_linear_peak, default_peak),
        "- `buffer_peak_linear_step` wins on avg peak buffer in `%d / %d` seeds"
        % (
            sum(
                1
                for index in range(len(seeds))
                if peak_linear["runs"][index]["metrics"]["mean_avg_peak_buffer_utilization"]
                < baseline["runs"][index]["metrics"]["mean_avg_peak_buffer_utilization"]
                and peak_linear["runs"][index]["metrics"]["mean_avg_peak_buffer_utilization"]
                < default_step["runs"][index]["metrics"]["mean_avg_peak_buffer_utilization"]
            ),
            len(seeds),
        ),
        "",
        "## Artifacts",
        "",
        "- Summary JSON: `%s`" % os.path.join(output_dir, "summary.json"),
        "- Each run directory contains `model.zip`, `train.log`, `eval.log`, and `eval.json`.",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    python_executable = sys.executable
    output_dir = arg_or_default(
        "--output-dir",
        default=os.path.join(repo_root, "experiments", "peak_validation_20260429"),
    )
    train_timesteps = arg_or_default("--train-timesteps", default=90112)
    n_steps = arg_or_default("--n-steps", default=8192)
    batch_size = arg_or_default("--batch-size", default=2048)
    num_models = arg_or_default("--num-models", default=1)
    eval_episodes = arg_or_default("--eval-episodes", default=20)
    seeds_arg = arg_or_default("--seeds", default="123,231,312")
    seeds = [int(seed.strip()) for seed in str(seeds_arg).split(",") if seed.strip()]

    os.makedirs(output_dir, exist_ok=True)
    variant_results = {variant["name"]: {"runs": []} for variant in VARIANTS}

    for seed in seeds:
        eval_seed = seed + 2000
        for variant in VARIANTS:
            run_name = "%s_seed_%d" % (variant["name"], seed)
            run_dir = os.path.join(output_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            model_path = os.path.join(run_dir, "model.zip")
            train_log = os.path.join(run_dir, "train.log")
            eval_log = os.path.join(run_dir, "eval.log")
            eval_json = os.path.join(run_dir, "eval.json")

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
            ] + build_flag_list(variant["flags"])

            print("[train] %s" % run_name, flush=True)
            run_command(train_command, train_log)

            eval_command = [
                python_executable,
                os.path.join(repo_root, "src", "gym", "evaluate_model.py"),
                "--model-path=%s" % model_path,
                "--output-path=%s" % eval_json,
                "--episodes=%d" % eval_episodes,
                "--seed=%d" % eval_seed,
            ] + build_flag_list(variant["flags"])

            print("[eval] %s" % run_name, flush=True)
            run_command(eval_command, eval_log)

            with open(eval_json) as f:
                metrics = json.load(f)

            variant_results[variant["name"]]["runs"].append({
                "seed": seed,
                "eval_seed": eval_seed,
                "run_dir": run_dir,
                "train_log": train_log,
                "eval_log": eval_log,
                "eval_json": eval_json,
                "metrics": metrics,
            })

            print(
                "[done] %s return=%0.3f avg_peak=%0.6f avg_buffer=%0.6f"
                % (
                    run_name,
                    metrics["mean_episode_return"],
                    metrics["mean_avg_peak_buffer_utilization"],
                    metrics["mean_avg_buffer_utilization"],
                ),
                flush=True,
            )

    summary = {
        "output_dir": os.path.abspath(output_dir),
        "seeds": seeds,
        "train_timesteps": train_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "num_models": num_models,
        "eval_episodes": eval_episodes,
        "variants": {},
    }

    for variant in VARIANTS:
        summary["variants"][variant["name"]] = summarize_variant(variant_results[variant["name"]]["runs"])

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    write_report(output_dir, summary)
    print("Finished peak validation. Summary: %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
