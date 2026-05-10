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


REFERENCE_SUMMARY = arg_or_default(
    "--reference-summary",
    default=os.path.join(repo_root, "experiments", "time_threshold_validation_20260429", "summary.json"),
)

THRESHOLDS = [0.1, 0.15, 0.2]
WEIGHTS = [50, 100, 150]

METRICS = [
    "mean_episode_return",
    "mean_legacy_reward",
    "mean_avg_throughput",
    "mean_avg_latency",
    "mean_avg_loss",
    "mean_avg_buffer_utilization",
    "mean_avg_peak_buffer_utilization",
    "mean_avg_buffer_time_above_threshold_fraction",
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


def variant_name(threshold, weight):
    threshold_label = str(threshold).replace(".", "")
    return "buffer_time_threshold_t%s_w%d" % (threshold_label, weight)


def parse_reference_variants():
    with open(REFERENCE_SUMMARY) as f:
        summary = json.load(f)
    reference_variants = {}
    for name, data in summary["variants"].items():
        reference_variants[name] = data
    return summary, reference_variants


def percent_change(new_value, base_value):
    if base_value == 0.0:
        return 0.0
    return 100.0 * (new_value - base_value) / base_value


def render_ranked_rows(variants):
    ranked = sorted(
        variants.items(),
        key=lambda item: (
            item[1]["mean_avg_peak_buffer_utilization_mean"],
            item[1]["mean_avg_buffer_utilization_mean"],
            -item[1]["mean_episode_return_mean"],
        ),
    )
    rows = []
    for rank, (name, variant) in enumerate(ranked, start=1):
        rows.append(
            "| %d | %s | `%0.6f` | `%0.6f` | `%0.6f` | `%0.3f` | `%0.3f` |"
            % (
                rank,
                name,
                variant["mean_avg_peak_buffer_utilization_mean"],
                variant["mean_avg_buffer_utilization_mean"],
                variant["mean_avg_buffer_time_above_threshold_fraction_mean"],
                variant["mean_episode_return_mean"],
                variant["mean_legacy_reward_mean"],
            )
        )
    return "\n".join(rows), ranked


def write_report(output_dir, summary):
    report_path = os.path.join(output_dir, "REPORT.md")
    baseline = summary["variants"]["baseline"]
    default_step = summary["variants"]["buffer_default_step"]

    rows, ranked = render_ranked_rows(summary["variants"])
    best_name, best_variant = ranked[0]

    lines = [
        "# Time-Above-Threshold Sweep Report",
        "",
        "## Setup",
        "",
        "- Training seeds: `%s`" % ", ".join(str(seed) for seed in summary["seeds"]),
        "- Training timesteps per run: `%d`" % summary["train_timesteps"],
        "- Training phases per run: `%d`" % summary["num_models"],
        "- PPO rollout / batch: `%d / %d`" % (summary["n_steps"], summary["batch_size"]),
        "- Threshold sweep: `%s`" % ", ".join(str(value) for value in THRESHOLDS),
        "- Weight sweep: `%s`" % ", ".join(str(value) for value in WEIGHTS),
        "",
        "This report reuses the existing multi-seed `baseline` and `buffer_default_step` runs and adds a grid sweep over the time-above-threshold reward family.",
        "",
        "## Ranked Results",
        "",
        "| Rank | Model | Avg Peak Buffer | Avg Buffer | Time Above Threshold | Return | Legacy Reward |",
        "|---|---|---:|---:|---:|---:|---:|",
        rows,
        "",
        "## Best Sweep Variant",
        "",
        "- best variant: `%s`" % best_name,
        "- avg peak buffer vs baseline: `%+.2f%%`"
        % percent_change(
            best_variant["mean_avg_peak_buffer_utilization_mean"],
            baseline["mean_avg_peak_buffer_utilization_mean"],
        ),
        "- avg peak buffer vs `buffer_default_step`: `%+.2f%%`"
        % percent_change(
            best_variant["mean_avg_peak_buffer_utilization_mean"],
            default_step["mean_avg_peak_buffer_utilization_mean"],
        ),
        "- avg buffer vs baseline: `%+.2f%%`"
        % percent_change(
            best_variant["mean_avg_buffer_utilization_mean"],
            baseline["mean_avg_buffer_utilization_mean"],
        ),
        "- time above threshold vs baseline: `%+.2f%%`"
        % percent_change(
            best_variant["mean_avg_buffer_time_above_threshold_fraction_mean"],
            baseline["mean_avg_buffer_time_above_threshold_fraction_mean"],
        ),
        "- return mean: `%0.3f`" % best_variant["mean_episode_return_mean"],
        "",
        "## Verdict",
        "",
    ]

    if best_name == "baseline":
        verdict = "None of the swept time-above-threshold variants beat baseline on the primary queue metrics."
    elif best_variant["mean_avg_peak_buffer_utilization_mean"] < baseline["mean_avg_peak_buffer_utilization_mean"]:
        verdict = "At least one time-above-threshold variant beat baseline on the primary peak-buffer metric."
    elif best_variant["mean_avg_peak_buffer_utilization_mean"] < default_step["mean_avg_peak_buffer_utilization_mean"]:
        verdict = "The sweep improved on `buffer_default_step`, but still did not beat baseline."
    else:
        verdict = "The sweep did not beat either baseline or `buffer_default_step` on the primary peak-buffer metric."

    lines.extend([
        verdict,
        "",
        "## Artifacts",
        "",
        "- Summary JSON: `%s`" % os.path.join(output_dir, "summary.json"),
        "- Each new run directory contains `model.zip`, `train.log`, `eval.log`, and `eval.json`.",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    python_executable = sys.executable
    output_dir = arg_or_default(
        "--output-dir",
        default=os.path.join(repo_root, "experiments", "time_threshold_sweep_20260429"),
    )
    train_timesteps = arg_or_default("--train-timesteps", default=90112)
    n_steps = arg_or_default("--n-steps", default=8192)
    batch_size = arg_or_default("--batch-size", default=2048)
    num_models = arg_or_default("--num-models", default=1)
    eval_episodes = arg_or_default("--eval-episodes", default=20)
    seeds_arg = arg_or_default("--seeds", default="123,231,312")
    seeds = [int(seed.strip()) for seed in str(seeds_arg).split(",") if seed.strip()]

    os.makedirs(output_dir, exist_ok=True)
    reference_summary, reference_variants = parse_reference_variants()
    variants = dict(reference_variants)

    for threshold in THRESHOLDS:
        for weight in WEIGHTS:
            name = variant_name(threshold, weight)
            if name == "buffer_time_threshold_t02_w150":
                # Existing validated run is stored as buffer_time_threshold_step.
                variants[name] = reference_variants["buffer_time_threshold_step"]
                continue

            flags = {
                "--reward-mode": "buffer_time_threshold_penalty",
                "--buffer-penalty-weight": "100.0",
                "--time-buffer-penalty-weight": str(weight),
                "--buffer-time-threshold": str(threshold),
                "--observe-buffer-util": "true",
                "--buffer-feedback-mode": "step",
                "--buffer-feedback-delay-steps": "1",
            }
            runs = []

            for seed in seeds:
                eval_seed = seed + 2000
                run_name = "%s_seed_%d" % (name, seed)
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
                ] + build_flag_list(flags)

                print("[train] %s" % run_name, flush=True)
                run_command(train_command, train_log)

                eval_command = [
                    python_executable,
                    os.path.join(repo_root, "src", "gym", "evaluate_model.py"),
                    "--model-path=%s" % model_path,
                    "--output-path=%s" % eval_json,
                    "--episodes=%d" % eval_episodes,
                    "--seed=%d" % eval_seed,
                ] + build_flag_list(flags)

                print("[eval] %s" % run_name, flush=True)
                run_command(eval_command, eval_log)

                with open(eval_json) as f:
                    metrics = json.load(f)

                runs.append({
                    "seed": seed,
                    "eval_seed": eval_seed,
                    "run_dir": run_dir,
                    "train_log": train_log,
                    "eval_log": eval_log,
                    "eval_json": eval_json,
                    "metrics": metrics,
                })

                print(
                    "[done] %s avg_peak=%0.6f avg_buffer=%0.6f time_above=%0.6f"
                    % (
                        run_name,
                        metrics["mean_avg_peak_buffer_utilization"],
                        metrics["mean_avg_buffer_utilization"],
                        metrics["mean_avg_buffer_time_above_threshold_fraction"],
                    ),
                    flush=True,
                )

            variants[name] = summarize_variant(runs)

    summary = {
        "output_dir": os.path.abspath(output_dir),
        "seeds": seeds,
        "train_timesteps": train_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "num_models": num_models,
        "eval_episodes": eval_episodes,
        "reference_summary": os.path.abspath(REFERENCE_SUMMARY),
        "variants": variants,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    write_report(output_dir, summary)
    print("Finished time-threshold sweep. Summary: %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
