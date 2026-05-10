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


DEFAULT_EXPERIMENTS = [
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
]


def command_to_string(command):
    return " ".join(command)


def run_command(command, log_path):
    with open(log_path, "w") as log_file:
        process = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    return process.returncode


def build_flag_list(flag_dict):
    result = []
    for key, value in flag_dict.items():
        result.append("%s=%s" % (key, value))
    return result


def main():
    python_executable = sys.executable
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = arg_or_default("--output-dir", default=os.path.join(repo_root, "experiments", "buffer_sweep_%s" % timestamp))
    train_timesteps = arg_or_default("--train-timesteps", default=45056)
    n_steps = arg_or_default("--n-steps", default=4096)
    batch_size = arg_or_default("--batch-size", default=512)
    num_models = arg_or_default("--num-models", default=1)
    eval_episodes = arg_or_default("--eval-episodes", default=20)
    seed = arg_or_default("--seed", default=123)

    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "output_dir": os.path.abspath(output_dir),
        "train_timesteps": train_timesteps,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "num_models": num_models,
        "eval_episodes": eval_episodes,
        "seed": seed,
        "experiments": [],
    }

    print("Writing experiment outputs to %s" % os.path.abspath(output_dir), flush=True)

    for experiment in DEFAULT_EXPERIMENTS:
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
            "--seed=%d" % (seed + 1000),
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

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("Finished sweep. Summary: %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
