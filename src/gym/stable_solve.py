# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(parentdir, ".mplconfig"))

try:
    import gymnasium as gym
except ImportError:
    import gym

import network_sim

from stable_baselines3 import PPO

sys.path.insert(0, parentdir)
from common.simple_arg_parse import arg_or_default


arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch), flush=True)

gamma = arg_or_default("--gamma", default=0.99)
timesteps = arg_or_default("--timesteps", default=(1600 * 410))
num_models = arg_or_default("--num-models", default=6)
save_checkpoints = not arg_or_default("--skip-checkpoints", default=False)
skip_export = arg_or_default("--skip-export", default=False)
device = arg_or_default("--device", default="auto")
n_steps = arg_or_default("--n-steps", default=8192)
batch_size = arg_or_default("--batch-size", default=2048)
seed = arg_or_default("--seed", default=0)

print("gamma = %f" % gamma, flush=True)
print("timesteps per training phase = %d" % timesteps, flush=True)
print("training phases = %d" % num_models, flush=True)
print("rollout steps = %d" % n_steps, flush=True)
print("batch size = %d" % batch_size, flush=True)
print("seed = %d" % seed, flush=True)

env = gym.make("PccNs-v0")

policy_kwargs = {}
if arch:
    policy_kwargs["net_arch"] = dict(pi=arch, vf=arch)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    gamma=gamma,
    n_steps=n_steps,
    batch_size=batch_size,
    policy_kwargs=policy_kwargs,
    device=device,
    seed=seed,
)

for i in range(0, num_models):
    if save_checkpoints:
        model.save("./pcc_model_%d" % i)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

default_export_dir = "/tmp/pcc_saved_models/model_A/"
export_dir = arg_or_default("--model-dir", default=default_export_dir)
if not skip_export:
    if export_dir.endswith(".zip"):
        export_path = export_dir
    else:
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, "model.zip")
    model.save(export_path)
