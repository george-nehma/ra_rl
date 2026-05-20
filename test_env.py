import os
import shutil
import time
from warnings import simplefilter

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml
from types import SimpleNamespace

# CHANGED: DDQNEnsemble as protagonist
from RARL.SACTrainer import SACTrainer
from RARL.SAC import SAC
from RARL.config import ceConfig
from RARL.utils import save_obj
from utils.utils import (
    plot_protagonist_adversary_actions,
    plot_RA_eval,
    PlotConfig,
)
from gym_reachability import gym_reachability    # noqa: F401 — registers envs

import argparse

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M_%S")

# == ARGS — YAML config, same as sim_new_point_mass.py ==
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", type=str)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load (for evaluation or resuming training).")
parser.add_argument("--checkpointVal", type=int, default=50000, help="Checkpoint iteration to load for evaluation (ignored if --checkpoint not provided).")
parser.add_argument(
    "--num_jobs", type=int, default=4,
    help="Total number of sim_ensemble.py processes running in parallel. "
         "Used to divide CPU threads evenly across jobs."
)
script_args = parser.parse_args()

# -- Pin PyTorch threads so parallel jobs don't fight over cores.
# With N jobs on a C-core machine each job gets C/N threads.
# e.g. 16 jobs on 64 cores → 4 threads each → all 64 cores busy, no contention.
_total_cores = os.cpu_count() or 1
_threads_per_job = max(1, _total_cores // script_args.num_jobs)
torch.set_num_threads(_threads_per_job)
torch.set_num_interop_threads(max(1, _threads_per_job // 2))
print(f"[Thread config] {_total_cores} cores / {script_args.num_jobs} jobs "
      f"= {_threads_per_job} threads per job")

if script_args.checkpoint is not None:
    with open(os.path.join(script_args.checkpoint, "init_configs.yaml"), "r") as f:
        config_dict = yaml.safe_load(f)
else:
    with open(script_args.config, "r") as f:
        config_dict = yaml.safe_load(f)

args = SimpleNamespace(**{k: v for section in config_dict.values() for k, v in section.items()})
print(args)

# == CONFIGURATION ==
env_name     = args.envName
num_envs     = args.numEnvs
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if env_name =="zermelo_show-v0":
    env_title = "point-mass" 
elif env_name == "one_player_reach_avoid_lunar_lander":
    env_title = "lunar-lander"
elif env_name == "cont-obs-avoid-v0":
    env_title = "cont-obs-avoid"

updatePeriod = int(args.maxUpdates / args.updateTimes)
                           
storeFigure  = args.storeFigure
plotFigure   = args.plotFigure

if script_args.checkpoint is not None:
    outFolder = script_args.checkpoint
elif args.showTime:
    fn = args.name + args.doneType
    fn = fn + '-' + timestr
    outFolder    = os.path.join(args.outFolder, 'ensemble', str(args.numCritics)+ "critics", env_title, 'SAC', args.mode, fn)

figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)
print(outFolder)

# == Epsilon / gamma schedule — identical to sim_new_point_mass.py ==
if args.mode == 'RA':
    agentMode = 'RA'
    if args.annealing:
        GAMMA_END        = 0.999999
        EPS_PERIOD       = int(args.updatePeriodEps / 10)
        EPS_RESET_PERIOD = args.updatePeriodEps
    else:
        GAMMA_END        = 0.999
        EPS_PERIOD       = args.updatePeriodEps
        EPS_RESET_PERIOD = args.maxUpdates

elif args.mode == 'AARA':
    agentMode = 'AARA'
    if args.annealing:
        GAMMA_END        = 0.999999
        EPS_PERIOD       = int(args.updatePeriodEps / 10)
        EPS_RESET_PERIOD = args.updatePeriodEps
    else:
        GAMMA_END        = 0.999
        EPS_PERIOD       = args.updatePeriodEps
        EPS_RESET_PERIOD = args.maxUpdates

sample_inside_obs = False

# == Environment — passes config=args like updated sim_new_point_mass.py ==
print("\n== Environment Information ==")
train_envs = gym.make_vec(env_name, num_envs=num_envs, max_episode_steps=5, config=args, device=device, sample_inside_obs=sample_inside_obs)
eval_env = gym.make(
    env_name, max_episode_steps=args.maxSteps, config=args, device=device,
    sample_inside_obs=sample_inside_obs
)
stateDim    = eval_env.unwrapped.state.shape[0]
actionNum   = eval_env.unwrapped.action_space.shape[0]

obs, _ = train_envs.reset()
for i in range(20):
    u = np.zeros((num_envs, actionNum))
    d = np.zeros((num_envs, actionNum))

    input = np.concatenate([u,d], axis=1)

    s_, r, done, truncated, info = train_envs.step(input)

