"""
sim_PPO.py - Simulation script for training a PPO policy in the reach-avoid environment.
"""

import os
import time
import argparse
import yaml
import torch
import gymnasium as gym
import numpy as np
from types import SimpleNamespace

from RARL.PPO import PPO
from RARL.PPOTrainer import PPOTrainer
from gym_reachability import gym_reachability

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_PPO.yaml", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found. Using default values.")
        # Default config for PPO
        config_dict = {
            "PPO": {
                "ENV_NAME": "cont-obs-avoid-v0",
                "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
                "SEED": 0,
                "GAMMA": 0.99,
                "LAMBDA": 0.95,
                "CLIP_EPSILON": 0.2,
                "LR_ACTOR": 3e-4,
                "LR_CRITIC": 1e-3,
                "BATCH_SIZE": 64,
                "ARCHITECTURE": [256, 256],
                "ACTIVATION": "Tanh",
                "MAX_UPDATES": 1000,
                "MAX_EP_STEPS": 200,
                "PPO_EPOCHS": 10,
            }
        }
    else:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)

    # Assuming the config is under a "PPO" section
    params = SimpleNamespace(**config_dict["PPO"])

    device = torch.device(params.DEVICE)
    env_name = params.ENV_NAME

    print(f"Creating environment: {env_name}")
    env = gym.make(
        env_name,
        config=params,
        device=device
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dim_list = [state_dim] + params.ARCHITECTURE + [action_dim]

    print("Initializing PPO Agent...")
    agent = PPO(params, dim_list, env.action_space)

    print("Initializing PPO Trainer...")
    trainer = PPOTrainer(agent, params)

    print("Starting training...")
    trainer.learn(
        env,
        MAX_UPDATES=params.MAX_UPDATES,
        MAX_EP_STEPS=params.MAX_EP_STEPS,
        outFolder=f"PPO_{env_name}"
    )

    print("Training complete.")

if __name__ == "__main__":
    main()
