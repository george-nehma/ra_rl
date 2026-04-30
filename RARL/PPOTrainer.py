"""
PPOTrainer.py

A trainer for reach-avoid reinforcement learning using Proximal Policy Optimization (PPO).
Designed for the continuous_obs_avoid environment.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .PPO import PPO
from .model import PPOPolicy

class PPOTrainer:
    def __init__(self, ppo_agent: PPO, CONFIG):
        self.agent = ppo_agent
        self.CONFIG = CONFIG

    def learn(
        self,
        env,
        MAX_UPDATES=1000, # For PPO, this is often number of iterations (rollouts)
        MAX_EP_STEPS=200,
        checkPeriod=50,
        outFolder="PPO_RA",
        verbose=True,
    ):
        """
        Main training loop for PPO.
        """
        os.makedirs(outFolder, exist_ok=True)
        logger = SummaryWriter(log_dir=os.path.join(outFolder, "logs"))

        training_records = []

        t0_learn = time.time()

        for iteration in range(MAX_UPDATES):
            # 1. Collect rollout
            rollout = self._collect_rollout(env, MAX_EP_STEPS)

            # 2. Update agent
            losses = self.agent.update(rollout)
            actor_loss, critic_loss = losses

            training_records.append([actor_loss, critic_loss])

            # Log to tensorboard
            logger.add_scalar("Loss/Actor", actor_loss, iteration)
            logger.add_scalar("Loss/Critic", critic_loss, iteration)

            if iteration % checkPeriod == 0:
                if verbose:
                    print(f"Iteration {iteration}/{MAX_UPDATES} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")

                # Save checkpoint
                self._save_checkpoint(iteration, outFolder)

        t1_learn = time.time()
        print(f"\nLearning finished in {t1_learn - t0_learn:.1f}s")

        logger.close()
        return np.array(training_records)

    def _collect_rollout(self, env, max_steps):
        """
        Collects a single rollout (trajectory) for PPO update.
        """
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        s, info = env.reset()

        for step in range(max_steps):
            # Use the agent to select action
            action, log_prob, mean = self.agent.select_action(s)

            # Get value from critic
            state_t = torch.FloatTensor(s).to(self.agent.device).unsqueeze(0)
            with torch.no_grad():
                val = self.agent.critic(state_t).item()

            # Step environment
            # Note: continuous_obs_avoid expects (action, disturbance)
            # For a single agent, we assume disturbance is zero or handled
            s_next, r, done, truncated, info = env.step((action, np.zeros_like(action)))

            states.append(s)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(r)
            dones.append(done)
            values.append(val)

            s = s_next
            if done or truncated:
                break

        # Convert to tensors
        return {
            'states': torch.FloatTensor(np.array(states)).to(self.agent.device),
            'actions': torch.FloatTensor(np.array(actions)).to(self.agent.device),
            'log_probs': torch.FloatTensor(np.array(log_probs)).to(self.agent.device),
            'rewards': torch.FloatTensor(np.array(rewards)).to(self.agent.device),
            'dones': torch.FloatTensor(np.array(dones)).to(self.agent.device),
            'values': torch.FloatTensor(np.array(values)).to(self.agent.device),
        }

    def _save_checkpoint(self, iteration, outFolder):
        path = os.path.join(outFolder, f"ppo_model_{iteration}.pt")
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
        }, path)
