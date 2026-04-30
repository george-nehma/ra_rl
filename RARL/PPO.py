"""
PPO.py – Proximal Policy Optimization for the reach-avoid environment.
This implementation focuses on a single-agent policy that learns to reach the target
while avoiding obstacles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .model import PPOPolicy, ValueNetwork

class PPO:
    def __init__(self, config, dimList, action_space):
        self.CONFIG = config
        self.gamma = config.GAMMA
        self.lmbda = getattr(config, 'LAMBDA', 0.95) # GAE parameter
        self.clip_epsilon = getattr(config, 'CLIP_EPSILON', 0.2)
        self.device = torch.device(config.DEVICE)

        # Actor
        self.actor = PPOPolicy(config, dimList, action_space.shape[0], action_space).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR if hasattr(config, 'LR_ACTOR') else config.LR_C)

        # Critic
        self.critic = ValueNetwork(config, dimList).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC if hasattr(config, 'LR_CRITIC') else config.LR_C)

    def select_action(self, state):
        """
        Returns action and log_prob.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, mean = self.actor.sample(state)

        # Scale action to environment bounds if necessary
        # Note: PPOPolicy in model.py doesn't scale actions in .sample(),
        # but the environment expect -1 to 1.
        # Let's assume the output is already in [-1, 1] or handled by the trainer.

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], mean.cpu().numpy()[0]

    def update(self, rollouts):
        """
        Update actor and critic based on a batch of rollouts.

        Args:
            rollouts: A dictionary containing:
                'states': torch.Tensor
                'actions': torch.Tensor
                'log_probs': torch.Tensor
                'rewards': torch.Tensor
                'dones': torch.Tensor
                'values': torch.Tensor
        """
        states = rollouts['states']
        actions = rollouts['actions']
        log_probs = rollouts['log_probs']
        rewards = rollouts['rewards']
        dones = rollouts['dones']
        values = rollouts['values']

        # 1. Compute Advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[-1] # Simplification
            else:
                next_value = values[t+1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lmbda * (1 - dones[t]) * last_gae_lam

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. PPO Update Loop
        # Typically we do multiple epochs over the same rollout
        epochs = getattr(self.CONFIG, 'PPO_EPOCHS', 10)
        for _ in range(epochs):
            # Actor update
            curr_log_probs, _ = self._eval_actions(states, actions)
            ratio = torch.exp(curr_log_probs - log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Critic update
            curr_values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(curr_values, returns)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        return actor_loss.item(), critic_loss.item()

    def _eval_actions(self, states, actions):
        """
        Helper to evaluate current policy log_probs for existing actions.
        """
        mean, sigma = self.actor.forward(states)
        dist = torch.distributions.Normal(mean, sigma)
        log_probs = dist.log_prob(actions).sum(-1)
        return log_probs, mean
