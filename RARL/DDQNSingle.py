"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements reach-avoid reinforcement learning with double deep
Q-network. It also supports the standard sum of discounted rewards (Lagrange
cost) reinforcement learning.

Here we aim to minimize the reach-avoid cost, given by the Bellman backup:
    - a' = argmin_a' Q_network(s', a')
    - V(s') = Q_target(s', a')
    - V(s) = gamma ( max{ g(s), min{ l(s), V(s') } }
             + (1-gamma) max{ g(s), l(s) }
    - loss = E[ ( V(f(s,a)) - Q_network(s,a) )^2 ]
"""

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from .model import Model
from .DDQN import DDQN, Transition


class DDQNSingle(DDQN):
  """
  Implements the double deep Q-network algorithm. Supports minimizing the
  reach-avoid cost or the standard sum of discounted costs.

  Args:
      DDQN (object): an object implementing the basic utils functions.
  """

  def __init__(
      self, CONFIG, numAction, memory, dimList, mode="AARA",
      terminalType="max", verbose=True
  ):
    """
    Initializes with a configuration object, environment information, neural
    network architecture, reinforcement learning algorithm type and type of
    the terminal value for reach-avoid reinforcement learning.

    Args:
        CONFIG (object): configuration.
        numAction (int): the number of actions.
        dimList (np.ndarray): dimensions of each layer in the neural network.
        mode (str, optional): the reinforcement learning mode.
            Defaults to 'RA'.
        terminalType (str, optional): type of the terminal value.
            Defaults to 'max'.
        verbose (bool, optional): print the messages if True. Defaults to True.
    """
    super(DDQNSingle, self).__init__(CONFIG)

    self.mode = mode  # 'normal' or 'RA'
    self.terminalType = terminalType

    self.memory = memory

    # == ENV PARAM ==
    self.numAction = numAction

    # == Build neural network for (D)DQN ==
    self.dimList = dimList
    self.actType = CONFIG.ACTIVATION
    self.build_network(dimList, self.actType, verbose)
    print(
        "DDQN: mode-{}; terminalType-{}".format(self.mode, self.terminalType)
    )

  def build_network(self, dimList, actType="Tanh", verbose=True):
    """Builds a neural network for the Q-network.

    Args:
        dimList (np.ndarray): dimensions of each layer in the neural network.
        actType (str, optional): activation function. Defaults to 'Tanh'.
        verbose (bool, optional): print the messages if True. Defaults to True.
    """
    self.Q_network = Model(dimList, actType, verbose=verbose)
    self.target_network = Model(dimList, actType)

    if self.device == torch.device("cuda"):
      self.Q_network.cuda()
      self.target_network.cuda()

    self.build_optimizer()

  def update(self, addBias=False):
    """Updates the Q-network using a batch of sampled replay transitions.

    Args:
        addBias (bool, optional): use biased version of value function if
            True. Defaults to False.

    Returns:
        float: critic loss.
    """
    if len(self.memory) < self.BATCH_SIZE * 20:
      return

    # == EXPERIENCE REPLAY ==
    transitions = self.memory.sample(self.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043
    # for detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    (non_final_mask, non_final_state_nxt, state, action, reward, g_x,
     l_x) = self.unpack_batch(batch)

    # == get Q(s,a) ==
    # `gather` reguires that idx is Long and input and index should have the
    # same shape with only difference at the dimension we want to extract.
    # value out[i][j][k] = input[i][j][ index[i][j][k] ], which has the
    # same dim as index
    # -> state_action_values = Q [ i ][ action[i] ]
    # view(-1): from mtx to vector
    self.Q_network.train()
    state_action_values = (
        self.Q_network(state).gather(dim=1, index=action).view(-1)
    )

    # == get a' by Q_network: a' = argmin_a' Q_network(s', a') ==
    with torch.no_grad():
      self.Q_network.eval()
      action_nxt = (
          self.Q_network(non_final_state_nxt).min(1, keepdim=True)[1]
      )

    # == get expected value ==
    state_value_nxt = torch.zeros(self.BATCH_SIZE).to(self.device)

    with torch.no_grad():  # V(s') = Q_target(s', a'), a' is from Q_network
      if self.double_network:
        self.target_network.eval()
        Q_expect = self.target_network(non_final_state_nxt)
      else:
        self.Q_network.eval()
        Q_expect = self.Q_network(non_final_state_nxt)
    state_value_nxt[non_final_mask] = \
        Q_expect.gather(dim=1, index=action_nxt).view(-1)

    # == Discounted Reach-Avoid Bellman Equation (DRABE) ==
    if self.mode == "RA":
      y = torch.zeros(self.BATCH_SIZE).float().to(self.device)
      final_mask = torch.logical_not(non_final_mask)
      if addBias:  # Bias version:
        # V(s) = gamma ( max{ g(s), min{ l(s), V_diff(s') } }
        #        - max{ g(s), l(s) } ),
        # where V_diff(s') = V(s') + max{ g(s'), l(s') }
        min_term = torch.min(l_x, state_value_nxt + torch.max(l_x, g_x))
        terminal = torch.max(l_x, g_x)
        non_terminal = torch.max(min_term, g_x) - terminal
        y[non_final_mask] = self.GAMMA * non_terminal[non_final_mask]
        y[final_mask] = terminal[final_mask]
      else:
        # Another version (discussed on Feb. 22, 2021):
        # we want Q(s, u) = V( f(s,u) ).
        non_terminal = torch.max(
            g_x[non_final_mask],
            torch.min(l_x[non_final_mask], state_value_nxt[non_final_mask]),
        )
        terminal = torch.max(l_x, g_x)

        # normal state
        y[non_final_mask] = non_terminal * self.GAMMA + terminal[
            non_final_mask] * (1 - self.GAMMA)

        # terminal state
        if self.terminalType == "g":
          y[final_mask] = g_x[final_mask]
        elif self.terminalType == "max":
          y[final_mask] = terminal[final_mask]
        else:
          raise ValueError("invalid terminalType")
    # == Discounted Adaptive Reach-Avoid Bellman Equation (DARABE) ==
    elif self.mode == "AARA":
      y = torch.zeros(self.BATCH_SIZE).float().to(self.device)
      final_mask = torch.logical_not(non_final_mask)
      # V(s) = ( 1 - gamma ) * max{ g(s), l(s) } ) + gamma * ( max{ g(s), min{ l(s), V_diff(s') } }
      #        
      # where V_diff(s') = V(s') + max{ g(s'), l(s') }
      min_term = torch.min(l_x, state_value_nxt + torch.max(l_x, g_x))
      terminal = torch.max(l_x, g_x)
      non_terminal = torch.max(min_term, g_x) - terminal
      y[non_final_mask] = self.GAMMA * non_terminal[non_final_mask]
      y[final_mask] = terminal[final_mask]
    else:  # V(s) = c(s, a) + gamma * V(s') == Lagrange method ==
      y = state_value_nxt * self.GAMMA + reward

    # == regression: Q(s, a) <- V(s) ==
    loss = smooth_l1_loss(
        input=state_action_values,
        target=y.detach(),
    )

    # == backpropagation ==
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
    self.optimizer.step()

    self.update_target_network()

    return loss.item()

  def select_action(self, state, explore=False):
    """Selects the action given the state and conditioned on `explore` flag.

    Args:
        state (np.ndarray): the state of the environment. If the adversary is selecting
            it's action then state has the protagonist action concatenated. 
        explore (bool, optional): randomize the deterministic action by
            epsilon-greedy algorithm if True. Defaults to False.

    Returns:
        int: action index
    """

    
    if (np.random.rand() < self.EPSILON) and explore:
      action_idx = np.random.randint(0, self.numAction)
    else:
      self.Q_network.eval()
      if len(state.shape) > 1:
        pro_action_idx = state[1]
        state = state[0]
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_idx = self.Q_network(state,pro_action_idx).max(dim=1)[1].item()
      else:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_idx = self.Q_network(state).min(dim=1)[1].item()

    return action_idx

