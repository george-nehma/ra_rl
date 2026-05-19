"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements the replay memory (buffer) for off-policy reinforcement
learning. In this codespace, we use it for double deep Q-network.

This file is based on Adam Paszke's implementation of Replay Memory,
available at:

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""

import numpy as np
from .SAC import Transition


class ReplayMemory(object):
  """Contains a replay memory (or a memory buffer).

  Attribures:
      capacity (int): the maximum number of transitions can be stored.
      memory (list): the transitions stored.
      position (list): the index where the new transition will be stored.
      isfull (bool): whether the memory is fully occupied.
      seed (int): the random seed for this memory, which influences the
          sampling method.
  """

  def __init__(self, capacity, env, seed=0):
    """Initializes the memory with the maximum capacity and a random seed.
    """
    self.capacity = capacity
    # self.memory = []

    self.s = np.zeros((capacity, env.get_attr("obs_dim")[0]), dtype=np.float32)
    self.a = np.zeros((capacity, env.get_attr("act_dim")[0]), dtype=np.float32)
    self.d = np.zeros((capacity, env.get_attr("act_dim")[0]), dtype=np.float32)
    self.r = np.zeros((capacity, 1), dtype=np.float32)
    self.s_ = np.zeros((capacity, env.get_attr("obs_dim")[0]), dtype=np.float32)
    self.a_ = np.zeros((capacity, env.get_attr("act_dim")[0]), dtype=np.float32)
    self.done = np.empty(capacity, dtype=bool)
    self.info = {
      "g_x": np.zeros((capacity,), dtype=np.float32),
      "l_x": np.zeros((capacity,), dtype=np.float32)
    }

    self.position = 0
    self.size = 0
    self.isfull = False
    self.seed = seed
    np.random.seed(self.seed)

  def reset(self, capacity, env):
    """Clears the memory and reset the position to be zero.
    """
    self.s = np.zeros((capacity, env.get_attr("obs_dim")[0]), dtype=np.float32)
    self.a = np.zeros((capacity, env.get_attr("act_dim")[0]), dtype=np.float32)
    self.d = np.zeros((capacity, env.get_attr("act_dim")[0]), dtype=np.float32)
    self.r = np.zeros((capacity, 1), dtype=np.float32)
    self.s_ = np.zeros((capacity, env.get_attr("obs_dim")[0]), dtype=np.float32)
    self.a_ = np.zeros((capacity, env.get_attr("act_dim")[0]), dtype=np.float32)
    self.done = np.empty(capacity, dtype=bool)
    self.info = {
      "g_x": np.zeros((capacity,), dtype=np.float32),
      "l_x": np.zeros((capacity,), dtype=np.float32)
    }

    self.position = 0
    self.isfull = False

  def update(self, transition):
    """Updates the memory given the newcoming transition.
    """

    n = transition.s.shape[0]

    idxs = (self.position + np.arange(n)) % self.capacity

    self.s[idxs] = transition.s
    self.a[idxs] = transition.a
    self.d[idxs] = transition.d
    self.r[idxs] = transition.r.reshape(-1,1)
    self.s_[idxs] = transition.s_
    self.a_[idxs] = transition.a_
    self.done[idxs] = transition.done
    self.info["g_x"][idxs] = transition.info["g_x"]
    self.info["l_x"][idxs] = transition.info["l_x"]

    self.position = (self.position + n) % self.capacity
    self.size = min(self.size + n, self.capacity)

    # if len(self.memory) < self.capacity:
    #   self.memory.append(None)
    # self.memory[self.position] = transition
    # self.position = int((self.position + 1) % self.capacity)
    # if len(self.memory) == self.capacity:
    #   self.isfull = True

  def sample(self, batch_size):
    """Samples batch_size transitions from the memory uniformly at random.
    """
    # length = len(self.memory)
    # indices = np.random.randint(low=0, high=length, size=(batch_size,))
    # return [self.memory[i] for i in indices]

    idxs = np.random.randint(0, self.size, size=batch_size)

    return Transition(
        self.s[idxs],
        self.a[idxs],
        self.d[idxs],
        self.r[idxs],
        self.s_[idxs],
        self.a_[idxs],
        self.done[idxs],
        {"g_x": self.info["g_x"][idxs], "l_x": self.info["l_x"][idxs]}
    )

  def __len__(self):
    """Returns the number of transitions in the memory.
    """
    return self.size # len(self.memory)
