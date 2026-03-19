"""
Please contact the author(s) of this library if you have any questions.
Authors: George Nehma   ( gnehma2020@fit.edu )

This module implements an environment considering the 2D point object dynamics.
This environemnt is roughly the same as the basic version of `zermelo_show.py`,
but this environment has a grid of cells (used for tabular Q-learning).
"""

import gymnasium.spaces
import numpy as np
import gymnasium as gym
import torch
import random
import matplotlib.pyplot as plt

from utils.utils import nearest_real_grid_point
from utils.utils import state_to_index


class PointMass2Env(gym.Env):

  def __init__(self, device, mode='RA', sample_inside_obs=False,):
    """Initializes the environment with given arguments.
    """
    # State bounds.
    self.bounds = np.array([
        [-2, 2],  # axis_0 = state, axis_1=bounds.
        [-2, 10]
    ])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]

    self.sample_inside_obs = sample_inside_obs

    # Time step parameter.
    self.time_step = 0.05

    # Dubins car parameters.
    self.upward_speed = 2.0

    # Control parameters.
    self.horizontal_rate = 1
    self.discrete_controls = np.array([
        -self.horizontal_rate, 0, self.horizontal_rate
    ])

    disturb_scale = 0.0
    self.discrete_disturb = np.array([
        -disturb_scale*self.horizontal_rate, 0, disturb_scale*self.horizontal_rate
    ])

    # Constraint set parameters.
    # X,Y position and Side Length.
    self.box1_x_y_length = np.array([1.25, 2, 1.5])  # Bottom right.
    self.corners1 = np.array([
        (self.box1_x_y_length[0] - self.box1_x_y_length[2] / 2.0),
        (self.box1_x_y_length[1] - self.box1_x_y_length[2] / 2.0),
        (self.box1_x_y_length[0] + self.box1_x_y_length[2] / 2.0),
        (self.box1_x_y_length[1] + self.box1_x_y_length[2] / 2.0)
    ])
    self.box2_x_y_length = np.array([-1.25, 2, 1.5])  # Bottom left.
    self.corners2 = np.array([
        (self.box2_x_y_length[0] - self.box2_x_y_length[2] / 2.0),
        (self.box2_x_y_length[1] - self.box2_x_y_length[2] / 2.0),
        (self.box2_x_y_length[0] + self.box2_x_y_length[2] / 2.0),
        (self.box2_x_y_length[1] + self.box2_x_y_length[2] / 2.0)
    ])
    self.box3_x_y_length = np.array([0, 6, 1.5])  # Top middle.
    self.corners3 = np.array([
        (self.box3_x_y_length[0] - self.box3_x_y_length[2] / 2.0),
        (self.box3_x_y_length[1] - self.box3_x_y_length[2] / 2.0),
        (self.box3_x_y_length[0] + self.box3_x_y_length[2] / 2.0),
        (self.box3_x_y_length[1] + self.box3_x_y_length[2] / 2.0)
    ])

    # Target set parameters.
    self.box4_x_y_length = np.array([0, 9.25, 1.5])  # Top.

    # Gym variables.
    self.action_space = gym.spaces.Discrete(3)  # horizontal_rate={-1,0,1}
    self.midpoint = (self.low + self.high) / 2.0
    self.interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(self.midpoint - self.interval / 2),
        np.float32(self.midpoint + self.interval / 2)
    )
    self.viewer = None

    # Discretization.
    self.grid_cells = None

    # Internal state.
    self.state = np.zeros(2)

    self.seed_val = 0
    self.costType = 'max'

    # Visualization params
    self.vis_init_flag = True
    (
        self.x_box1_pos, self.x_box2_pos, self.x_box3_pos, self.y_box1_pos,
        self.y_box2_pos, self.y_box3_pos
    ) = self.constraint_set_boundary()
    (self.x_box4_pos, self.y_box4_pos) = self.target_set_boundary()
    self.visual_initial_states = [
        np.array([0, 0]),
        np.array([-1, -2]),
        np.array([1, -2]),
        np.array([-1, 4]),
        np.array([1, 4])
    ]
    self.scaling = 1.

    # for torch
    self.device = device

    # Set random seed.
    np.random.seed(self.seed_val)

  def reset(self, *, seed=None, options=None, start=None):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): Which state to reset the environment to.
            If None, pick the state uniformly at random. Defaults to None.

    Returns:
        np.ndarray: The state the environment has been reset to.
    """
    if start is None:
      self.state = self.sample_random_state(sample_inside_obs=self.sample_inside_obs)
    else:
      self.state = start

    super().reset(seed=seed)
    info = {}
    return np.copy(self.state), info

  def sample_random_state(self, sample_inside_obs=False):
    """Picks the state uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
        the obstacles if True. Defaults to False.

    Returns:
        np.ndarray: sampled initial state.
    """
    inside_obs = True
    # Repeat sampling until outside obstacle if needed.
    while inside_obs:
      xy_sample = np.random.uniform(low=self.low, high=self.high)
      g_x = self.safety_margin(xy_sample)
      inside_obs = (g_x > 0)
      if sample_inside_obs:
        break

    return xy_sample

  def step(self, action):
    """Evolves the environment one step forward under given action.

    Args:
        action (tuple): the indexes of the action and disturbance
                        in the action and disturbance set.

    Returns:
        np.ndarray: next state.
        float: target margin at the new state.
        bool: True if the episode is terminated.
        dictionary: consist of safety margin at the new state.
    """
    action, disturbance = action
    # The signed distance must be computed before the environment steps
    # forward.
    if self.grid_cells is None:
      l_x = self.target_margin(self.state)
      g_x = self.safety_margin(self.state)
    else:
      nearest_point = nearest_real_grid_point(
          self.grid_cells, self.bounds, self.state
      )
      l_x = self.target_margin(nearest_point)
      g_x = self.safety_margin(nearest_point)

    # Move dynamics one step forward.
    x, y = self.state
    u = self.discrete_controls[action]
    d = self.discrete_disturb[disturbance]

    u_tot = u - d

    x, y = self.integrate_forward(x, y, u_tot)
    self.next_state = np.array([x, y])

    # Calculate whether episode is done.
    done = ((g_x > 0) or (l_x <= 0)) # unsafe or target is reached
    info = {"g_x": g_x, "l_x": l_x}
    return np.copy(self.next_state), max(l_x, g_x), done, info

  def integrate_forward(self, x, y, u):
    """Integrates the dynamics forward by one step.

    Args:
        x (float): Position in x-axis.
        y (float): Position in y-axis
        u (float): contol inputs, consisting of
                   horizontal speed minus disturbance.

    Returns:
        np.ndarray: (x, y) position of the next state.
    """
    x = x + self.time_step * u
    y = y + self.time_step * self.upward_speed
    return x, y

  def set_seed(self, seed):
    """Sets the seed for `numpy`, `random`, `PyTorch` packages.

    Args:
        seed (int): seed value.
    """
    self.seed_val = seed
    np.random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
    random.seed(self.seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  def safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: postivive numbers indicate being inside the failure set (safety
            violation).
    """
    box1_safety_margin = -(
        np.linalg.norm(s - self.box1_x_y_length[:2], ord=np.inf)
        - self.box1_x_y_length[-1] / 2.0
    )
    box2_safety_margin = -(
        np.linalg.norm(s - self.box2_x_y_length[:2], ord=np.inf)
        - self.box2_x_y_length[-1] / 2.0
    )
    box3_safety_margin = -(
        np.linalg.norm(s - self.box3_x_y_length[:2], ord=np.inf)
        - self.box3_x_y_length[-1] / 2.0
    )

    vertical_margin = (
        np.abs(s[1] - (self.low[1] + self.high[1]) / 2.0)
        - self.interval[1] / 2.0
    )
    horizontal_margin = np.abs(s[0]) - 2.0
    enclosure_safety_margin = max(horizontal_margin, vertical_margin)

    safety_margin = max(
        box1_safety_margin, box2_safety_margin, box3_safety_margin,
        enclosure_safety_margin
    )

    return self.scaling * safety_margin

  def target_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """
    box4_target_margin = (
        np.linalg.norm(s - self.box4_x_y_length[:2], ord=np.inf)
        - self.box4_x_y_length[-1] / 2.0
    )

    target_margin = box4_target_margin
    return self.scaling * target_margin

  def set_grid_cells(self, grid_cells):
    """Sets the number of grid cells.

    Args:
        grid_cells (tuple of ints): the ith value is the number of grid_cells
            for ith dimension of state.
    """
    self.grid_cells = grid_cells

  def set_bounds(self, bounds):
    """Sets the boundary and the observation_space of the environment.

    Args:
        bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
    """
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]

    # Double the range in each state dimension for Gym interface.
    self.observation_space = gym.spaces.Box(
        np.float32(self.midpoint - self.interval / 2),
        np.float32(self.midpoint + self.interval / 2)
    )

  def set_discretization(self, grid_cells, bounds):
    """Sets the number of grid cells and state bounds.

    Args:
        grid_cells (tuple of ints): the ith value is the number of grid_cells
            for ith dimension of state.
        bounds (np.ndarray): Bounds for the state.
    """
    self.set_grid_cells(grid_cells)
    self.set_bounds(bounds)

  def get_value(self, q_func, nx=41, ny=121, addBias=False):
    """Gets the state values given the Q-network.

    Args:
        q_func (object): agent's Q-network.
        nx (int, optional): # points in x-axis. Defaults to 41.
        ny (int, optional): # points in y-axis. Defaults to 121.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.

    Returns:
        np.ndarray: x-position of states
        np.ndarray: y-position of states
        np.ndarray: values
    """
    v = np.zeros((nx, ny))
    it = np.nditer(v, flags=['multi_index'])
    xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
    while not it.finished:
      idx = it.multi_index

      x = xs[idx[0]]
      y = ys[idx[1]]
      l_x = self.target_margin(np.array([x, y]))
      g_x = self.safety_margin(np.array([x, y]))

    #   if self.mode == 'normal' or self.mode == 'AARA':
      state = torch.FloatTensor([x, y]).to(self.device).unsqueeze(0)
    #   else:
        # z = max([l_x, g_x])
        # state = torch.FloatTensor([x, y, z]).to(self.device).unsqueeze(0)

      if addBias:
        v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
      else:
        v[idx] = q_func(state).min(dim=1)[0].item()
      it.iternext()
    return xs, ys, v

  def render(self, mode='human'):
    pass

  def constraint_set_boundary(self):
    """Computes the safe set boundary based on the analytic solution.

    Returns:
        tuple of np.ndarray: each array is of the shape (5, ). Since we use the
            box constraint in this environment, we need five points to plot a
            box.
    """
    x_box1_pos = np.array([
        self.box1_x_y_length[0] - self.box1_x_y_length[-1] / 2.0,
        self.box1_x_y_length[0] - self.box1_x_y_length[-1] / 2.0,
        self.box1_x_y_length[0] + self.box1_x_y_length[-1] / 2.0,
        self.box1_x_y_length[0] + self.box1_x_y_length[-1] / 2.0,
        self.box1_x_y_length[0] - self.box1_x_y_length[-1] / 2.0
    ])
    x_box2_pos = np.array([
        self.box2_x_y_length[0] - self.box2_x_y_length[-1] / 2.0,
        self.box2_x_y_length[0] - self.box2_x_y_length[-1] / 2.0,
        self.box2_x_y_length[0] + self.box2_x_y_length[-1] / 2.0,
        self.box2_x_y_length[0] + self.box2_x_y_length[-1] / 2.0,
        self.box2_x_y_length[0] - self.box2_x_y_length[-1] / 2.0
    ])
    x_box3_pos = np.array([
        self.box3_x_y_length[0] - self.box3_x_y_length[-1] / 2.0,
        self.box3_x_y_length[0] - self.box3_x_y_length[-1] / 2.0,
        self.box3_x_y_length[0] + self.box3_x_y_length[-1] / 2.0,
        self.box3_x_y_length[0] + self.box3_x_y_length[-1] / 2.0,
        self.box3_x_y_length[0] - self.box3_x_y_length[-1] / 2.0
    ])

    y_box1_pos = np.array([
        self.box1_x_y_length[1] - self.box1_x_y_length[-1] / 2.0,
        self.box1_x_y_length[1] + self.box1_x_y_length[-1] / 2.0,
        self.box1_x_y_length[1] + self.box1_x_y_length[-1] / 2.0,
        self.box1_x_y_length[1] - self.box1_x_y_length[-1] / 2.0,
        self.box1_x_y_length[1] - self.box1_x_y_length[-1] / 2.0
    ])
    y_box2_pos = np.array([
        self.box2_x_y_length[1] - self.box2_x_y_length[-1] / 2.0,
        self.box2_x_y_length[1] + self.box2_x_y_length[-1] / 2.0,
        self.box2_x_y_length[1] + self.box2_x_y_length[-1] / 2.0,
        self.box2_x_y_length[1] - self.box2_x_y_length[-1] / 2.0,
        self.box2_x_y_length[1] - self.box2_x_y_length[-1] / 2.0
    ])
    y_box3_pos = np.array([
        self.box3_x_y_length[1] - self.box3_x_y_length[-1] / 2.0,
        self.box3_x_y_length[1] + self.box3_x_y_length[-1] / 2.0,
        self.box3_x_y_length[1] + self.box3_x_y_length[-1] / 2.0,
        self.box3_x_y_length[1] - self.box3_x_y_length[-1] / 2.0,
        self.box3_x_y_length[1] - self.box3_x_y_length[-1] / 2.0
    ])

    return (
        x_box1_pos, x_box2_pos, x_box3_pos, y_box1_pos, y_box2_pos, y_box3_pos
    )

  def target_set_boundary(self):
    """Computes the target set boundary based on the analytic solution.

    Returns:
        tuple of np.ndarray: each array is of the shape (5, ). Since we use the
            box target in this environment, we need five points to plot a box.
    """
    x_box4_pos = np.array([
        self.box4_x_y_length[0] - self.box4_x_y_length[-1] / 2.0,
        self.box4_x_y_length[0] - self.box4_x_y_length[-1] / 2.0,
        self.box4_x_y_length[0] + self.box4_x_y_length[-1] / 2.0,
        self.box4_x_y_length[0] + self.box4_x_y_length[-1] / 2.0,
        self.box4_x_y_length[0] - self.box4_x_y_length[-1] / 2.0
    ])

    y_box4_pos = np.array([
        self.box4_x_y_length[1] - self.box4_x_y_length[-1] / 2.0,
        self.box4_x_y_length[1] + self.box4_x_y_length[-1] / 2.0,
        self.box4_x_y_length[1] + self.box4_x_y_length[-1] / 2.0,
        self.box4_x_y_length[1] - self.box4_x_y_length[-1] / 2.0,
        self.box4_x_y_length[1] - self.box4_x_y_length[-1] / 2.0
    ])

    return (x_box4_pos, y_box4_pos)
  
  def visualize(
      self, q_func, vmin=-1, vmax=1, nx=201, ny=201, labels=None,
      boolPlot=False, addBias=False, cmap='seismic'
  ):
    """
    Visulaizes the trained Q-network in terms of state values and trajectories
    rollout.

    Args:
        q_func (object): agent's Q-network.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 41.
        ny (int, optional): # points in y-axis. Defaults to 121.
        labels (list, optional): x- and y- labels. Defaults to None.
        boolPlot (bool, optional): plot the values in binary form if True.
            Defaults to False.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
        cmap (str, optional): color map. Defaults to 'seismic'.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    cbarPlot = True

    # == Plot failure / target set ==
    self.plot_target_failure_set(ax)

    # == Plot reach-avoid set ==
    self.plot_reach_avoid_set(ax)

    # == Plot V ==
    self.plot_v_values(
        q_func, ax=ax, fig=fig, vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
        boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias
    )

    # == Plot Trajectories ==
    # self.plot_trajectories(
    #     pro_q_func, adv_q_func, states=self.visual_initial_states, toEnd=False, ax=ax
    # )

    # == Formatting ==
    self.plot_formatting(ax=ax, labels=labels)
    fig.tight_layout()

  def plot_v_values(
      self, q_func, ax=None, fig=None, vmin=-1, vmax=1, nx=201, ny=201,
      cmap='seismic', alpha=0.8, boolPlot=False, cbarPlot=True, addBias=False
  ):
    """Plots state values.

    Args:
        q_func (object): agent's Q-network.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        fig (matplotlib.figure, optional): Defaults to None.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 201.
        ny (int, optional): # points in y-axis. Defaults to 201.
        cmap (str, optional): color map. Defaults to 'seismic'.
        alpha (float, optional): opacity. Defaults to 0.8.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        cbarPlot (bool, optional): plot the color bar if True.
            Defaults to True.
        addBias (bool, optional): adding bias to the values if True.
            Defaults to False.
    """
    axStyle = self.get_axes()

    # == Plot V ==
    _, _, v = self.get_value(q_func, nx, ny, addBias=addBias)
    vmax = np.ceil(max(np.max(v), np.max(-v)))
    vmin = -vmax

    if boolPlot:
      im = ax.imshow(
          v.T > 0., interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, alpha=alpha
      )
    else:
      im = ax.imshow(
          v.T, interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
      )
      if cbarPlot:
        cbar = fig.colorbar(
            im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
            ticks=[vmin, 0, vmax]
        )
        cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)

  def visualize_analytic_comparison(
      self, v, vmin=-1, vmax=1, boolPlot=False, cmap='seismic', ax=None
  ):
    """Overlays state value function.

    Args:
        q_values (np.ndarray): State-action values, which is of dimension
            (grid_cells, env.action_space.n).
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        boolPlot (bool, optional): plot the values in binary form if True.
            Defaults to False.
        cmap (str, optional): color map. Defaults to 'seismic'.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
    """
    axes = self.get_axes()

    if boolPlot:
      ax.imshow(
          v.T > vmin, interpolation='none', extent=axes[0], origin="lower",
          cmap='coolwarm'
      )
    else:
      ax.imshow(
          v.T, interpolation='none', extent=axes[0], origin="lower", cmap=cmap,
          vmin=vmin, vmax=vmax
      )
      # fig.colorbar(im, pad=0.01, shrink=0.95)

  def simulate_one_trajectory(self, pro_q_func, adv_q_func, T=10, state=None):
    """Simulates the trajectory given the state or randomly initialized.

    Args:
        pro_q_func (np.ndarray): Protagonist State-action values, which is of dimension
            (grid_cells, env.action_space.n).
        adv_q_func (np.ndarray): Adversary State-action values, which is of dimension
            (grid_cells, env.action_space.n).
        T (int, optional): the maximum length of the trajectory.
            Defaults to 10.
        state (np.ndarray, optional): if provided, set the initial state to its
            value. Defaults to None.

    Returns:
        np.ndarray: x-position of states in the trajectory.
        np.ndarray: y-position of states in the trajectory.
    """
    if state is None:
      state = self.sample_random_state()
    x, y = state
    traj_x = [x]
    traj_y = [y]
    result = 0 # not finished 

    for t in range(T):
      outsideTop = (state[1] >= self.bounds[1, 1])
      outsideLeft = (state[0] <= self.bounds[0, 0])
      outsideRight = (state[0] >= self.bounds[0, 1])
      done = outsideTop or outsideLeft or outsideRight
      if done:
        result = -1
        break
      else:
        if self.safety_margin(state[:2]) > 0:
          result = -1  # failed
          break
        elif self.target_margin(state[:2]) <= 0:
          result = 1  # succeeded
          break
      '''
            if self.safety_margin(state) > 0 or self.target_margin(state) < 0:
                break
            '''

      state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
      action_idx = pro_q_func(state_tensor).max(dim=1)[1].item()
      sa = torch.cat([state_tensor, torch.tensor([[action_idx]], dtype=torch.float32).to(self.device)], dim=1)
      disturb_idx = adv_q_func(sa).max(dim=1)[1].item()
    #   state_ix = state_to_index(self.grid_cells, self.bounds, state)
    #   action_ix = np.argmin(pro_q_func[state_ix])
    #   disturb_ix = np.argmax(adv_q_func[state_ix])
      u = self.discrete_controls[action_idx]
      d = self.discrete_disturb[disturb_idx]

      u_tot = u - d

      x, y = self.integrate_forward(x, y, u_tot)
      state = np.array([x, y])
      traj_x.append(x)
      traj_y.append(y)

    return traj_x, traj_y, result

  def simulate_trajectories(
      self, pro_q_func, adv_q_func, T=10, num_rnd_traj=None, states=None
  ):
    """
    Simulates the trajectories. If the states are not provided, we pick the
    initial states from the discretized state space.

    Args:
        pro_q_func (object): protagonist agent's Q-network.
        adv_q_func (object): adversary agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 10.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarrays, optional): if provided, set the initial
            states to its value. Defaults to None.

    Returns:
        list of np.ndarrays: each element is a tuple consisting of x and y
            positions along the trajectory.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    trajectories = []

    if states is None:
    #   for _ in range(num_rnd_traj):
      nx = 21
      ny = 61
      xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
      ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
      results = np.empty((nx, ny), dtype=int)
      it = np.nditer(results, flags=['multi_index'])
      print()
      while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]
        state = np.array([x, y])
        traj_x, traj_y, result = self.simulate_one_trajectory(pro_q_func, adv_q_func, T=T, state=state)
        trajectories.append((traj_x,traj_y))          
        results[idx] = result
        it.iternext()
      results = results.reshape(-1)
    else:
      for idx, state in enumerate(states):
        results = np.empty(shape=(len(states),), dtype=int)
        traj_x, traj_y, result = self.simulate_one_trajectory(pro_q_func, adv_q_func, T=T, state=state)
        trajectories.append((traj_x,traj_y))          
        results[idx] = result

    return trajectories, results

  def plot_trajectories(
      self, pro_q_func, adv_q_func, T=250, num_rnd_traj=None, states=None, ax=None, c='k',
      lw=2, zorder=2
  ):
    """Plots trajectories given the state-action values.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 10.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarrays, optional): if provided, set the initial
            states to its value. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        c (str, optional): color of the trajectories. Defaults to 'k'.
        lw (float, optional): linewidth of the trajectories. Defaults to 2.
        zorder (int, optional): graph layers order. Defaults to 2.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))

    trajectories, results = self.simulate_trajectories(
        pro_q_func, adv_q_func, T=T, num_rnd_traj=num_rnd_traj, states=states
    )

    for traj in trajectories:
      traj_x, traj_y = traj
      ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
      ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

  def plot_target_failure_set(
      self, ax=None, c_c='m', c_t='y', lw=1.5, zorder=1
  ):
    """Plots the target and the failure set.

    Args:
        ax (matplotlib.axes.Axes, optional)
        c_c (str, optional): color of the constraint set boundary.
            Defaults to 'm'.
        c_t (str, optional): color of the target set boundary.
            Defaults to 'y'.
        lw (float, optional): liewidth. Defaults to 1.5.
        zorder (int, optional): graph layers order. Defaults to 1.
    """
    # Plot bounadries of constraint set.
    ax.plot(self.x_box1_pos, self.y_box1_pos, color=c_c, lw=lw, zorder=zorder)
    ax.plot(self.x_box2_pos, self.y_box2_pos, color=c_c, lw=lw, zorder=zorder)
    ax.plot(self.x_box3_pos, self.y_box3_pos, color=c_c, lw=lw, zorder=zorder)

    # Plot boundaries of target set.
    ax.plot(self.x_box4_pos, self.y_box4_pos, color=c_t, lw=lw, zorder=zorder)

  def plot_reach_avoid_set(self, ax=None, c='g', lw=3, zorder=2):
    """Plots the analytic reach-avoid set.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
        c (str, optional): color of the rach-avoid set boundary.
            Defaults to 'g'.
        lw (int, optional): liewidth. Defaults to 3.
        zorder (int, optional): graph layers order. Defaults to 2.
    """
    slope = self.upward_speed / self.horizontal_rate

    def get_line(slope, end_point, x_limit, ns=100):
      x_end, y_end = end_point
      b = y_end - slope*x_end

      xs = np.linspace(x_limit, x_end, ns)
      ys = xs*slope + b
      return xs, ys

    # left unsafe set
    x = self.box2_x_y_length[0] + self.box2_x_y_length[2] / 2.0
    y = self.box2_x_y_length[1] - self.box2_x_y_length[2] / 2.0
    xs, ys = get_line(slope, end_point=[x, y], x_limit=-2.)
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

    # right unsafe set
    x = self.box1_x_y_length[0] - self.box1_x_y_length[2] / 2.0
    y = self.box1_x_y_length[1] - self.box1_x_y_length[2] / 2.0
    xs, ys = get_line(-slope, end_point=[x, y], x_limit=2.)
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

    # middle unsafe set
    x1 = self.box3_x_y_length[0] - self.box3_x_y_length[2] / 2.0
    x2 = self.box3_x_y_length[0] + self.box3_x_y_length[2] / 2.0
    x3 = self.box3_x_y_length[0]
    y = self.box3_x_y_length[1] - self.box3_x_y_length[2] / 2.0
    xs, ys = get_line(-slope, end_point=[x1, y], x_limit=x3)
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
    xs, ys = get_line(slope, end_point=[x2, y], x_limit=x3)
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

    # border unsafe set
    x1 = self.box4_x_y_length[0] - self.box4_x_y_length[2] / 2.0
    x2 = self.box4_x_y_length[0] + self.box4_x_y_length[2] / 2.0
    y = self.box4_x_y_length[1] + self.box4_x_y_length[2] / 2.0
    xs, ys = get_line(slope, end_point=[x1, y], x_limit=-2.)
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
    xs, ys = get_line(-slope, end_point=[x2, y], x_limit=2.)
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

  def plot_formatting(self, ax=None, labels=None):
    """Formats the visualization.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
        labels (list, optional): x- and y- labels. Defaults to None.
    """
    axStyle = self.get_axes()
    # == Formatting ==
    ax.axis(axStyle[0])
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    ax.grid(False)
    if labels is not None:
      ax.set_xlabel(labels[0], fontsize=52)
      ax.set_ylabel(labels[1], fontsize=52)

    ax.tick_params(
        axis='both', which='both', bottom=False, top=False, left=False,
        right=False
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])

  def get_axes(self):
    """Gets the bounds for the environment.

    Returns:
        list: contain np.ndarray (axes bounds) and float (aspect ratio).
    """
    x_span = self.bounds[0, 1] - self.bounds[0, 0]
    y_span = self.bounds[1, 1] - self.bounds[1, 0]
    aspect_ratio = x_span / y_span
    axes = np.array([
        self.bounds[0, 0] - .05, self.bounds[0, 1] + .05,
        self.bounds[1, 0] - .05, self.bounds[1, 1] + .05
    ])
    return [axes, aspect_ratio]
