import gymnasium.spaces
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import random
from .env_utils import calculate_margin_rect, calculate_margin_circle

from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ContinuousObsAvoidEnv(gym.Env):
    
    def __init__(self, config, device, mode='AARA', doneType='TF', thickness=0.1, 
                 sample_inside_obs=False,
    ):
        """Initializes the environment with given arguments.

    Args:
        config(object): configuration parameters.
        device (str): device type (used in PyTorch).
        mode (str, optional): reinforcement learning type. Defaults to 'AARA'.
        doneType (str, optional): conditions to raise `done flag in
            training. Defaults to 'TF'.
        thickness (float, optional): the thickness of the obstacles.
            Defaults to 0.1.
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        
    """
        
        super(ContinuousObsAvoidEnv, self).__init__()
        
        self.config = config
        self.mode = mode
        self.bounds = np.array([[0, 6.88], [0, 11.0]]) # x,y
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.sample_inside_obs = sample_inside_obs

        # Time-step Parameters.
        self.time_step = self.config.timeStep   

        self.state = np.zeros(2) # x, y, theta 
        self.obs_high = np.array([6.88, 11.0], dtype=np.float32) # x, y, theta
        self.obs_low = np.array([0.0, 0.0], dtype=np.float32) # x, y, theta
        self.act_bound = np.array([2.0, 2.0]).T # v, v_w
        self.d_bound = np.array([0.0, 0.0]).T # d_v, d_w
        self.act_dim = self.act_bound.shape[0]
        self.obs_dim = self.obs_high.shape[0]

        self.midpoint = (self.obs_low[0:2] + self.obs_high[0:2]) / 2.0
        self.interval = self.obs_high[0:2] - self.obs_low[0:2]

        # Define action and observation space
        self.action_space = gymnasium.spaces.Box(low=-np.ones((self.act_dim,), dtype=np.float32), high=np.ones((self.act_dim,), dtype=np.float32), shape=(self.act_dim,))
        self.disturbance_space = gymnasium.spaces.Box(low=-np.ones((self.act_dim,), dtype=np.float32), high=np.ones((self.act_dim,), dtype=np.float32), shape=(self.act_dim,))
        self.observation_space = gymnasium.spaces.Box(low=self.obs_low, high=self.obs_high, shape=(self.obs_dim,))
        
        self.viewer = None

        # Define target set parameters 
        self.target_x_y_w_h = np.array([[1.44, 7.5, 1.5, 1.5]])

        # Define the bounds of the robot (x, y, w, h)
        self.robot_bounds = np.array([self.state[0], self.state[1], 0.5, 0.7])  # x, y, w, h

        # Set random seed.
        self.seed_val = 0
        self.set_seed(self.seed_val)

        # Cost Parameters
        self.penalty = 1.
        self.reward = -1.
        self.costType = 'sparse'
        self.scaling = 1.

        self.doneType = doneType

        # Visualization Parameters
        self.target_set_boundary = self.get_target_set_boundary()
        self.visual_initial_states = [
            np.array([0.5, 0.5]),
            np.array([5, 1.2]),
            np.array([3.5, 1.5]),
            np.array([4, 3.8]),
            np.array([6.3, 6.5]),
        ]

        print(
            "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
                self.mode, self.doneType, self.sample_inside_obs
            )
        )

        # for torch
        self.device = device

    def reset(self, *, seed=None, options=None, start=None):
        """Resets the state of the environment.

        Args:
            start (np.ndarray, optional): state to reset the environment to.
                If None, pick the state uniformly at random. Defaults to None.

        Returns:
            np.ndarray: The state the environment has been reset to.
        """

        if start is None:
            self.state = self.sample_random_state(
                sample_inside_obs=self.sample_inside_obs
            )
        else:
            self.state = start

        super().reset(seed=seed)
        info = {}
        return np.asarray(self.state, dtype=np.float32), info

    def sample_random_state(self, sample_inside_obs=False):
        """Picks the state uniformly at random.

        Args:
            sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.

        Returns:
            np.ndarray: sampled initial state.
        """
        # Define obstacles as a list of (center_x, center_y, radius)
        self.randomiseObstacles(self.bounds, num_obstacles=3)

        inside_obs = True
        # Repeat sampling until outside obstacle if needed.
        while inside_obs:
            sample_state = self.observation_space.sample()
            xy_sample = sample_state[:2]

            g_x = self.safety_margin(xy_sample)
            inside_obs = (g_x > 0)

            if sample_inside_obs:
                break

        return sample_state

    def randomiseObstacles(self, bounds, num_obstacles=2):
        """Randomizes the obstacles in the environment.

        Args:
            num_obstacles (int, optional): number of obstacles. Defaults to 2.
            obs_size_range (tuple of floats, optional): range of the width and
                height of the obstacles. Defaults to (0.5, 1.5).
        """
        random.seed(0)
        x_lim = bounds[0,:]
        y_lim = bounds[1,:]

        radius = 0.5
        self.obstacles = []

        count = 0 
        while count < num_obstacles:
            center_x = random.random() * (x_lim[1] - x_lim[0]) + x_lim[0]
            center_y = random.random() * (y_lim[1] - y_lim[0]) + y_lim[0]

            # Find the closest point on the target set to the obstacle center
            closest_x = np.clip(center_x, self.target_x_y_w_h[0, 0] - self.target_x_y_w_h[0, 2] / 2, self.target_x_y_w_h[0, 0] + self.target_x_y_w_h[0, 2] / 2)
            closest_y = np.clip(center_y, self.target_x_y_w_h[0, 1] - self.target_x_y_w_h[0, 3] / 2, self.target_x_y_w_h[0, 1] + self.target_x_y_w_h[0, 3] / 2)

            # Check if the new obstacle overlaps with existing obstacles
            overlap = False
            for obs in self.obstacles:
                (obs_center_x, obs_center_y), _ = obs

                # check if the obstacle is within 3r of any existing obstacle (make it easier to get through obs)
                if (abs(center_x - obs_center_x) < 3*radius) and (abs(center_y - obs_center_y) < 3*radius):
                    overlap = True
                    break
                
                # check if the obstacle overlaps with the boundary of the environment
                if (abs(center_x - x_lim[0]) < radius) or (abs(center_x - x_lim[1]) < radius) or (abs(center_y - y_lim[0]) < radius) or (abs(center_y - y_lim[1]) < radius):
                    overlap = True
                    break

                # check if the obstacle overlaps with the target set
                if (center_x - closest_x) ** 2 + (center_y - closest_y) ** 2 <= radius ** 2:
                    overlap = True
                    break

            if not overlap:
                self.obstacles.append((np.array([center_x, center_y]), radius))
                count += 1

    # == Setting Hyper-Parameters ==
    def set_costParam(
        self, penalty=1., reward=-1., costType='sparse', scaling=1.
    ):
        """
        Sets the hyper-parameters for the `cost` signal used in training, important
        for standard (Lagrange-type) reinforcement learning.

        Args:
            penalty (float, optional): cost when entering the obstacles or crossing
                the environment boundary. Defaults to 1.0.
            reward (float, optional): cost when reaching the targets.
                Defaults to -1.0.
            costType (str, optional): providing extra information when in
                neither the failure set nor the target set. Defaults to 'sparse'.
            scaling (float, optional): scaling factor of the cost. Defaults to 1.0.
        """
        self.penalty = penalty
        self.reward = reward
        self.costType = costType
        self.scaling = scaling

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


    def get_target_set_boundary(self):
        """Gets the target set boundary.

        Returns:
            np.ndarray: of the shape (#target, 5, 2). Since we use the box target
                in this environment, we need 5 points to plot the box. The last
                axis consists of the (x, y) position.
        """
        num_target_set = self.target_x_y_w_h.shape[0]
        target_set_boundary = np.zeros((num_target_set, 5, 2))

        for idx, target_set in enumerate(self.target_x_y_w_h):
            x, y, w, h = target_set
            x_l = x - w/2.0
            x_h = x + w/2.0
            y_l = y - h/2.0
            y_h = y + h/2.0
            target_set_boundary[idx, :, 0] = [x_l, x_l, x_h, x_h, x_l]
            target_set_boundary[idx, :, 1] = [y_l, y_h, y_h, y_l, y_l]

        return target_set_boundary
    

    # == Getting Margin ==
    def safety_margin(self, s):
        """Computes the margin (e.g. distance) between the state and the failue set.

        Args:
            s (np.ndarray): the state of the agent.

        Returns:
            float: postivive numbers indicate being inside the failure set (safety
                violation).
        """
        g_x_list = []

        # constraint_set_safety_margin
        for _, constraint_set in enumerate(self.obstacles):
            g_x = calculate_margin_circle(s, constraint_set, negativeInside=False)
            g_x_list.append(g_x)

        # enclosure_safety_margin
        boundary_x_y_w_h = np.append(self.midpoint, self.interval)
        g_x = calculate_margin_rect(s, boundary_x_y_w_h, negativeInside=True)
        g_x_list.append(g_x)

        safety_margin = np.max(np.array(g_x_list))

        # return np.tanh(safety_margin/3)
        return self.scaling * safety_margin

    def target_margin(self, s):
        """Computes the margin (e.g. distance) between the state and the target set.

        Args:
            s (np.ndarray): the state of the agent.

        Returns:
            float: negative numbers indicate reaching the target. If the target set
                is not specified, return None.
        """
        l_x_list = []

        # target_set_safety_margin
        for _, target_set in enumerate(self.target_x_y_w_h):
            l_x = calculate_margin_rect(s, target_set, negativeInside=True)
            # if l_x < 0:
            #     l_x = 5 * l_x
            l_x_list.append(l_x)

        target_margin = np.max(np.array(l_x_list))

        # return np.clip(self.scaling * target_margin, -self.scaling, self.scaling)
        return self.scaling * target_margin

    # == Getting Information ==
    def check_within_env(self, state):
        """Checks if the robot is still in the environment.

        Args:
            state (np.ndarray): the state of the agent.

        Returns:
            bool: True if the agent is not in the environment.
        """
        outsideTop = (state[1] >= self.bounds[1, 1])
        outsideLeft = (state[0] <= self.bounds[0, 0])
        outsideRight = (state[0] >= self.bounds[0, 1])
        return outsideTop or outsideLeft or outsideRight
        
    # == Dynamics ==
    def step(self, action):
        """Evolves the environment one step forward under given action.

        Args:
            action (int): the index of the action in the action set.

        Returns:
            np.ndarray: next state.
            float: the standard cost used in reinforcement learning.
            bool: True if the episode is terminated.
            dict: consist of target margin and safety margin at the new state.
        """

        action, disturbance = action # unpack control and disturbance
        action = action * self.act_bound
        disturbance = disturbance * self.d_bound
        u_tot = action + disturbance
        state, [l_x, g_x] = self.integrate_forward(self.state, u_tot)
        self.state = state

        fail = g_x > 0
        success = l_x <= 0

        # = `cost` signal
        if self.mode == 'RA' or self.mode == "AARA":
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                cost = 0.0 #l_x + g_x
        else:
            if fail:
                cost = self.penalty
            elif success:
                cost = self.reward
            else:
                if self.costType == 'dense_ell':
                    cost = l_x
                elif self.costType == 'dense':
                    cost = l_x + g_x
                elif self.costType == 'sparse':
                    cost = 0. * self.scaling
                elif self.costType == 'max_ell_g':
                    cost = max(l_x, g_x)
                else:
                    raise ValueError("invalid cost type!")

        
        done = fail or success
        # done = fail

        # if success: print(f"""
        # State:   {self.state}
        # l_x:     {l_x}
        # g_x:     {g_x}""")

        # if done: print(f"""
        # State:   {self.state}
        # l_x:     {l_x}
        # g_x:     {g_x}""")

        # = `info`
        info = {"g_x": g_x, "l_x": l_x}

        truncated = False # don't use truncated but added for Gym API

        return np.asarray(self.state, dtype=np.float32), cost, done, truncated, info

    def integrate_forward(self, state, u):
        """Integrates the dynamics forward by one step.

        Args:
            state (np.ndarray): x, y - position
                                [z]  - optional, extra state dimension
                                    capturing reach-avoid outcome so far)
            u (np.ndarray): contol inputs, consisting of v_x and v_y

        Returns:
            np.ndarray: next state.
        """
        # x, y, theta = state
        x, y = state

        # one step forward
        # x_dot = x_dot + self.time_step * u[0]
        # y_dot = y_dot + self.time_step * u[1]
        # theta = theta + self.time_step * u[1]
        x = x + self.time_step * u[0]
        y = y + self.time_step * u[1]

        l_x = self.target_margin(np.array([x, y]))
        g_x = self.safety_margin(np.array([x, y]))

        # state = np.array([x, y, theta])
        state = np.array([x, y])

        info = np.array([l_x, g_x])

        return state, info


    def get_axes(self):
        """Gets the axes bounds and aspect_ratio.

        Returns:
            np.ndarray: axes bounds.
            float: aspect ratio.
        """
        x_span = self.bounds[0, 1] - self.bounds[0, 0]
        y_span = self.bounds[1, 1] - self.bounds[1, 0]
        aspect_ratio = x_span / y_span
        axes = np.array([
            self.bounds[0, 0] - .05, self.bounds[0, 1] + .05,
            self.bounds[1, 0] - .05, self.bounds[1, 1] + .05
        ])
        return [axes, aspect_ratio]

    def get_value(self, sacAgent, nx=71, ny=121, addBias=False):
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

            if self.mode == 'normal' or self.mode == 'RA' or self.mode == 'AARA' and sacAgent is None:
                state = torch.FloatTensor([x, y]).to(self.device).unsqueeze(0)
            elif self.mode == 'AARA' and sacAgent is not None:
                state = torch.FloatTensor([x, y]).to(self.device).unsqueeze(0)
                _, _, action = sacAgent.protagonist.sample(state)
                _, _, disturbance = sacAgent.adversary.sample(state)

            if addBias:
                qf1, qf2 = sacAgent.critic(state, action, disturbance)
                value = torch.min(qf1, qf2)
                v[idx] = value + max(l_x, g_x)
            else:
                qf1, qf2 = sacAgent.critic(state, action, disturbance)
                value = torch.min(qf1, qf2)
                v[idx] = value 
            it.iternext()
        return xs, ys, v

    # == Trajectory Functions ==
    def simulate_one_trajectory(
        self, sacAgent, T=250, state=None, keepOutOf=False, toEnd=False
    ):
        """Simulates the trajectory given the state or randomly initialized.

        Args:
            q_func (object): agent's Q-network.
            T (int, optional): the maximum length of the trajectory.
                Defaults to 250.
            state (np.ndarray, optional): if provided, set the initial state to its
                value. Defaults to None.
            keepOutOf (bool, optional): smaple states inside obstacles if False.
                Defaults to False.
            toEnd (bool, optional): simulate the trajectory until the robot
                crosses the boundary if True. Defaults to False.

        Returns:
            np.ndarray: x-positions of the trajectory.
            np.ndarray: y-positions of the trajectory.
            int: the binary reach-avoid outcome.
        """
        protagonist = sacAgent.protagonist
        adversary = sacAgent.adversary
        if state is None:
            state = self.sample_random_state(sample_inside_obs=not keepOutOf)
        state_traj = [state]
        control_traj = []
        traj_val = [max(self.target_margin(state[:2]), self.safety_margin(state[:2]))]
        result = 0  # not finished

        for _ in range(T):
            if self.safety_margin(state[:2]) > 0: 
                result = -1  # failed
                break
            elif self.target_margin(state[:2]) <= 0:
                result = 1  # succeeded
                break

            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.to(self.device).unsqueeze(0)

            _, _, action =  protagonist.sample(state_tensor) # deterministic action
            _, _, disturb = adversary.sample(state_tensor) # deterministic disturbance

            action = action.cpu().detach().numpy()[0] * self.act_bound
            disturb = disturb.cpu().detach().numpy()[0] * self.d_bound

            u_tot = action + disturb

            l_x = self.target_margin(state[:2])
            g_x = self.safety_margin(state[:2])
            value = max(l_x, g_x)
            traj_val.append(value)

            state, _ = self.integrate_forward(state, u_tot)

            state_traj.append(state)
            control_traj.append(action)

        return np.array(state_traj), np.array(result), np.array(traj_val), np.array(control_traj)

    def simulate_trajectories(
        self, sacAgent, T=250, num_rnd_traj=None, states=None, toEnd=False
    ):
        """
        Simulates the trajectories. If the states are not provided, we pick the
        initial states from the discretized state space.

        Args:
            q_func (object): agent's Q-network.
            T (int, optional): the maximum length of the trajectory.
                Defaults to 250.
            num_rnd_traj (int, optional): #states. Defaults to None.
            states (list of np.ndarrays, optional): if provided, set the initial
                states to its value. Defaults to None.
            toEnd (bool, optional): simulate the trajectory until the robot crosses
                the boundary if True. Defaults to False.

        Returns:
            list of np.ndarrays: each element is a tuple consisting of x and y
                positions along the trajectory.
            np.ndarray: the binary reach-avoid outcomes.
        """

        assert ((num_rnd_traj is None and states is not None)
                or (num_rnd_traj is not None and states is None)
                or (len(states) == num_rnd_traj))
        trajectories = []
        state_hist = []
        control_hist = []
        values = []

        if states is None:
            nx = 41
            ny = 100
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
                # state = np.array([x, y, 0])
                state = np.array([x, y])
                state_traj, result, traj_val, control_traj = self.simulate_one_trajectory(
                    sacAgent, T=T, state=state, toEnd=toEnd
                )
                values.append(traj_val)
                state_hist.append(state_traj)
                control_hist.append(control_traj)
                results[idx] = result
                it.iternext()
            results = results.reshape(-1)
        else:
            results = np.empty(shape=(len(states),), dtype=int)
            for idx, state in enumerate(states):
                state_traj, result, traj_val, control_traj = self.simulate_one_trajectory(
                    sacAgent, T=T, state=state, toEnd=toEnd
                )
                print(len(state_traj), idx)
                values.append(traj_val)
                state_hist.append(state_traj)
                control_hist.append(control_traj)
                results[idx] = result

        return state_hist, results, values, control_hist

    # == Visualizing ==
    def render(self):
        pass

    def visualize(
        self, sacAgent, vmin=-1, vmax=1, nx=201, ny=201, labels=None,
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
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        cbarPlot = True

        # == Plot failure / target set ==
        self.plot_target_failure_set(ax[0])

        # # == Plot reach-avoid set ==
        # self.plot_reach_avoid_set(ax[0])

        # # == Plot V ==
        self.plot_v_values(
            sacAgent, ax=ax[0], fig=fig, vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
            boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias
        )

        # == Plot Trajectories ==
        self.plot_trajectories(
            sacAgent, T=self.config.maxSteps, states=self.visual_initial_states, toEnd=False, ax=ax[0]
        )


        # == Formatting ==
        # self.plot_formatting(ax=ax[0], labels=labels)
        # fig.tight_layout()

    def plot_v_values(
        self, sacAgent, ax=None, fig=None, vmin=-1, vmax=1, nx=201, ny=201,
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
        _, _, v = self.get_value(sacAgent, nx, ny, addBias=addBias)
        # vmax = np.ceil(max(np.max(v), np.max(-v)))
        # vmin = -vmax

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)

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
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2, vmax=vmax)
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, cax=cax, ticks=[vmin, 0, vmax], pad = 0.15, location='right')
                # cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)
        
        ax.set_title(r'Global $\hat{V}(s)$', fontsize=16)
        
    def plot_trajectories(
        self, sacAgent, T=300, num_rnd_traj=None, states=None, toEnd=False,
        ax=None, c='k', lw=2, zorder=2
    ):
        """Plots trajectories given the agent's Q-network.

        Args:
            q_func (object): agent's Q-network.
            T (int, optional): the maximum length of the trajectory.
                Defaults to 250.
            num_rnd_traj (int, optional): #states. Defaults to None.
            states (list of np.ndarrays, optional): if provided, set the initial
                states to its value. Defaults to None.
            toEnd (bool, optional): simulate the trajectory until the robot crosses
                the boundary if True. Defaults to False.
            ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
            c (str, optional): color of the trajectories. Defaults to 'k'.
            lw (float, optional): linewidth of the trajectories. Defaults to 2.
            zorder (int, optional): graph layers order. Defaults to 2.
        Returns:
            np.ndarray: the binary reach-avoid outcomes.
        """

        assert ((num_rnd_traj is None and states is not None)
                or (num_rnd_traj is not None and states is None)
                or (len(states) == num_rnd_traj))

        state_hist, results, values, control_hist = self.simulate_trajectories(
            sacAgent, T=T, num_rnd_traj=num_rnd_traj, states=states, toEnd=toEnd
        )

        # Build a shared normalizer across all trajectories so colors are comparable
        all_vals = np.concatenate(values)
        vmin_traj = all_vals.min()
        vmax_traj = all_vals.max()
        center_traj = (vmax_traj + vmin_traj) / 2.0 
        norm      = mcolors.TwoSlopeNorm(vmin=vmin_traj - 1e-6, vcenter=center_traj, vmax=vmax_traj + 1e-6)
        cmap      = cm.get_cmap('RdYlGn_r')

        # for traj in trajectories:
        #   traj_x, traj_y = traj
        #   ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
        #   ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

        for traj, traj_vals in zip(state_hist, values):
            traj = np.array(traj) 
            traj_x, traj_y = traj[:, :2].T

            # Build (N-1) segments: each segment is [[x0,y0],[x1,y1]]
            points   = np.array([traj_x, traj_y]).T                  # (N, 2)
            segments = np.stack([points[:-1], points[1:]], axis=1)   # (N-1, 2, 2)

            # Color each segment by the value at its start point
            seg_vals = traj_vals[:-1]

            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder)
            lc.set_array(seg_vals)
            ax.add_collection(lc)

            # Start point marker colored by initial value
            ax.scatter(traj_x[0], traj_y[0], s=48, zorder=zorder,
                        color=cmap(norm(traj_vals[0])))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="5%", pad=0.3)

        # Shared colorbar — add once to the figure
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, cax=cax, label=r'$\hat{V}(s)$ along trajectory', pad=0.15, fraction=0.05, shrink=0.9, location='left')

        # axes = ax[1]

        # colours = ['blue', 'orange', 'pink', 'green', 'red']

        # for i, (traj, states) in enumerate(zip(control_hist, state_hist)):

        #     # handle more trajectories than colors
        #     color = colours[i % len(colours)]

        #     # traj: (T_i, control_dim)
        #     # states: (T_i, state_dim)

        #     # T = traj.shape[0]
        #     # t = np.arange(T)

        #     # --- plot controls ---
        #     for j in range(traj.shape[1]):  # control dimension
        #         linestyle = "--" if j == 0 else "-"
        #         axes.plot(traj[:, j],
        #                 linestyle=linestyle,
        #                 color=color,
        #                 label=f"traj {i} u{j}" if i == 0 else None)

        #     # --- plot states ---
        #     for j in range(states.shape[1]):  # state dimension
        #         linestyle = "--" if j == 0 else "-"
        #         ax[2].plot(states[:, j],
        #                 linestyle=linestyle,
        #                 label=f"traj {i} x{j}" if i == 0 else None)

        # axes.set_xlabel("time")
        # axes.set_ylabel("control")
        # axes.set_title("control history")
        # axes.legend()

        # ax[2].set_xlabel("time")
        # ax[2].set_ylabel("states")
        # ax[2].set_title("state history")
        # ax[2].legend()

        return results

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
        # Plot constraint set (obstacles).
        theta = np.linspace(0, 2 * np.pi, 200)

        for (center, radius) in self.obstacles:
            cx, cy = center
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            ax.plot(x, y, color=c_c, lw=lw, zorder=zorder)

        # Plot boundaries of target set.
        for one_boundary in self.target_set_boundary:
            ax.plot(
                one_boundary[:, 0], one_boundary[:, 1], color=c_t, lw=lw,
                zorder=zorder
            )

    # def plot_reach_avoid_set(self, ax=None, c='c', lw=3, zorder=1):
    #     """Plots the analytic reach-avoid set.

    #     Args:
    #         ax (matplotlib.axes.Axes, optional): ax to plot. Defaults to None.
    #         c (str, optional): color of the rach-avoid set boundary.
    #             Defaults to 'g'.
    #         lw (int, optional): liewidth. Defaults to 3.
    #         zorder (int, optional): graph layers order. Defaults to 1.
    #     """

    #     def get_line(slope, end_point, x_limit, ns=100):
    #         x_end, y_end = end_point
    #         b = y_end - slope*x_end

    #         xs = np.linspace(x_limit, x_end, ns)
    #         ys = xs*slope + b
    #         return xs, ys

    #     # unsafe set
    #     for cons, cType in zip(self.constraint_x_y_w_h, self.constraint_type):
    #         x, y, w, h = cons
    #         x1 = x - w/2.0
    #         x2 = x + w/2.0
    #         y_min = y - h/2.0
    #         if cType == 'C':
    #             # for max Reach-Avoid Set (worst case disturbance)
    #             xs, ys = get_line(-r_slope_max, end_point=[x1, y_min], x_limit=x)
    #             ax.plot(xs, ys, '--', color=c, linewidth=lw, zorder=zorder)
    #             xs, ys = get_line(l_slope_max, end_point=[x2, y_min], x_limit=x)
    #             ax.plot(xs, ys, '--', color=c, linewidth=lw, zorder=zorder)
    #             # for min Reach-Avoid Set (no disturbance)
    #             xs, ys = get_line(-slope_min, end_point=[x1, y_min], x_limit=x)
    #             ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
    #             xs, ys = get_line(slope_min, end_point=[x2, y_min], x_limit=x)
    #             ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
    #         elif cType == 'L':
    #             # for max Reach-Avoid Set (worst case disturbance)
    #             x_limit = self.bounds[0, 0]
    #             xs, ys = get_line(l_slope_max, end_point=[x2, y_min], x_limit=x_limit)
    #             ax.plot(xs, ys, '--', color=c, linewidth=lw, zorder=zorder)
    #             # for min Reach-Avoid Set (no disturbance)
    #             xs, ys = get_line(slope_min, end_point=[x2, y_min], x_limit=x_limit)
    #             ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
    #         elif cType == 'R':
    #             # for max Reach-Avoid Set (worst case disturbance)
    #             x_limit = self.bounds[0, 1]
    #             xs, ys = get_line(-r_slope_max, end_point=[x1, y_min], x_limit=x_limit)
    #             ax.plot(xs, ys, '--', color=c, linewidth=lw, zorder=zorder)
    #             # for min Reach-Avoid Set (no disturbance)
    #             xs, ys = get_line(-slope_min, end_point=[x1, y_min], x_limit=x_limit)
    #             ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

    #     # border unsafe set
    #     x, y, w, h = self.target_x_y_w_h[0]
    #     x1 = x - w/2.0
    #     x2 = x + w/2.0
    #     y_max = y + h/2.0
    #     # for max Reach-Avoid Set (worst case disturbance)
    #     xs, ys = get_line(l_slope_max, end_point=[x1, y_max], x_limit=self.low[0])
    #     ax.plot(xs, ys, '--', color=c, linewidth=lw, zorder=zorder)
    #     xs, ys = get_line(-r_slope_max, end_point=[x2, y_max], x_limit=self.high[0])
    #     ax.plot(xs, ys, '--', color=c, linewidth=lw, zorder=zorder, label='Max boundary')
    #     # for min Reach-Avoid Set (no disturbance)
    #     xs, ys = get_line(slope_min, end_point=[x1, y_max], x_limit=self.low[0])
    #     ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder, label='Min boundary')
    #     xs, ys = get_line(-slope_min, end_point=[x2, y_max], x_limit=self.high[0])
    #     ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

    #     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3))

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