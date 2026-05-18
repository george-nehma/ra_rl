"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements a Sin activation function and neural network model as
torch.nn.Module. Also, it implements a scheduler for hyper-parameters.
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_SIG_MAX = 1
LOG_SIG_MIN = -5
epsilon = 1e-6

## ==== DDQN Networks ====

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Sin(nn.Module):
  """An element-wise sin activation wrapped as a nn.Module.

  Shape:
      - Input: `(N, *)` where `*` means, any number of additional dimensions
      - Output: `(N, *)`, same shape as the input

  Examples:
      >>> m = Sin()
      >>> input = torch.randn(2)
      >>> output = m(input)
  """

  def forward(self, input):
    return torch.sin(input)  # simply apply already implemented sin


class Model(nn.Module):
  """
  Constructs a fully-connected neural network with flexible depth, width and
  activation function choices.
  """

  def __init__(
      self, dimList, actType="Tanh", output_activation=nn.Identity,
      verbose=False
  ):
    """
    Initalizes the neural network with dimension of each layer and the
    following activation layer.

    Args:
        dimList (List): the dimension of each layer.
        actType (str, optional): the type of activation function. Defaults to
            'Tanh'. Currently supports 'Sin', 'Tanh' and 'ReLU'.
        verbose (bool, optional):print messages if True. Defaults to False.
    """
    super(Model, self).__init__()

    # Construct module list: if use `Python List`, the modules are not
    # added to computation graph. Instead, we should use `nn.ModuleList()`.
    self.moduleList = nn.ModuleList()
    numLayer = len(dimList) - 1
    for idx in range(numLayer):
      i_dim = dimList[idx]
      o_dim = dimList[idx + 1]

      self.moduleList.append(nn.Linear(in_features=i_dim, out_features=o_dim))
      if idx == numLayer - 1:  # final linear layer, no act.
        self.moduleList.append(output_activation())
      else:
        if actType == "Sin":
          self.moduleList.append(Sin())
        elif actType == "Tanh":
          self.moduleList.append(nn.Tanh())
        elif actType == "ReLU":
          self.moduleList.append(nn.ReLU())
        else:
          raise ValueError(
              "Activation type ({:s}) is not included!".format(actType)
          )
    
    if verbose:
      print(self.moduleList)

  def forward(self, x):
    for m in self.moduleList:
      x = m(x)
    return x


# == Scheduler ==
class _scheduler(abc.ABC):
  """
  The parent class for schedulers. It implements some basic functions that will
  be used in all scheduler.
  """

  def __init__(self, last_epoch=-1, verbose=False):
    """Initializes the scheduler with the index of last epoch.
    """
    self.cnt = last_epoch
    self.verbose = verbose
    self.variable = None
    self.step()

  def step(self):
    """Updates the index of the last epoch and the variable.
    """
    self.cnt += 1
    value = self.get_value()
    self.variable = value

  @abc.abstractmethod
  def get_value(self):
    raise NotImplementedError

  def get_variable(self):
    """Returns the variable.
    """
    return self.variable


class StepLR(_scheduler):
  """This scheduler will decay to end value periodically.
  """

  def __init__(
      self, initValue, period, decay=0.1, endValue=0., last_epoch=-1,
      verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        decay (float, optional): the amount by which the variable decays.
            Defaults to 0.1.
        endValue (float, optional): the target value to decay to.
            Defaults to 0.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    super(StepLR, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable.
    """
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    tmpValue = self.initValue * (self.decay**numDecay)
    if self.endValue is not None and tmpValue <= self.endValue:
      return self.endValue
    return tmpValue


class StepLRMargin(_scheduler):

  def __init__(
      self, initValue, period, goalValue, decay=0.1, endValue=1, last_epoch=-1,
      verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        goalValue (float):the target value to anneal to.
        decay (float, optional): the amount by which the margin between the
            variable and the goal value decays. Defaults to 0.1.
        endValue (float, optional): the maximum value of the variable.
            Defaults to 1.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    self.goalValue = goalValue
    super(StepLRMargin, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable.
    """
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    #if numDecay > 0:
    #  print('gamma update')
    tmpValue = self.goalValue - (self.goalValue
                                 - self.initValue) * (self.decay**numDecay)
    if self.endValue is not None and tmpValue >= self.endValue:
      return self.endValue
    return tmpValue


class StepResetLR(_scheduler):

  def __init__(
      self, initValue, period, resetPeriod, decay=0.1, endValue=0,
      last_epoch=-1, verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        resetPeriod (int): the period to reset the variable to its initial
            value.
        decay (float, optional): the amount by which the variable decays.
            Defaults to 0.1.
        endValue (float, optional): the target value to decay to.
            Defaults to 0.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    self.resetPeriod = resetPeriod
    super(StepResetLR, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable.
    """
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    tmpValue = self.initValue * (self.decay**numDecay)
    if self.endValue is not None and tmpValue <= self.endValue:
      return self.endValue
    return tmpValue

  def step(self):
    """
    Updates the index of the last epoch and the variable. It overrides the same
    function in the parent class.
    """
    self.cnt += 1
    value = self.get_value()
    self.variable = value
    if (self.cnt + 1) % self.resetPeriod == 0:
      self.cnt = -1


# ==== SAC Networks ====

def build_mlp(dimList, activation):
    act_map = {
        "Sin":  Sin(),
        "Tanh": nn.Tanh(),
        "ReLU": nn.ReLU(),
    }
    if activation not in act_map:
        raise ValueError(f"Activation type ({activation}) is not included!")

    layers = []
    numLayer = len(dimList) - 1
    for idx in range(numLayer):
        layers.append(nn.Linear(dimList[idx], dimList[idx + 1]))
        if idx < numLayer - 1:          # no activation after final layer
            layers.append(act_map[activation])

    return nn.Sequential(*layers)

class ValueNetwork(nn.Module):
    def __init__(self,CONFIG, dimList):
        super(ValueNetwork, self).__init__()

        self.config = CONFIG
        self.actType = CONFIG.ACTIVATION
        sa_dimList = dimList.copy()
        sa_dimList[-1] = 1  # output is Q-value

        self.value_head = build_mlp(sa_dimList, self.actType)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.value_head(state)
        return x


class QNetwork(nn.Module):
    def __init__(self, CONFIG, dimList, numActions, numDisturb):
        super(QNetwork, self).__init__()

        self.config = CONFIG
        self.actType = CONFIG.ACTIVATION
        sa_dimList = dimList.copy()
        sa_dimList[0] = dimList[0] + numActions + numDisturb  # input is state-action pair
        sa_dimList[-1] = 1  # output is Q-value
        print("QNetwork dimList:", sa_dimList)

        self.q_head1 = build_mlp(sa_dimList, self.actType)
        self.q_head2 = build_mlp(sa_dimList, self.actType)

        self.apply(weights_init_)

    def forward(self, state, action, disturbance):
        xu = torch.cat([state, action, disturbance], 1)

        x1 = self.q_head1(xu)
        x2 = self.q_head2(xu)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, CONFIG, dimList, numActions, action_space=None, conditioned_sigma=False):
        super(GaussianPolicy, self).__init__()
        
        self.config = CONFIG
        self.actType = CONFIG.ACTIVATION
        self.c_sigma = conditioned_sigma
        print("GaussianPolicy dimList:", dimList)

        self.mu_head = build_mlp(dimList, self.actType)
        log_std_dimList = dimList.copy()
        log_std_dimList[-1] = numActions
        if self.c_sigma:
            self.log_std_head = build_mlp(log_std_dimList, self.actType)
        else:
            self.log_std_head = nn.Parameter(torch.zeros(numActions))

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        mean = self.mu_head(state)
        if self.c_sigma:
            log_std = self.log_std_head(state)
            sigma = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            shape = [1] * len(mean.shape)
            shape[1] = -1
            sigma = (self.log_std_head.view(shape) + torch.zeros_like(mean)).exp()
        return mean, sigma

    def sample(self, state):
        mean, std = self.forward(state)
        if self.c_sigma:
            std = std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= (2 * (torch.log(torch.tensor(2)) - x_t - F.softplus(-2 * x_t)))
        log_prob = log_prob.sum(-1, keepdim=True)
        # log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, CONFIG, dimList, numActions, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.config = CONFIG
        self.actType = CONFIG.ACTIVATION
        self.mu_head = build_mlp(dimList, self.actType)

        self.noise = torch.Tensor(numActions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        mean = torch.tanh(self.mu_head(state)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
    

class VectorizedLinear(nn.Module):
    """
    A linear layer that computes outputs for N critics in parallel.
    Weight shape: (num_critics, out_features, in_features)
    Bias shape: (num_critics, out_features)
    """
    def __init__(self, in_features, out_features, num_critics):
        super().__init__()
        self.num_critics = num_critics
        self.weight = nn.Parameter(torch.randn(num_critics, out_features, in_features))
        self.bias = nn.Parameter(torch.randn(num_critics, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_critics):
            nn.init.xavier_uniform_(self.weight[i], gain=1)
            nn.init.constant_(self.bias[i], 0)

    def forward(self, x):
        # x: (batch, in_features) -> (num_critics, batch, in_features)
        x_expanded = x.unsqueeze(0).expand(self.num_critics, -1, -1)
        # res: (num_critics, batch, out_features)
        res = torch.matmul(x_expanded, self.weight.transpose(-1, -2))
        return res + self.bias.unsqueeze(1)

class VectorizedMLP(nn.Module):
    """
    A multi-layer perceptron that computes outputs for N critics in parallel.
    """
    def __init__(self, dimList, actType="Tanh", num_critics=1, output_activation=nn.Identity):
        super().__init__()
        self.num_critics = num_critics
        self.moduleList = nn.ModuleList()
        numLayer = len(dimList) - 1
        for idx in range(numLayer):
            i_dim = dimList[idx]
            o_dim = dimList[idx + 1]
            self.moduleList.append(VectorizedLinear(i_dim, o_dim, num_critics))
            if idx < numLayer - 1:
                if actType == "Sin":
                    self.moduleList.append(Sin())
                elif actType == "Tanh":
                    self.moduleList.append(nn.Tanh())
                elif actType == "ReLU":
                    self.moduleList.append(nn.ReLU())
                else:
                    raise ValueError(f"Activation type {actType} not supported!")

        self.output_act = output_activation()

    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return self.output_act(x)